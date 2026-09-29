# dft/ — DFT (PySCF) による標準生成エンタルピー推算

Streamlit アプリの外で動かす、Vertex AI 用のバッチ計算です。SMILES を書いた YAML を渡すと、
配座探索 → xTB 構造最適化 → xTB の振動解析 → DFT 一点計算 を行い、気相 ΔHf°(298 K) を出します。
結果は GCS に保存され、フリー体の値はアプリの「ΔHf° 手動入力」にそのまま貼れます。

## 何を計算するか

| 入力 | 出力 |
|------|------|
| 中性の単一分子 | 気相 ΔHf° |
| 溶媒和物・水和物（`.` 区切りで中性の成分） | 複合体の気相 ΔHf°、会合エンタルピー、フリー体の ΔHf° |
| 塩（電荷を持つ成分） | イオン対の気相 ΔHf°、フリー体（中和した分子）の ΔHf° |

アプリの反応熱計算で使うのは**フリー体**の値です（`app_input` にまとめて出力します）。
塩・溶媒和物そのものの値を反応に使いたい場合は、結果の `whole` を手動入力してください。

塩の固体の ΔHf° は、気相のイオン対の値から「固体 → 気相イオン対」のエンタルピーを引いて求めます。
この値は DFT の一点計算では出ないので、実測値か周期系の計算値を `sublimation_enthalpy_kJ_mol` で与えてください。
与えなければ気相の値のみを出力し、固体の値とは書きません。

## 手法

```
ΔHf°(298) = E_DFT + H_corr + Σ n_e·c_e + Σ m_b·d_b
```

- `E_DFT`: PySCF の DFT 一点計算（既定は B3LYP-D3(BJ)/def2-SVP、密度フィッティング）
- `H_corr`: xTB (GFN2) の構造・ヘッセ行列から、調和振動・剛体回転・理想気体近似で求めた H(298) − E
- `c_e`（元素）, `d_b`（結合タイプ）: ΔHf° が既知の小分子 `reference_set.csv` への最小二乗フィット（結合項のみリッジ正則化）。
  DFT の元素ごと・結合ごとの系統誤差を吸収する。フィットは `calibrate` ジョブで行い、結果を JSON に保存する。
- 不確かさ: 校正の leave-one-out RMSE を、参照分子より大きい分だけ √(重原子数比) で拡大した目安。

精度の限界:
- 参照分子は重原子 14 個以下です。それより大きい分子は外挿になり、表示される不確かさより実際の誤差が大きくなり得ます。
- 柔軟な大分子では、配座探索の不足が誤差の主因になります。配座ごとのエネルギー差は数十 kJ/mol あります。
- 反応熱を知りたいだけなら、反応中心を含む小さなモデル化合物の値で足りることが多く、その方が精度も費用も有利です。

## 使い方

### 1. 見積もり（ローカル、重い依存なしで動く）

```bash
cd dft
python -m hf_dft estimate jobs/example.yaml --machine n2-highmem-32
```

### 2. イメージのビルド

```bash
gcloud builds submit dft \
  --tag asia-northeast1-docker.pkg.dev/<PROJECT>/hf-dft/hf-dft:latest
```

### 3. 校正ジョブ（計算レベルごとに 1 回）

```bash
python dft/submit_vertex.py calibrate dft/jobs/calibrate.yaml \
  --project <PROJECT> --bucket gs://<BUCKET>/hf-dft \
  --image asia-northeast1-docker.pkg.dev/<PROJECT>/hf-dft/hf-dft:latest \
  --machine-type n2-standard-32 --spot
```

校正ファイルは `gs://<BUCKET>/hf-dft/calibration/calibration_<xc>_<basis>_<disp>.json` に出力されます。
出力される RMSE(LOO) と外れ値の一覧（`rejected`）を必ず確認してください。

### 4. 推算ジョブ

`jobs/example.yaml` の `calibration:` に上の JSON を指定して投入します。

```bash
python dft/submit_vertex.py run dft/jobs/example.yaml \
  --project <PROJECT> --bucket gs://<BUCKET>/hf-dft \
  --image asia-northeast1-docker.pkg.dev/<PROJECT>/hf-dft/hf-dft:latest \
  --machine-type n2-highmem-32 --spot
```

`--dry-run` を付けると、送信せずに投入内容と見積もりだけを表示します。

Spot VM が中断されると Vertex AI が自動で再実行します。`results.json` を化合物ごとに書き出しているので、
再実行では計算済みの化合物を飛ばします。

### 出力（`gs://<BUCKET>/hf-dft/<job_name>/<日時>/out/`）

| ファイル | 内容 |
|----------|------|
| `results.json` | 全結果。エネルギー、補正、不確かさ、警告、所要時間 |
| `results.csv` | 一覧表 |
| `structures/*.xyz` | xTB で最適化した構造 |

`results.json` の各化合物の `app_input` が、アプリの手動入力欄に入れる値です。

| キー | アプリの入力欄 |
|------|----------------|
| `smiles` | SMILES（フリー体） |
| `hf_kJ_mol` | ΔHf° 手動入力 |
| `uncertainty_kJ` | 不確かさ ± |
| `note` | 出典メモ |

## 入力 YAML

`jobs/example.yaml` を参照。主な項目:

| キー | 内容 |
|------|------|
| `level` | `xc`, `basis`, `disp`, `xtb_method`, `temperature_K`。校正時と同じにする（違うと実行時にエラー） |
| `sampling` | `n_conformers`（RDKit の配座数）, `n_keep`（xTB で最適化する数）, `n_orient`（複合体の初期配置数） |
| `resources` | `threads`, `max_memory_gb` |
| `calibration` | 校正 JSON のパス（`run` で必須） |
| `compounds[].id / smiles` | 必須。id は一意 |
| `compounds[].kind` | `neutral` / `salt` / `solvate`。省略時は SMILES から自動判定 |
| `compounds[].free_form_smiles` | フリー体。省略時は最大成分を中和したもの |
| `compounds[].charge / multiplicity` | 電荷・多重度の上書き（省略時は SMILES から） |
| `compounds[].sublimation_enthalpy_kJ_mol` | 固体 → 気相複合体のエンタルピー（あれば固体の値を出す） |

## ファイル

```
hf_dft/            本体 (species, conformers, engines, thermo, pipeline, runner, cost, storage)
jobs/              YAML の例
tests/             PySCF なしで動く単体テスト (python tests/test_hf_dft.py)
reference_set.csv  校正用の参照分子 (build_reference_set.py で再生成)
submit_vertex.py   Vertex AI へのジョブ投入
Dockerfile         実行イメージ
```
