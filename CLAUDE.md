# react_analysis

反応熱推算 Streamlit アプリ。SMILES を入力すると反応スキームを描画し、ΔH_rxn を推算する。あわせて、推算した反応熱から暴走反応のリスク（MTSR / TD24 / Stoessel）を評価する。

## ローカル起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker / デプロイ

```bash
docker build -t react-analysis .
docker run -p 8080:8080 react-analysis
```

Cloud Run (asia-northeast1) にデプロイする想定で、ポートは 8080。イメージには `app.py`、`src/`、`data/` だけを入れる（`.dockerignore` で他は除外）。

## ファイル構成

```
app.py                     # Streamlit エントリーポイント (シングルページ)
requirements.txt
Dockerfile / .dockerignore
src/
  thermochemistry.py       # ΔHf° 取得、反応熱計算、Cp / 温度補正、SVG 生成、プロセス安全評価
data/
  hf_gas.csv               # 文献値 ΔHf° (気体, 298.15 K)。InChIKey で照合する
scripts/
  build_hf_db.py           # hf_gas.csv の生成スクリプト
dft/                       # Streamlit とは別の、DFT による ΔHf° 推算 (Vertex AI ジョブ)。詳細は dft/README.md
  hf_dft/                  # 本体 (配座 → xTB → PySCF → 校正)
  submit_vertex.py         # ジョブ投入
  jobs/                    # 入力 YAML の例
  tests/                   # PySCF なしで動く単体テスト
  reference_set.csv        # 校正用の参照分子
```

`data/hf_gas.csv` を編集するときは、CSV を直接書き換えず `scripts/build_hf_db.py` の `ROWS` を修正して再実行する。SMILES から InChIKey を付与しているため。

## ΔHf° の取得順序

`get_hf()` は次の順に試し、最初に値が得られた手法を採用する。

| 順 | 手法 | 不確かさ (kJ/mol) |
|----|------|-------------------|
| 1 | 手動入力 | 0 |
| 2 | 組込みの文献値表 (`_KNOWN_HF`, SMILES 照合) | 1 |
| 3 | `data/hf_gas.csv` (InChIKey 照合) | CSV の値（化合物ごと） |
| 4 | Gani 基団寄与法 (ugropy) | 10 |
| 5 | Joback 基団寄与法 (ugropy) | 25 |

- ΔH_rxn の不確かさは、各化合物の不確かさ × 係数の二乗和平方根で求める。
- 3〜4 員環を含む化合物を基団寄与法で推算した場合は、環ひずみが入らないため警告を出す。
- 文献値と推算値が混在する場合も警告する。
- 立体異性体だけが CSV と異なる場合は、接続層が一意に一致するときに限って値を流用し、警告を出す。

## 塩・溶媒和物・イオン

基団寄与法は中性の単一分子だけが対象。SMILES に `.` が含まれるか、正味の電荷がある場合は、
文献値表・CSV に載っていなければ推算せず、フリー体の SMILES を使うか手動入力するよう案内する。
反応熱の計算に使うのはフリー体の ΔHf°。塩や溶媒和物そのものの値は `dft/` で求められる。

手動入力には、値のほかに不確かさ（既定 ±10 kJ/mol）と出典メモを付けられる。不確かさは ΔH_rxn の
不確かさに二乗和平方根で加わる。`calc_reaction_heat()` には `ManualHf(value, uncertainty_kJ, note)` で渡す
（float を渡した場合は不確かさ 0 として扱う）。

## 液相補正

化合物ごとに相（気体 / 液体）を選べる。液体を選ぶと `get_hf(..., phase="l")` が気相 ΔHf° から 298.15 K の蒸発エンタルピーを引く。

- ΔHvap は、組込み表 → `data/hf_gas.csv` の `hvap_kJ_mol` 列 → Joback 法の ΔHvap(Tb) を Watson 式で 298 K に換算、の順で取得する。推算値の不確かさは max(4, 15%) kJ/mol。
- ΔHvap が得られない化合物（常温で気体の物質など）は、液相値を求められない。手動入力する。
- 手動入力値は選んだ相の値としてそのまま使い、相補正はしない。
- Kirchhoff 補正は気相 Cp のままなので、液相を含む場合は警告を出す。

## Cp の取得

`get_cp_estimate()` は、文献値表 → Joback 法 → 原子数による概算、の順に試す。Joback 法に基団がない化合物（硫黄系など）は、文献 Cp 表 25 件への回帰式 `Cp ≈ 19.6 + 6.27 × 原子数(H 含む)` で概算する。独立データ（Poling, 344 化合物）と比べた平均誤差は 11%、誤差の大きい上位 10% は約 30%。硫黄化合物 14 件（チオール、スルフィド、SO₂ など）の平均誤差は 8%。この場合は温度補正に警告が付く。

## 機能概要

| 機能 | 内容 |
|------|------|
| 反応スキーム可視化 | SMILES → RDKit 2D 構造式 + → 記号で反応物/生成物を表示 |
| ΔH_rxn 推算 | 上表の順に ΔHf° を取得して反応熱を計算し、不確かさも表示 |
| 反応テンプレート | 付加・置換・脱離・環化・燃焼・縮合など 8 種類をプリセット |
| 温度補正 | Kirchhoff 則 ΔH(T) ≈ ΔH°(298K) + ΔCp(T-298.15)。Cp は文献値表、なければ Joback 法 |
| 溶媒補正 | 溶媒選択(誘電率表示) + 手動補正値入力 |
| エネルギー図 | Plotly によるエンタルピーレベル図 |
| プロセス安全評価 | MTSR / TD24 の算出と Stoessel 5 段階のクラス分け |

## 既知の制限

- 相の補正は気体 → 液体（蒸発熱）のみ。固相は扱わない。
- `data/hf_gas.csv` の ΔHf° と ΔHvap は、`chemicals` パッケージ内の ATcT / API TDB / Yaws / CRC の値と照合した（2026-09-30）。出典間で食い違った 6 件は不確かさを広げてあり、出典列に「要確認」と記載している。
- Cp の概算式は原子数だけを使う粗い式。スルホニルクロリドなど、照合データにない構造は精度が未確認（SF₄ のように 30% 近く外れる例もある）。必要なら文献値を `_CP_RAW` に追加する。

## 依存パッケージ

```
streamlit==1.55.0
pandas==2.3.3
numpy==2.4.3
plotly==6.6.0
rdkit==2026.3.1
ugropy==3.1.6
```
