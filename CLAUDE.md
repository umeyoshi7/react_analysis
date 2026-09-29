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

- ΔHf° はすべて理想気体の値。液相・固相の補正（蒸発熱・融解熱）は未実装。
- `data/hf_gas.csv` の値は一次資料との照合が済んでいない（出典列に「要照合」と記載）。安全評価に使う前に NIST WebBook などで確認する。
- Joback 法には硫黄系の基団がなく、その場合の Cp は推算できず ΔCp=0 として扱う。

## 依存パッケージ

```
streamlit==1.55.0
pandas==2.3.3
numpy==2.4.3
plotly==6.6.0
rdkit==2026.3.1
ugropy==3.1.6
```
