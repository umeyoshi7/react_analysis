# react_analysis

反応熱推算 Streamlit アプリ。SMILES 入力で反応スキームを可視化し、ΔH_rxn を推算する。

## ローカル起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ファイル構成

```
app.py               # Streamlit エントリーポイント (シングルページ)
requirements.txt
src/
  __init__.py
  thermochemistry.py # 反応熱推算 (Joback法 + 文献値 + SVG生成 + 温度/溶媒補正)
```

## 機能概要

| 機能 | 内容 |
|------|------|
| 反応スキーム可視化 | SMILES → RDKit 2D 構造式 + → 記号で反応物/生成物を表示 |
| ΔH_rxn 推算 | Joback 基団寄与法 / 文献値 / 手動入力でΔHf°を取得し反応熱を計算 |
| 反応テンプレート | 付加・置換・脱離・環化・燃焼・縮合など8種類をプリセット |
| 温度補正 | Kirchhoff 則 ΔH(T) ≈ ΔH°(298K) + ΔCp(T-298.15) |
| 溶媒補正 | 溶媒選択(誘電率表示) + 手動補正値入力 |
| エネルギー図 | Plotly によるエンタルピーレベル図 |

## 依存パッケージ

```
streamlit==1.55.0
pandas==2.3.3
numpy==2.4.3
plotly==6.6.0
rdkit==2026.3.1
ugropy==3.1.6
```

