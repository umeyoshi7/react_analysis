# 反応速度解析アプリ

反応速度定数・反応次数を実験データから推算する Streamlit アプリです。
単純反応・逐次反応・並列反応に対応し、複数温度データからアレニウスパラメータも算出できます。

---

## 機能

### 対応反応タイプ

| タイプ | 反応式 | 必要データ |
|--------|--------|-----------|
| 単純反応 | A → products | 濃度A |
| 逐次反応 | A → B → C | 濃度A + 濃度B（+ 濃度C） |
| 並列反応 | A → B + A → C | 濃度A + 濃度B + 濃度C |

### 解析手法

| 手法 | 対象 | 概要 |
|------|------|------|
| **RK4+最小二乗法** | 全反応タイプ | RK45 数値積分 + `scipy.optimize.least_squares` でパラメータ最適化。k, n, C0 を同時推定、マルチスタートで局所解を回避 |
| **アレニウス解析** | 複数温度データ | 温度別速度定数から Ea（活性化エネルギー）と頻度因子 A を線形回帰で算出 |
| **反応シミュレーション** | 全反応タイプ | 解析済みパラメータ（k, n）または任意の温度でのアレニウス外挿値を用いて RK45 で濃度プロファイルを順方向予測 |

---

## セットアップ

### ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

### Docker

```bash
docker build -t reaction-kinetics .
docker run -p 8080:8080 reaction-kinetics
```

### Google Cloud Run へのデプロイ

```bash
# Artifact Registry にプッシュ
docker build -t asia-northeast1-docker.pkg.dev/<PROJECT>/reaction-kinetics/app .
docker push asia-northeast1-docker.pkg.dev/<PROJECT>/reaction-kinetics/app

# Cloud Run にデプロイ
gcloud run deploy reaction-kinetics \
  --image asia-northeast1-docker.pkg.dev/<PROJECT>/reaction-kinetics/app \
  --region asia-northeast1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080
```

---

## 使い方

1. **テンプレートDL** — サイドバーから Excel テンプレートをダウンロード
2. **データ入力** — テンプレートに実験データを記入して保存
3. **アップロード** — サイドバーの「データアップロード」から xlsx / csv ファイルを選択
4. **解析設定** — 反応タイプを選択（自動判定あり）
5. **解析実行** — 「解析実行」ボタンを押す
6. **結果確認・出力** — グラフ確認後、Excel レポートをダウンロード
7. **シミュレーション**（任意） — 「🧪 シミュレーション」タブで温度・初期濃度・終了時刻を変えて濃度プロファイルを予測。複数条件を重ね描きして比較できます

---

## データ形式

### Excel フォーマット（テンプレート推奨）

**シート名: 実験データ**

| 列名 | 必須 | 説明 |
|------|------|------|
| 時間 (Time) | ✅ | 測定時刻（min） |
| 濃度_A (Concentration_A) | ✅ | 成分A の濃度（mol/L） |
| 濃度_B (Concentration_B) | 逐次・並列反応時 | 成分B の濃度（mol/L） |
| 濃度_C (Concentration_C) | 並列反応時 | 成分C の濃度（mol/L） |
| 温度 (Temperature) | アレニウス時 | 測定温度（°C）|
| 備考 (Notes) | — | 自由記述 |

**シート名: 実験条件**（任意）

実験名・反応物質・初期濃度・実験日・担当者・備考

### データ入力のルール

- **異なる時間点で成分ごとに測定した場合**: 他成分の欄を空白にしてください（NaN として認識）
- **複数温度データ**: 各行の Temperature 列に測定温度を記入し、すべて 1 枚のシートにまとめます
- **アレニウス解析の要件**: 各温度グループで 3 点以上の有効データ・R² ≥ 0.5 が必要です

---

## サンプルデータ

`sample_data/` に 6 種類のサンプルが含まれています。

| ファイル | 内容 | 真値 |
|----------|------|------|
| `sample1_simple_1st_order.xlsx` | 単純1次反応（ノイズ2%付き） | k = 0.0234 min⁻¹ |
| `sample2_sequential.xlsx` | 逐次反応 A→B→C | k1 = 0.05, k2 = 0.02 min⁻¹ |
| `sample3_parallel.xlsx` | 並列反応 A→B + A→C | k1 = 0.03, k2 = 0.01 min⁻¹ |
| `sample4_multi_temp_arrhenius.xlsx` | 単純反応 4温度（25/35/45/55°C） | Ea = 50 kJ/mol, A = 6×10⁶ |
| `sample5_different_timepoints.xlsx` | 逐次反応（A/B/C が異なる時間点） | k1 = 0.04, k2 = 0.015 min⁻¹ |
| `sample6_multi_temp_sequential.xlsx` | 逐次反応 3温度（25/40/55°C） | Ea(k1) = 50 kJ/mol, Ea(k2) = 30 kJ/mol |

サンプルデータを再生成する場合:

```bash
python generate_samples.py
```

---

## 出力

### アプリ内タブ

| タブ | 内容 |
|------|------|
| 📊 データ確認 | 生データ表・濃度プロファイル・質量バランスチェック |
| 🔬 解析結果 | RK4+最小二乗法の結果・フィットグラフ・残差プロット。多温度データ時は温度別に展開表示 |
| 🌡️ Arrheniusパラメータ | ln(k) vs 1/T プロット・Ea・A・R²。温度別速度定数テーブル |
| 📄 レポート出力 | 解析結果サマリー・Excel レポートダウンロード |
| 🧪 シミュレーション | 解析結果のパラメータを用いた濃度プロファイル予測。複数条件の重ね描き・CSV ダウンロード対応 |
| 📖 解析ロジック・結果の読み取り方 | 解析手法の詳細・各指標の意味・シミュレーションロジック・トラブルシューティング |

### Excel レポート構成

| シート | 内容 | 出力条件 |
|--------|------|----------|
| サマリー | 全手法の結果・アレニウスパラメータ・警告メッセージ | 常時 |
| 生データ | 元の実験データ | 常時 |
| 温度別反応次数 | 各温度の k・R²・R²加重平均 n | 多温度データ時 |

---

## ファイル構成

```
react_analysis/
├── app.py                        # Streamlit エントリーポイント
├── requirements.txt              # 依存パッケージ（固定バージョン）
├── Dockerfile                    # python:3.11-slim, port 8080
├── .dockerignore
├── create_template.py            # Excel テンプレート生成
├── generate_samples.py           # サンプルデータ生成
├── src/
│   ├── __init__.py
│   ├── analysis.py               # 解析パイプライン統括
│   ├── arrhenius.py              # アレニウス解析
│   ├── data_loader.py            # Excel/CSV 読み込み・バリデーション
│   ├── fitting.py                # RK4+最小二乗法フィッティング
│   ├── models.py                 # データクラス定義
│   ├── ode_systems.py            # ODE 系定義・ソルバー
│   ├── plotting.py               # Plotly グラフ生成
│   └── reporter.py               # Excel レポート出力
├── sample_data/                  # サンプルデータ 6種
└── template/
    └── experiment_template.xlsx  # ダウンロード用テンプレート
```

---

## 依存パッケージ

```
streamlit==1.32.0
pandas==2.2.1
numpy==1.26.4
scipy==1.12.0
plotly==5.20.0
openpyxl==3.1.2
xlsxwriter==3.1.9
```

---

## 注意事項

- **多温度データ解析**: 全データ結合後のフィット結果は参考値です。温度依存性の正確な評価には「Arrhenius パラメータ」タブの温度別結果を使用してください。
- **アレニウス解析の品質**: R² (アレニウス) が低い場合（< 0.95）、温度別の k 推定精度が不十分な可能性があります。データ点数を増やすか、温度範囲を見直してください。
- **RK4法の内部動作**: RK45 ODE ソルバー + `scipy.optimize.least_squares` のセットです。収束しない場合はデータ点数の増加・時間スケールの確認を推奨します。
