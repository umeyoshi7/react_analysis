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
| **積分法** | 単純反応 | 0/1/2次の解析解に線形回帰。R²・AIC でベスト次数を自動選択 |
| **最小二乗法（解析解）** | 単純反応 | `dC/dt = -kCⁿ` の厳密解に非線形最小二乗フィット。次数 n を自由パラメータとして推定 |
| **RK4法（ODE）** | 全反応タイプ | RK45 数値積分 + 最小二乗法でパラメータを最適化。逐次・並列反応の k1/k2 を同時推定 |
| **温度別反応次数推算** | 複数温度・単純反応 | 各温度グループを個別解析して n と k を推定。R² 加重平均で最適反応次数を決定 |
| **アレニウス解析** | 複数温度データ | 温度別速度定数から Ea（活性化エネルギー）と頻度因子 A を算出 |

> **手法の使い分け**
> - 単純反応で次数が 0/1/2 に近い場合 → **積分法**（高速・解釈容易）
> - 単純反応で非整数次数を推定したい場合 → **最小二乗法（解析解）**
> - 逐次・並列反応 → **RK4法（ODE）** 一択
> - 複数温度データ（単純反応）→ **温度別反応次数推算** で各温度の n/k を個別算出 → **アレニウス解析** で Ea を算出

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
3. **アップロード** — サイドバーの「データアップロード」から xlsx ファイルを選択
4. **解析設定** — 反応タイプと解法を選択
5. **解析実行** — 「解析実行」ボタンを押す
6. **結果確認・出力** — グラフ確認後、Excel レポートまたは PNG をダウンロード

---

## データ形式

### Excel フォーマット（テンプレート推奨）

**シート1: 実験データ**

| 列名 | 必須 | 説明 |
|------|------|------|
| 時間 (Time) | ✅ | 測定時刻（min） |
| 濃度_A (Concentration_A) | ✅ | 成分A の濃度（mol/L） |
| 濃度_B (Concentration_B) | 逐次・並列反応時 | 成分B の濃度（mol/L） |
| 濃度_C (Concentration_C) | 並列反応時 | 成分C の濃度（mol/L） |
| 温度 (Temperature) | アレニウス時 | 測定温度（°C）|
| 備考 (Notes) | — | 自由記述 |

**シート2: 実験条件**（任意）

実験名・反応物質・初期濃度・実験日・担当者・備考

### データ入力のルール

- **異なる時間点で成分ごとに測定した場合**: 他成分の欄を空白にしてください（NaN として認識）
- **複数温度データ**: 各行の Temperature 列に測定温度を記入し、すべて 1 枚のシートにまとめます
- 濃度が 0 以下の値は自動的に除外されます

---

## サンプルデータ

`sample_data/` に 6 種類のサンプルが含まれています。

| ファイル | 内容 | 真値 |
|----------|------|------|
| `sample1_simple_1st_order.xlsx` | 単純1次反応（ノイズ2%付き） | k = 0.0234 min⁻¹ |
| `sample2_sequential.xlsx` | 逐次反応 A→B→C | k1 = 0.05, k2 = 0.02 min⁻¹ |
| `sample3_parallel.xlsx` | 並列反応 A→B + A→C | k1 = 0.03, k2 = 0.01 min⁻¹ |
| `sample4_multi_temp_arrhenius.xlsx` | 単純反応 4温度（25/35/45/55°C） | Ea = 60 kJ/mol, A = 5×10⁹ |
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
| データ確認 | 生データ表・濃度プロファイル・質量バランスチェック |
| 解析結果 | 積分法 / 最小二乗法（解析解）/ RK4法 の結果・グラフ・手法比較表。多温度データ時は最適反応次数 n を追加表示 |
| Arrhenius パラメータ | ln(k) vs 1/T プロット・Ea/A/95%CI。単純反応の多温度データ時は「温度別反応次数推算」セクション（テーブル・棒グラフ）を表示 |
| レポート出力 | Excel レポート・PNG ダウンロード |

### Excel レポート構成

| シート | 内容 | 出力条件 |
|--------|------|----------|
| サマリー | 全手法の結果・アレニウスパラメータ・警告メッセージ | 常時 |
| 積分法詳細 | 0/1/2次ごとの k, R², AIC | 常時 |
| 生データ | 元の実験データ | 常時 |
| 温度別反応次数 | 各温度の推算次数 n・速度定数 k・R²・R²加重平均 n | 多温度かつ単純反応時 |

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
│   ├── data_loader.py            # Excel 読み込み・バリデーション
│   ├── kinetics.py               # 解析コア（積分法・LSQ・RK4・Arrhenius）
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

- **多温度データの解析について**: 複数温度のデータを結合した状態での積分法・LSQ・RK4 の結果は参考値です。温度別の速度定数・反応次数は「Arrhenius パラメータ」タブの「温度別反応次数推算」セクションを参照してください。
- **温度別反応次数推算は単純反応のみ対応**です。逐次・並列反応では n=1.0 固定のため温度別次数推算はスキップされます。
- **最小二乗法（解析解）は単純反応のみ対応**です。逐次・並列反応は RK4法を使用してください。
- **RK4法（ODE）は内部で最小二乗最適化を含みます**（RK45 ODE ソルバー + `scipy.optimize.least_squares` のセット）。
