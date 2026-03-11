"""Excel/CSV data loading and validation for reaction kinetics analysis."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name detection helpers
# ---------------------------------------------------------------------------

_TIME_KEYWORDS     = ["time", "時間", "t(", "t ("]
_CONC_A_KEYWORDS   = ["concentration", "濃度", "conc_a", "conca", "[a]"]
_CONC_B_KEYWORDS   = ["concentration_b", "conc_b", "concb", "[b]", "濃度_b", "濃度b"]
_CONC_C_KEYWORDS   = ["concentration_c", "conc_c", "concc", "[c]", "濃度_c", "濃度c"]
_TEMP_KEYWORDS     = ["temperature", "温度", "temp"]
_NOTES_KEYWORDS    = ["notes", "備考", "note", "remarks"]


def _match_col(col_lower: str, keywords: list[str]) -> bool:
    return any(kw in col_lower for kw in keywords)


def _detect_columns(raw: pd.DataFrame) -> dict[str, str | None]:
    """
    Detect column roles by name matching with positional fallback.
    Returns mapping: role -> actual column name (or None).
    """
    cols_lower = {col: col.lower() for col in raw.columns}

    result: dict[str, str | None] = {
        "time": None,
        "concentration": None,
        "concentration_B": None,
        "concentration_C": None,
        "temperature": None,
        "notes": None,
    }

    for col, col_l in cols_lower.items():
        if result["time"] is None and _match_col(col_l, _TIME_KEYWORDS):
            result["time"] = col
        elif result["concentration_B"] is None and _match_col(col_l, _CONC_B_KEYWORDS):
            result["concentration_B"] = col
        elif result["concentration_C"] is None and _match_col(col_l, _CONC_C_KEYWORDS):
            result["concentration_C"] = col
        elif result["concentration"] is None and _match_col(col_l, _CONC_A_KEYWORDS):
            result["concentration"] = col
        elif result["temperature"] is None and _match_col(col_l, _TEMP_KEYWORDS):
            result["temperature"] = col
        elif result["notes"] is None and _match_col(col_l, _NOTES_KEYWORDS):
            result["notes"] = col

    # Positional fallback for mandatory columns
    n_cols = len(raw.columns)
    if result["time"] is None and n_cols >= 1:
        result["time"] = raw.columns[0]
    if result["concentration"] is None and n_cols >= 2:
        result["concentration"] = raw.columns[1]
    if result["temperature"] is None and n_cols >= 3:
        # Only apply positional fallback if the column wasn't already assigned to B/C
        candidates = [
            raw.columns[i] for i in range(2, n_cols)
            if raw.columns[i] not in (result["concentration_B"], result["concentration_C"])
        ]
        for cand in candidates:
            if result["temperature"] is None:
                result["temperature"] = cand
    if result["notes"] is None and n_cols >= 4:
        candidates = [
            raw.columns[i] for i in range(3, n_cols)
            if raw.columns[i] not in (
                result["concentration_B"], result["concentration_C"], result["temperature"]
            )
        ]
        for cand in candidates:
            if result["notes"] is None:
                result["notes"] = cand

    return result


# ---------------------------------------------------------------------------
# Reaction type auto-detection
# ---------------------------------------------------------------------------

def auto_detect_reaction_type(df: pd.DataFrame) -> tuple[str, str]:
    """
    Suggest reaction type from data patterns.

    Returns
    -------
    (suggested_type, reason_message)
    suggested_type: "simple" | "sequential" | "parallel"
    """
    has_B = "concentration_B" in df.columns and df["concentration_B"].notna().any()
    has_C = "concentration_C" in df.columns and df["concentration_C"].notna().any()

    if not has_B:
        return "simple", "濃度Bデータがないため単純反応 A→products として解析します。"

    B_valid = df["concentration_B"].dropna().values
    if len(B_valid) < 3:
        return "sequential", "濃度Bのデータ点数が少ないため逐次反応を仮定します。"

    # Check for peak in B (sequential indicator)
    peak_idx = int(np.argmax(B_valid))
    if 0 < peak_idx < len(B_valid) - 1:
        return "sequential", (
            f"濃度Bが時刻インデックス {peak_idx} でピークを持つため、"
            "逐次反応 A→B→C を推奨します。"
        )

    # B monotonically increases -> parallel
    if has_C:
        return "parallel", (
            "濃度Bが単調増加かつ濃度Cデータがあるため、"
            "並列反応 A→B + A→C を推奨します。"
        )

    return "parallel", (
        "濃度Bが単調増加するため並列反応 A→B + A→C を推奨します。"
        "（濃度CがなければAのみの解析も可能です）"
    )


# ---------------------------------------------------------------------------
# Internal: build and validate DataFrame from raw
# ---------------------------------------------------------------------------

def _build_dataframe(
    raw: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """Build, validate, and clean output DataFrame from raw data."""
    col_map = _detect_columns(raw)

    time_col = col_map["time"]
    conc_col = col_map["concentration"]
    if time_col is None or conc_col is None:
        raise ValueError("時間列または濃度A列が見つかりません。テンプレートの列名を確認してください。")

    # Build output dict
    out_cols: dict[str, Any] = {
        "time":          pd.to_numeric(raw[time_col], errors="coerce"),
        "concentration": pd.to_numeric(raw[conc_col], errors="coerce"),
    }

    if col_map["concentration_B"] is not None:
        out_cols["concentration_B"] = pd.to_numeric(raw[col_map["concentration_B"]], errors="coerce")
    if col_map["concentration_C"] is not None:
        out_cols["concentration_C"] = pd.to_numeric(raw[col_map["concentration_C"]], errors="coerce")

    temp_col  = col_map["temperature"]
    notes_col = col_map["notes"]
    out_cols["temperature"] = pd.to_numeric(raw[temp_col], errors="coerce") if temp_col else np.nan
    out_cols["notes"]       = raw[notes_col].astype(str) if notes_col else ""

    df = pd.DataFrame(out_cols)

    # ----------------------------------------------------------------
    # 1. timeがNaNの行のみ除外（concentrationがNaNの行は多成分データの
    #    可能性があるため保持）
    # ----------------------------------------------------------------
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    # concentrationAの有効点数チェック
    n_valid_A = df["concentration"].notna().sum()
    if n_valid_A < 3:
        raise ValueError(
            f"濃度A の有効データが {n_valid_A} 点です。解析には最低3点必要です。"
        )

    # ----------------------------------------------------------------
    # 2. B/C 列が全NaNなら除去（全NaN列は選択肢に出さない）
    # ----------------------------------------------------------------
    for sp_col in ("concentration_B", "concentration_C"):
        if sp_col in df.columns:
            if df[sp_col].isna().all():
                df = df.drop(columns=[sp_col])
                sp = sp_col[-1].upper()
                warnings.append(
                    f"濃度{sp} は全て空欄/欠損値のため無視します（単純反応として扱います）。"
                )

    # ----------------------------------------------------------------
    # 3. 時間列の検証・重複除去
    # ----------------------------------------------------------------
    if not df["time"].is_monotonic_increasing:
        warnings.append("時間列が昇順になっていません。自動でソートします。")
        df = df.sort_values("time").reset_index(drop=True)

    # 重複除去: 複数温度データでは (time, temperature) の組合せで判定
    temp_series_valid = df["temperature"].dropna()
    has_multi_temp = (temp_series_valid.nunique() > 1) if len(temp_series_valid) > 0 else False
    dup_cols = ["time", "temperature"] if has_multi_temp else ["time"]
    dup_mask = df.duplicated(subset=dup_cols, keep="first")
    if dup_mask.any():
        n_dup = dup_mask.sum()
        warnings.append(
            f"重複する測定値が {n_dup} 件あります（同一時刻・同一温度）。各グループの最初の行を使用します。"
        )
        df = df[~dup_mask].reset_index(drop=True)

    # ----------------------------------------------------------------
    # 4. 濃度Aの品質警告
    # ----------------------------------------------------------------
    if (df["concentration"] <= 0).any():
        n_neg = (df["concentration"] <= 0).sum()
        warnings.append(
            f"濃度Aが0以下のデータ点が {n_neg} 件あります。1次・2次積分法から除外します。"
        )

    if df["concentration"].notna().sum() < 8:
        warnings.append(
            f"濃度Aのデータ点数が {df['concentration'].notna().sum()} 点です。"
            "解析の精度が低下する場合があります（推奨: 8点以上）。"
        )

    # ----------------------------------------------------------------
    # 5. 多成分データの質量バランス検証
    # ----------------------------------------------------------------
    has_B = "concentration_B" in df.columns
    has_C = "concentration_C" in df.columns

    if has_B:
        warnings.append(
            "濃度B列を検出しました。逐次反応 (A→B→C) または並列反応 (A→B + A→C) 解析が利用可能です。"
        )
    if has_C:
        warnings.append("濃度C列を検出しました。")

    if has_B:
        mb_ok, mb_cv = check_mass_balance(df)
        if not mb_ok:
            warnings.append(
                f"質量バランス検証: A+B+C の変動係数 = {mb_cv:.3f} (>5%)。"
                "開放系反応または測定誤差が大きい可能性があります。"
            )

    # ----------------------------------------------------------------
    # 6. 成分ごとの時間点が異なる場合の通知
    # ----------------------------------------------------------------
    if has_B:
        n_only_A = (df["concentration"].notna() & df["concentration_B"].isna()).sum()
        n_only_B = (df["concentration"].isna() & df["concentration_B"].notna()).sum()
        if n_only_A > 0 or n_only_B > 0:
            warnings.append(
                f"成分ごとに測定時間点が異なります（A専用行: {n_only_A}件, B専用行: {n_only_B}件）。"
                "ODE解析では各成分の測定時刻のみで残差を計算します。"
            )

    return df


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv_data(
    file: io.BytesIO,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    Load experiment data from uploaded CSV file.

    Expected CSV format (header row required):
        time,concentration,concentration_B,concentration_C,temperature,notes

    Returns
    -------
    df       : validated DataFrame
    metadata : dict (empty)
    warnings : list of Japanese warning messages
    """
    warnings: list[str] = []
    try:
        raw = pd.read_csv(file, header=0)
    except Exception as exc:
        raise ValueError(f"CSVファイルの読み込みに失敗しました: {exc}") from exc

    df = _build_dataframe(raw, warnings)
    return df, {}, warnings


# ---------------------------------------------------------------------------
# Main loader (auto-detect CSV vs Excel)
# ---------------------------------------------------------------------------

def load_experiment_data(
    file: io.BytesIO,
    filename: str = "",
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """
    Load experiment data from uploaded Excel or CSV file.

    Auto-detects format from filename extension; defaults to Excel if unknown.

    Returns
    -------
    df : DataFrame with columns:
         必須: [time, concentration, temperature, notes]
         任意: [concentration_B, concentration_C]
         ※ B/Cが全NaNの場合は列ごと除去される
    metadata : dict (空)
    warnings : list of Japanese warning messages
    """
    if filename.lower().endswith(".csv"):
        return load_csv_data(file)

    # Default: Excel
    warnings: list[str] = []
    try:
        raw = pd.read_excel(file, sheet_name="実験データ", header=0)
    except Exception as exc:
        raise ValueError(f"Excelファイルの読み込みに失敗しました: {exc}") from exc

    df = _build_dataframe(raw, warnings)
    return df, {}, warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_mass_balance(
    df: pd.DataFrame,
    tol: float = 0.05,
) -> tuple[bool, float]:
    """
    Check if A + B + C ≈ constant (closed system).

    Returns
    -------
    (is_balanced, coefficient_of_variation)
    Only rows where ALL available species have valid data are used.
    """
    species_cols = ["concentration"]
    if "concentration_B" in df.columns:
        species_cols.append("concentration_B")
    if "concentration_C" in df.columns:
        species_cols.append("concentration_C")

    if len(species_cols) < 2:
        return True, 0.0

    valid_mask = df[species_cols].notna().all(axis=1)
    if valid_mask.sum() < 2:
        return True, 0.0

    total = df.loc[valid_mask, species_cols].sum(axis=1)
    mean_total = total.mean()
    if mean_total == 0:
        return True, 0.0

    cv = float(total.std() / mean_total)
    return cv <= tol, cv


def get_temperature_groups(
    df: pd.DataFrame,
    tol: float = 1.0,
) -> dict[float, pd.DataFrame]:
    """
    Group DataFrame rows by temperature (±tol°C = same group).

    Groups are determined by finding clusters among the unique temperatures.
    Each group is keyed by its representative temperature (rounded mean).

    Returns
    -------
    dict: {representative_T_celsius: sub-DataFrame}
    Empty dict if no temperature column or all-NaN.
    """
    if "temperature" not in df.columns:
        return {}

    temp_series = df["temperature"].dropna()
    if temp_series.empty:
        return {}

    unique_temps = sorted(temp_series.unique())

    # Cluster by ±tol
    clusters: list[list[float]] = []
    for t in unique_temps:
        placed = False
        for grp in clusters:
            grp_mean = float(np.mean(grp))
            if abs(t - grp_mean) <= tol:
                grp.append(t)
                placed = True
                break
        if not placed:
            clusters.append([t])

    result: dict[float, pd.DataFrame] = {}
    for grp in clusters:
        rep_temp = round(float(np.mean(grp)), 2)
        mask = df["temperature"].apply(
            lambda x: pd.notna(x) and any(abs(x - t) <= tol for t in grp)
        )
        result[rep_temp] = df[mask].reset_index(drop=True)

    return result
