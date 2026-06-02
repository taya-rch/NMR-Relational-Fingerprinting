"""
1H NMR Relational Fingerprinting — Streamlit MVP

Goal
----
Build a Shazam-inspired fingerprinting system for 1H NMR peak lists.

Initial use case:
1. Build a reference database from 1H NMR peak lists downloaded from BMRB or prepared as CSV/TSV.
2. Convert each 1H NMR spectrum into relational hashes based on peak spacing, relative intensity, area, and width.
3. Search unknown spectra or mixtures against the reference database.

Expected input columns, if using CSV/TSV/TXT:
- compound_id or bmrb_id
- compound_name, optional
- peak_id, optional
- delta_H
- intensity, optional
- area, optional
- width, optional
- multiplicity, optional

Minimal required column:
- delta_H

Important note
--------------
This first MVP works with extracted peak tables, not raw FID or full spectral matrices.
A BMRB NMR-STAR parser can be added later in the function `parse_bmrb_nmrstar_text()`.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="1H NMR Relational Fingerprinting",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# Data model
# ============================================================

REQUIRED_COLUMNS = ["delta_H"]
OPTIONAL_COLUMNS = [
    "compound_id",
    "compound_name",
    "peak_id",
    "intensity",
    "area",
    "width",
    "multiplicity",
    "source_file",
    "SMILES",
    "INCHI_KEY",
    "GENERIC_NAME",
]


@dataclass
class ProtonHashParameters:
    h_bin_size: float = 0.01
    min_delta_h: float = 0.02
    max_delta_h: float = 5.0
    max_neighbors_per_peak: int = 30
    include_intensity_ratio: bool = True
    intensity_ratio_bin_size: float = 0.25
    include_area_ratio: bool = False
    area_ratio_bin_size: float = 0.25
    include_width_ratio: bool = False
    width_ratio_bin_size: float = 0.25
    include_multiplicity: bool = False
    use_absolute_pair_direction: bool = True
    min_intensity_quantile: float = 0.0
    match_shift_tolerance: float = 0.05
    min_relative_intensity: float = 0.02


# ============================================================
# Column standardization and import
# ============================================================

def normalize_column_name(col: str) -> str:
    """Normalize common peak-list column names into the internal schema."""
    c = str(col).strip().lower()
    c = c.replace(" ", "_").replace("-", "_").replace(".", "_")
    c = c.replace("1h", "h")

    aliases = {
        "id": "peak_id",
        "peak": "peak_id",
        "peak_number": "peak_id",
        "peak_id": "peak_id",
        "assignment": "peak_id",
        "atom_id": "peak_id",
        "h": "delta_H",
        "proton": "delta_H",
        "proton_shift": "delta_H",
        "shift": "delta_H",
        "chemical_shift": "delta_H",
        "chemical_shift_h": "delta_H",
        "delta": "delta_H",
        "delta_h": "delta_H",
        "ppm": "delta_H",
        "h_ppm": "delta_H",
        "ppm_h": "delta_H",
        "f2": "delta_H",
        "entry_id": "compound_id",
        "bmrb": "compound_id",
        "bmrb_id": "compound_id",
        "accession": "compound_id",
        "hmdb_id": "compound_id",
        "compound_id": "compound_id",
        "name": "compound_name",
        "compound": "compound_name",
        "compound_name": "compound_name",
        "metabolite": "compound_name",
        "generic_name": "GENERIC_NAME",
        "smiles": "SMILES",
        "inchi_key": "INCHI_KEY",
        "inchikey": "INCHI_KEY",
        "height": "intensity",
        "int": "intensity",
        "intensity": "intensity",
        "relative_intensity": "intensity",
        "amplitude": "intensity",
        "integral": "area",
        "integration": "area",
        "area": "area",
        "peak_area": "area",
        "line_width": "width",
        "linewidth": "width",
        "width": "width",
        "fwhm": "width",
        "mult": "multiplicity",
        "multiplet": "multiplicity",
        "multiplicity": "multiplicity",
        "position": "delta_H",
        "chemicalshift": "delta_H",
        "cs": "delta_H",
        "ppm_1h": "delta_H",
        "signal": "intensity",
        "height_abs": "intensity",
        "lw": "width",
        "position_ppm": "delta_H",
        "peak_position": "delta_H",
        "peak_position_ppm": "delta_H",
        "chemical_shift_ppm": "delta_H",
        "positionppm": "delta_H",
        "peakposition": "delta_H",
    }
    return aliases.get(c, col)


def standardize_proton_peak_table(
    df: pd.DataFrame,
    default_compound_id: Optional[str] = None,
    default_compound_name: Optional[str] = None,
    source_file: Optional[str] = None,
) -> pd.DataFrame:
    """Standardize a 1H NMR peak list into the internal format."""
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    if "compound_id" not in df.columns:
        df["compound_id"] = default_compound_id or "unknown_compound"

    if "compound_name" not in df.columns:
        df["compound_name"] = default_compound_name or df["compound_id"].astype(str)

    if "peak_id" not in df.columns:
        df["peak_id"] = np.arange(1, len(df) + 1)

    if "intensity" not in df.columns:
        df["intensity"] = 1.0

    if "area" not in df.columns:
        df["area"] = np.nan

    if "width" not in df.columns:
        df["width"] = np.nan

    if "multiplicity" not in df.columns:
        df["multiplicity"] = "unknown"

    if source_file is not None:
        df["source_file"] = source_file
    elif "source_file" not in df.columns:
        df["source_file"] = "unknown_source"

    # Optional structure metadata. These are preserved when present or merged later
    # from a separate HMDB mapping file.
    for optional_col in ["SMILES", "INCHI_KEY", "GENERIC_NAME"]:
        if optional_col not in df.columns:
            df[optional_col] = np.nan

    possible_shift_cols = [
        c for c in df.columns
        if any(
            token in c.lower()
            for token in [
                "ppm",
                "shift",
                "position",
                "delta",
                "f2",
            ]
        )
    ]

    if "delta_H" not in df.columns and len(possible_shift_cols) == 1:

        df = df.rename(columns={
            possible_shift_cols[0]: "delta_H"
        })

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if "delta_H" not in df.columns:

        st.error("Could not identify the chemical shift column.")

        st.write("Detected columns:")
        st.write(df.columns.tolist())

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["delta_H"] = pd.to_numeric(df["delta_H"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce").fillna(1.0)
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["width"] = pd.to_numeric(df["width"], errors="coerce")

    df = df.dropna(subset=["delta_H"])
    df = df[df["delta_H"].between(-1, 15)]

    # Avoid zeros or negative values in ratio calculations.
    df["intensity"] = df["intensity"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df.loc[df["intensity"] <= 0, "intensity"] = 1e-9

    columns = [
        "compound_id",
        "delta_H",
        "compound_name",
        "peak_id",
        "intensity",
        "area",
        "width",
        "multiplicity",
        "source_file",
        "SMILES",
        "INCHI_KEY",
        "GENERIC_NAME",
    ]

    columns = [c for c in columns if c in df.columns]
    df = df[columns].copy()

    df["compound_id"] = df["compound_id"].astype(str)
    df["compound_name"] = df["compound_name"].astype(str)
    df["peak_id"] = df["peak_id"].astype(str)
    df["multiplicity"] = df["multiplicity"].astype(str).str.lower().str.strip()
    for optional_col in ["SMILES", "INCHI_KEY", "GENERIC_NAME"]:
        if optional_col in df.columns:
            df[optional_col] = df[optional_col].astype("string")

    return df.sort_values("delta_H").reset_index(drop=True)


def read_table_from_upload(uploaded_file) -> pd.DataFrame:
    """Read CSV/TSV/TXT using automatic delimiter inference."""
    raw = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")


def standardize_structure_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize an HMDB structure mapping table.

    Expected columns can include HMDB_ID, SMILES, INCHI_KEY and GENERIC_NAME.
    The output uses compound_id so it can be merged directly with the peak database.
    """
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    st.write(df.columns.tolist())

    if "HMDB_ID" in df.columns and "compound_id" not in df.columns:
        df = df.rename(columns={"HMDB_ID": "compound_id"})

    # Some files may keep the original uppercase column after normalization if alias was not applied.
    rename_map = {}
    for col in df.columns:
        col_upper = str(col).upper()
        if col_upper == "HMDB_ID":
            rename_map[col] = "compound_id"
        elif col_upper == "SMILES":
            rename_map[col] = "SMILES"
        elif col_upper in {"INCHI_KEY", "INCHIKEY"}:
            rename_map[col] = "INCHI_KEY"
        elif col_upper == "GENERIC_NAME":
            rename_map[col] = "GENERIC_NAME"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["compound_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Structure mapping is missing required columns: {missing}")

    for col in ["SMILES", "INCHI_KEY", "GENERIC_NAME"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[["compound_id", "SMILES", "INCHI_KEY", "GENERIC_NAME"]].copy()
    df["compound_id"] = df["compound_id"].astype(str).str.strip()
    for col in ["SMILES", "INCHI_KEY", "GENERIC_NAME"]:
        df[col] = df[col].astype("string").str.strip()

    df = df.drop_duplicates(subset=["compound_id"], keep="first")
    return df


def merge_structure_mapping(peak_library: pd.DataFrame, structure_mapping: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge optional structure metadata into the peak library by compound_id/HMDB_ID."""
    if structure_mapping is None or structure_mapping.empty:
        return peak_library

    structure_mapping = standardize_structure_mapping(structure_mapping)

    # Remove existing structure columns before merging to avoid _x/_y suffixes.
    base = peak_library.drop(columns=[c for c in ["SMILES", "INCHI_KEY", "GENERIC_NAME"] if c in peak_library.columns])
    merged = base.merge(structure_mapping, on="compound_id", how="left")

    # If the peak file uses HMDB IDs as compound_name but not compound_id, try that too.
    if merged["SMILES"].isna().all() and "compound_name" in peak_library.columns:
        temp_mapping = structure_mapping.rename(columns={"compound_id": "compound_name"})
        base = peak_library.drop(columns=[c for c in ["SMILES", "INCHI_KEY", "GENERIC_NAME"] if c in peak_library.columns])
        merged = base.merge(temp_mapping, on="compound_name", how="left")

    return merged


def parse_bmrb_nmrstar_text(text: str, source_name: str) -> pd.DataFrame:
    """Placeholder parser for BMRB NMR-STAR files.

    BMRB files can store chemical shifts in loops containing atom identifiers,
    atom types, and chemical shift values. This function is intentionally minimal
    in the MVP and should be expanded once you decide which BMRB export format
    you will download.

    For now, use BMRB-exported CSV/TSV peak lists or manually prepared peak tables.
    """
    raise NotImplementedError(
        "NMR-STAR parsing is not implemented in this MVP. "
        "Please upload CSV/TSV peak lists with at least a delta_H column."
    )


def load_peak_tables_from_uploads(uploaded_files) -> pd.DataFrame:
    """Load multiple CSV/TSV/TXT/ZIP files and concatenate standardized peak tables."""
    all_tables = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        lower_name = file_name.lower()

        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file) as zf:
                for member in zf.namelist():
                    lower_member = member.lower()
                    if lower_member.endswith((".csv", ".tsv", ".txt")):
                        with zf.open(member) as fh:
                            df = pd.read_csv(fh, sep=None, engine="python")
                        compound_id = Path(member).stem
                        table = standardize_proton_peak_table(
                            df,
                            default_compound_id=compound_id,
                            default_compound_name=compound_id,
                            source_file=member,
                        )
                        all_tables.append(table)
                    elif lower_member.endswith((".str", ".star")):
                        with zf.open(member) as fh:
                            text = fh.read().decode("utf-8", errors="ignore")
                        table = parse_bmrb_nmrstar_text(text, member)
                        all_tables.append(table)
        elif lower_name.endswith((".str", ".star")):
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            table = parse_bmrb_nmrstar_text(text, file_name)
            all_tables.append(table)
        else:
            df = read_table_from_upload(uploaded_file)
            compound_id = Path(file_name).stem
            table = standardize_proton_peak_table(
                df,
                default_compound_id=compound_id,
                default_compound_name=compound_id,
                source_file=file_name,
            )
            all_tables.append(table)

    if not all_tables:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    return pd.concat(all_tables, ignore_index=True)


# ============================================================
# Hashing functions
# ============================================================

def quantize(value: float, bin_size: float) -> int:
    """Quantize a continuous value into an integer bin."""
    return int(np.round(value / bin_size))


def stable_hash(tokens: Iterable[object], digest_size: int = 12) -> str:
    """Create a stable compact hash from a list of tokens."""
    text = "|".join(str(t) for t in tokens)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size).hexdigest()


def safe_log_ratio(value_b: float, value_a: float) -> float:
    """Return log10(value_b / value_a) safely."""
    a = float(value_a) if pd.notna(value_a) and value_a > 0 else 1e-9
    b = float(value_b) if pd.notna(value_b) and value_b > 0 else 1e-9
    return float(np.log10(b / a))


def filter_peaks_by_intensity_quantile(df: pd.DataFrame, min_quantile: float) -> pd.DataFrame:
    """Optionally remove very weak peaks within each compound."""
    if min_quantile <= 0 or "intensity" not in df.columns:
        return df.copy()

    filtered = []
    for compound_id, group in df.groupby("compound_id", sort=False):
        cutoff = group["intensity"].quantile(min_quantile)
        filtered.append(group[group["intensity"] >= cutoff])

    if not filtered:
        return df.iloc[0:0].copy()

    return pd.concat(filtered, ignore_index=True)


def generate_hashes_for_proton_spectrum(
    peak_table: pd.DataFrame,
    params: ProtonHashParameters,
) -> pd.DataFrame:
    """Generate relational pair hashes for one 1H NMR spectrum.

    Each hash describes a relationship between two peaks:
    - chemical shift spacing, ΔδH
    - optional relative intensity
    - optional relative area
    - optional relative width
    - optional multiplicity pair
    """
    peaks = peak_table.copy().sort_values("delta_H").reset_index(drop=True)

    # 1) Remove peaks below a relative intensity threshold.
    # Example: min_relative_intensity = 0.02 keeps only peaks >= 2% of the maximum intensity.
    if (
        hasattr(params, "min_relative_intensity")
        and params.min_relative_intensity > 0
        and "intensity" in peaks.columns
        and not peaks.empty
    ):
        max_intensity = peaks["intensity"].max()

        if pd.notna(max_intensity) and max_intensity > 0:
            intensity_cutoff = max_intensity * params.min_relative_intensity
            peaks = peaks[peaks["intensity"] >= intensity_cutoff].copy()
            peaks = peaks.sort_values("delta_H").reset_index(drop=True)

    # 2) Then remove the weakest peaks by quantile.
    if params.min_intensity_quantile > 0:
        peaks = filter_peaks_by_intensity_quantile(peaks, params.min_intensity_quantile)
        peaks = peaks.sort_values("delta_H").reset_index(drop=True)

    if len(peaks) < 2:
        return pd.DataFrame()

    compound_id = str(peaks.loc[0, "compound_id"])
    compound_name = str(peaks.loc[0, "compound_name"])

    records = []
    coords = peaks["delta_H"].to_numpy(float)

    for i in range(len(peaks)):
        diffs = coords - coords[i]
        distances = np.abs(diffs)
        candidate_indices = np.argsort(distances)
        candidate_indices = [j for j in candidate_indices if j != i]
        candidate_indices = candidate_indices[: params.max_neighbors_per_peak]

        for j in candidate_indices:
            dH = float(peaks.loc[j, "delta_H"] - peaks.loc[i, "delta_H"])
            min_delta_h = params.min_delta_h

            if abs(dH) < min_delta_h:
                continue

            if abs(dH) > params.max_delta_h:
                continue

            if params.use_absolute_pair_direction:
                if dH < 0:
                    dH = -dH
                    anchor_idx = j
                    target_idx = i
                else:
                    anchor_idx = i
                    target_idx = j
            else:
                anchor_idx = i
                target_idx = j

            h_bin = quantize(dH, params.h_bin_size)
            tokens = ["PROTON_PAIR", h_bin]

            intensity_ratio_bin = None
            area_ratio_bin = None
            width_ratio_bin = None
            multiplicity_pair = None

            if params.include_intensity_ratio:
                ratio = safe_log_ratio(
                    peaks.loc[target_idx, "intensity"],
                    peaks.loc[anchor_idx, "intensity"],
                )
                intensity_ratio_bin = quantize(ratio, params.intensity_ratio_bin_size)
                tokens.append(f"I{intensity_ratio_bin}")

            if params.include_area_ratio and "area" in peaks.columns:
                ratio = safe_log_ratio(
                    peaks.loc[target_idx, "area"],
                    peaks.loc[anchor_idx, "area"],
                )
                area_ratio_bin = quantize(ratio, params.area_ratio_bin_size)
                tokens.append(f"A{area_ratio_bin}")

            if params.include_width_ratio and "width" in peaks.columns:
                ratio = safe_log_ratio(
                    peaks.loc[target_idx, "width"],
                    peaks.loc[anchor_idx, "width"],
                )
                width_ratio_bin = quantize(ratio, params.width_ratio_bin_size)
                tokens.append(f"W{width_ratio_bin}")

            if params.include_multiplicity and "multiplicity" in peaks.columns:
                m1 = str(peaks.loc[anchor_idx, "multiplicity"])
                m2 = str(peaks.loc[target_idx, "multiplicity"])
                multiplicity_pair = f"{m1}>{m2}"
                tokens.append(f"M{multiplicity_pair}")

            hash_value = stable_hash(tokens)

            records.append(
                {
                    "compound_id": compound_id,
                    "compound_name": compound_name,
                    "SMILES": str(peaks.loc[0, "SMILES"]) if "SMILES" in peaks.columns and pd.notna(peaks.loc[0, "SMILES"]) else None,
                    "INCHI_KEY": str(peaks.loc[0, "INCHI_KEY"]) if "INCHI_KEY" in peaks.columns and pd.notna(peaks.loc[0, "INCHI_KEY"]) else None,
                    "GENERIC_NAME": str(peaks.loc[0, "GENERIC_NAME"]) if "GENERIC_NAME" in peaks.columns and pd.notna(peaks.loc[0, "GENERIC_NAME"]) else None,
                    "hash_value": hash_value,
                    "h_bin": h_bin,
                    "intensity_ratio_bin": intensity_ratio_bin,
                    "area_ratio_bin": area_ratio_bin,
                    "width_ratio_bin": width_ratio_bin,
                    "multiplicity_pair": multiplicity_pair,
                    "anchor_peak_id": str(peaks.loc[anchor_idx, "peak_id"]),
                    "target_peak_id": str(peaks.loc[target_idx, "peak_id"]),
                    "anchor_delta_H": float(peaks.loc[anchor_idx, "delta_H"]),
                    "target_delta_H": float(peaks.loc[target_idx, "delta_H"]),
                    "delta_H_spacing": dH,
                    "h_bin_size": params.h_bin_size,
                }
            )

    hash_df = pd.DataFrame(records)
    if hash_df.empty:
        return hash_df

    hash_df = hash_df.drop_duplicates(
        subset=["compound_id", "hash_value", "anchor_peak_id", "target_peak_id"]
    ).reset_index(drop=True)

    return hash_df


def generate_proton_hash_library(peak_library: pd.DataFrame, params: ProtonHashParameters) -> pd.DataFrame:
    """Generate relational hash library for all reference compounds."""
    all_hashes = []

    for compound_id, group in peak_library.groupby("compound_id", sort=False):
        hashes = generate_hashes_for_proton_spectrum(group, params)
        if not hashes.empty:
            all_hashes.append(hashes)

    if not all_hashes:
        return pd.DataFrame()

    return pd.concat(all_hashes, ignore_index=True)


def compute_hash_rarity_weights(hash_library: pd.DataFrame) -> pd.DataFrame:
    """Compute rarity weights for hashes.

    A hash found in many compounds is less informative than a rare hash.
    The weight is a simple inverse document frequency-like score.
    """
    if hash_library.empty:
        return pd.DataFrame(columns=["hash_value", "compound_frequency", "rarity_weight"])

    total_compounds = hash_library["compound_id"].nunique()
    freq = (
        hash_library[["hash_value", "compound_id"]]
        .drop_duplicates()
        .groupby("hash_value")
        .agg(compound_frequency=("compound_id", "nunique"))
        .reset_index()
    )
    freq["rarity_weight"] = np.log((1 + total_compounds) / (1 + freq["compound_frequency"])) + 1
    return freq

def get_matched_hash_records(
    unknown_peaks: pd.DataFrame,
    library_hashes: pd.DataFrame,
    params: ProtonHashParameters,
) -> pd.DataFrame:

    unknown_hashes = generate_hashes_for_proton_spectrum(unknown_peaks, params)

    if unknown_hashes.empty or library_hashes.empty:
        return pd.DataFrame()

    unknown_hashes = unknown_hashes.drop_duplicates(
        subset=[
            "hash_value",
            "anchor_delta_H",
            "target_delta_H",
        ]
    )

    library_hashes = library_hashes.drop_duplicates(
        subset=[
            "compound_id",
            "hash_value",
            "anchor_delta_H",
            "target_delta_H",
        ]
    )

    matched = unknown_hashes.merge(
        library_hashes,
        on="hash_value",
        how="inner",
        suffixes=("_unknown", "_reference"),
    )

    return matched

def score_unknown_against_library(
    unknown_peaks: pd.DataFrame,
    library_hashes: pd.DataFrame,
    reference_hash_counts: pd.DataFrame,
    rarity_weights: pd.DataFrame,
    params: ProtonHashParameters,
) -> pd.DataFrame:
    """Score unknown using hash match + absolute δH agreement."""
    unknown_hashes = generate_hashes_for_proton_spectrum(unknown_peaks, params)

    if unknown_hashes.empty or library_hashes.empty:
        return pd.DataFrame()

    hits = unknown_hashes.merge(
        library_hashes,
        on="hash_value",
        how="inner",
        suffixes=("_unknown", "_reference"),
    )

    if hits.empty:
        return pd.DataFrame()

    tol = float(params.match_shift_tolerance)

    direct_match = (
        (hits["anchor_delta_H_unknown"] - hits["anchor_delta_H_reference"]).abs() <= tol
    ) & (
        (hits["target_delta_H_unknown"] - hits["target_delta_H_reference"]).abs() <= tol
    )

    swapped_match = (
        (hits["anchor_delta_H_unknown"] - hits["target_delta_H_reference"]).abs() <= tol
    ) & (
        (hits["target_delta_H_unknown"] - hits["anchor_delta_H_reference"]).abs() <= tol
    )

    hits = hits[direct_match | swapped_match].copy()

    if hits.empty:
        return pd.DataFrame()

    hits = hits.drop_duplicates(
        subset=[
            "compound_id_reference",
            "hash_value",
            "anchor_delta_H_reference",
            "target_delta_H_reference",
        ]
    )

    hits = hits.merge(rarity_weights, on="hash_value", how="left")
    hits["rarity_weight"] = hits["rarity_weight"].fillna(1.0)

    scores = (
        hits.groupby(
            [
                "compound_id_reference",
                "compound_name_reference",
                "SMILES_reference",
                "INCHI_KEY_reference",
                "GENERIC_NAME_reference",
            ],
            dropna=False,
        )
        .agg(
            matched_hashes=("hash_value", "nunique"),
            weighted_matched_score=("rarity_weight", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "compound_id_reference": "compound_id",
                "compound_name_reference": "compound_name",
                "SMILES_reference": "SMILES",
                "INCHI_KEY_reference": "INCHI_KEY",
                "GENERIC_NAME_reference": "GENERIC_NAME",
            }
        )
    )

    scores = scores.merge(reference_hash_counts, on=["compound_id", "compound_name"], how="left")
    scores["coverage_score"] = scores["matched_hashes"] / scores["reference_hashes"]
    scores["weighted_coverage_score"] = scores["weighted_matched_score"] / scores["reference_weighted_hashes"]

    scores = scores.sort_values(
        ["weighted_coverage_score", "coverage_score", "matched_hashes"],
        ascending=False,
    ).reset_index(drop=True)

    return scores

# ============================================================
# Visualization
# ============================================================

def plot_proton_peak_map(df: pd.DataFrame, title: str = "1H NMR peak list") -> go.Figure:
    """Plot 1H NMR peaks as lollipop markers."""
    plot_df = df.copy().sort_values("delta_H")
    y_col = "intensity" if "intensity" in plot_df.columns else None
    if y_col is None:
        plot_df["intensity"] = 1.0
        y_col = "intensity"

    fig = go.Figure()

    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["delta_H"], row["delta_H"]],
                y=[0, row[y_col]],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["delta_H"],
            y=plot_df[y_col],
            mode="markers",
            marker=dict(size=8),
            text=plot_df["peak_id"],
            customdata=np.stack(
                [
                    plot_df["compound_id"].astype(str),
                    plot_df["compound_name"].astype(str),
                    plot_df["multiplicity"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "δH: %{x:.4f} ppm<br>"
                "Intensity: %{y:.3g}<br>"
                "Peak: %{text}<br>"
                "Compound ID: %{customdata[0]}<br>"
                "Compound: %{customdata[1]}<br>"
                "Multiplicity: %{customdata[2]}<extra></extra>"
            ),
            name="Peaks",
        )
    )

    fig.update_xaxes(autorange="reversed", title="¹H chemical shift / ppm")
    fig.update_yaxes(title="Relative intensity")
    fig.update_layout(title=title, height=500)
    return fig


def plot_reference_library_overview(df: pd.DataFrame) -> go.Figure:
    """Plot all peaks from a reference library."""
    fig = px.scatter(
        df,
        x="delta_H",
        y="intensity",
        color="compound_id" if df["compound_id"].nunique() <= 20 else None,
        hover_data=["compound_id", "compound_name", "peak_id", "multiplicity"],
        title="Reference 1H NMR peak library",
    )
    fig.update_xaxes(autorange="reversed", title="¹H chemical shift / ppm")
    fig.update_yaxes(title="Relative intensity")
    fig.update_layout(height=600)
    return fig


def plot_spacing_distribution(hash_df: pd.DataFrame) -> go.Figure:
    """Plot distribution of pairwise 1H spacings used in hashes."""
    fig = px.histogram(
        hash_df,
        x="delta_H_spacing",
        nbins=100,
        title="Distribution of pairwise ¹H peak spacings used for hashes",
    )
    fig.update_xaxes(title="ΔδH / ppm")
    fig.update_yaxes(title="Number of pair records")
    fig.update_layout(height=450)
    return fig

def plot_matched_unknown_vs_reference(
    unknown_peaks: pd.DataFrame,
    reference_peaks: pd.DataFrame,
    matched_selected: pd.DataFrame,
    selected_hit: str,
) -> go.Figure:

    u = unknown_peaks.copy()
    r = reference_peaks.copy()

    unknown_matched_ppm = pd.unique(
        matched_selected[
            ["anchor_delta_H_unknown", "target_delta_H_unknown"]
        ].values.ravel()
    )

    reference_matched_ppm = pd.unique(
        matched_selected[
            ["anchor_delta_H_reference", "target_delta_H_reference"]
        ].values.ravel()
    )

    unknown_matched_ppm = set(np.round(unknown_matched_ppm.astype(float), 4))
    reference_matched_ppm = set(np.round(reference_matched_ppm.astype(float), 4))

    u["matched"] = np.round(u["delta_H"].astype(float), 4).isin(unknown_matched_ppm)
    r["matched"] = np.round(r["delta_H"].astype(float), 4).isin(reference_matched_ppm)

    fig = go.Figure()

    # Unknown spectrum, top
    for _, row in u.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["delta_H"], row["delta_H"]],
                y=[1.0, 1.0 + row["intensity"]],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=u["delta_H"],
            y=1.0 + u["intensity"],
            mode="markers",
            marker=dict(
                size=np.where(u["matched"], 12, 5),
                symbol=np.where(u["matched"], "circle", "circle-open"),
            ),
            text=u["peak_id"],
            customdata=np.stack(
                [
                    u["delta_H"],
                    u["intensity"],
                    u["matched"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Unknown</b><br>"
                "δH: %{customdata[0]:.4f} ppm<br>"
                "Intensity: %{customdata[1]:.3g}<br>"
                "Peak ID: %{text}<br>"
                "Matched: %{customdata[2]}<extra></extra>"
            ),
            name="Unknown peaks",
        )
    )

    # Reference spectrum, bottom, mirrored
    for _, row in r.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["delta_H"], row["delta_H"]],
                y=[-1.0, -1.0 - row["intensity"]],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=r["delta_H"],
            y=-1.0 - r["intensity"],
            mode="markers",
            marker=dict(
                size=np.where(r["matched"], 12, 5),
                symbol=np.where(r["matched"], "diamond", "diamond-open"),
            ),
            text=r["peak_id"],
            customdata=np.stack(
                [
                    r["delta_H"],
                    r["intensity"],
                    r["matched"],
                    r["compound_name"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Reference</b><br>"
                "Compound: %{customdata[3]}<br>"
                "δH: %{customdata[0]:.4f} ppm<br>"
                "Intensity: %{customdata[1]:.3g}<br>"
                "Peak ID: %{text}<br>"
                "Matched: %{customdata[2]}<extra></extra>"
            ),
            name="Reference peaks",
        )
    )

    # Lines connecting matched relationships
    for _, row in matched_selected.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[
                    row["anchor_delta_H_unknown"],
                    row["target_delta_H_unknown"],
                ],
                y=[1.0, 1.0],
                mode="lines+markers",
                line=dict(width=1, dash="dot"),
                marker=dict(size=6),
                showlegend=False,
                hovertemplate=(
                    "Matched unknown relationship<br>"
                    "Hash: " + str(row["hash_value"]) + "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    row["anchor_delta_H_reference"],
                    row["target_delta_H_reference"],
                ],
                y=[-1.0, -1.0],
                mode="lines+markers",
                line=dict(width=1, dash="dot"),
                marker=dict(size=6),
                showlegend=False,
                hovertemplate=(
                    "Matched reference relationship<br>"
                    "Hash: " + str(row["hash_value"]) + "<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=1.0, line_width=1)
    fig.add_hline(y=-1.0, line_width=1)

    fig.update_xaxes(
        autorange="reversed",
        title="¹H chemical shift / ppm",
    )

    fig.update_yaxes(
        title="Unknown spectrum / Reference spectrum",
        showticklabels=False,
    )

    fig.update_layout(
        title=f"Matched ¹H NMR peaks: unknown vs reference — {selected_hit}",
        height=650,
        hovermode="closest",
    )

    return fig

def plot_constellation_overlay_on_unknown(
    unknown_peaks: pd.DataFrame,
    matched_selected: pd.DataFrame,
    selected_hit: str,
) -> go.Figure:

    u = unknown_peaks.copy()

    max_int = u["intensity"].max()
    if max_int <= 0:
        max_int = 1.0

    u["intensity_norm"] = u["intensity"] / max_int

    matched_ppm = pd.unique(
        matched_selected[
            ["anchor_delta_H_unknown", "target_delta_H_unknown"]
        ].values.ravel()
    )

    matched_ppm_set = set(np.round(matched_ppm.astype(float), 4))

    u["matched"] = np.round(u["delta_H"].astype(float), 4).isin(matched_ppm_set)

    fig = go.Figure()

    # Experimental peak sticks
    for _, row in u.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["delta_H"], row["delta_H"]],
                y=[0, row["intensity_norm"]],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # All experimental peaks
    fig.add_trace(
        go.Scatter(
            x=u["delta_H"],
            y=u["intensity_norm"],
            mode="markers",
            marker=dict(
                size=np.where(u["matched"], 11, 5),
                symbol=np.where(u["matched"], "circle", "circle-open"),
            ),
            text=u["peak_id"],
            customdata=np.stack(
                [
                    u["delta_H"],
                    u["intensity"],
                    u["matched"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Experimental peak</b><br>"
                "δH: %{customdata[0]:.4f} ppm<br>"
                "Intensity: %{customdata[1]:.3g}<br>"
                "Peak ID: %{text}<br>"
                "Matched: %{customdata[2]}<extra></extra>"
            ),
            name="Experimental peaks",
        )
    )

    # Constellation hash relationships
    for _, row in matched_selected.iterrows():
        x1 = row["anchor_delta_H_unknown"]
        x2 = row["target_delta_H_unknown"]

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[1.10, 1.10],
                mode="lines+markers",
                line=dict(width=1, dash="dot"),
                marker=dict(size=6),
                customdata=[
                    [
                        row["hash_value"],
                        row["anchor_delta_H_reference"],
                        row["target_delta_H_reference"],
                        row["delta_H_spacing_unknown"],
                        row["delta_H_spacing_reference"],
                    ],
                    [
                        row["hash_value"],
                        row["anchor_delta_H_reference"],
                        row["target_delta_H_reference"],
                        row["delta_H_spacing_unknown"],
                        row["delta_H_spacing_reference"],
                    ],
                ],
                hovertemplate=(
                    "<b>Matched constellation edge</b><br>"
                    "Hash: %{customdata[0]}<br>"
                    "Unknown edge: %{x:.4f} ppm<br>"
                    "Reference anchor: %{customdata[1]:.4f} ppm<br>"
                    "Reference target: %{customdata[2]:.4f} ppm<br>"
                    "ΔδH unknown: %{customdata[3]:.4f}<br>"
                    "ΔδH reference: %{customdata[4]:.4f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_xaxes(
        autorange="reversed",
        title="¹H chemical shift / ppm",
    )

    fig.update_yaxes(
        title="Normalized experimental intensity + constellation layer",
        range=[0, 1.25],
    )

    fig.update_layout(
        title=f"Experimental ¹H NMR spectrum with matched constellation overlay — {selected_hit}",
        height=650,
        hovermode="closest",
    )

    return fig

def get_matched_hash_records_for_compound(
    unknown_peaks: pd.DataFrame,
    library_hashes: pd.DataFrame,
    selected_compound_id: str,
    params: ProtonHashParameters,
    max_records: int = 5000,
) -> pd.DataFrame:

    unknown_hashes = generate_hashes_for_proton_spectrum(unknown_peaks, params)

    if unknown_hashes.empty or library_hashes.empty:
        return pd.DataFrame()

    ref_hashes = library_hashes[
        library_hashes["compound_id"].astype(str) == str(selected_compound_id)
    ].copy()

    if ref_hashes.empty:
        return pd.DataFrame()

    matched = unknown_hashes.merge(
        ref_hashes,
        on="hash_value",
        how="inner",
        suffixes=("_unknown", "_reference"),
    )

    tol = float(params.match_shift_tolerance)

    direct_match = (
        (matched["anchor_delta_H_unknown"] - matched["anchor_delta_H_reference"]).abs() <= tol
    ) & (
        (matched["target_delta_H_unknown"] - matched["target_delta_H_reference"]).abs() <= tol
    )

    swapped_match = (
        (matched["anchor_delta_H_unknown"] - matched["target_delta_H_reference"]).abs() <= tol
    ) & (
        (matched["target_delta_H_unknown"] - matched["anchor_delta_H_reference"]).abs() <= tol
    )

    matched = matched[direct_match | swapped_match].copy()

    matched = matched.drop_duplicates(
        subset=[
            "hash_value",
            "anchor_delta_H_unknown",
            "target_delta_H_unknown",
            "anchor_delta_H_reference",
            "target_delta_H_reference",
        ]
    )

    return matched.head(max_records)

def plot_constellation_network(
    matched_selected: pd.DataFrame,
    selected_hit: str,
) -> go.Figure:

    fig = go.Figure()

    unique_ppm = sorted(
        set(
            matched_selected["anchor_delta_H_unknown"].tolist()
            + matched_selected["target_delta_H_unknown"].tolist()
        )
    )

    ppm_to_y = {
        ppm: i
        for i, ppm in enumerate(unique_ppm)
    }

    # Nodes
    fig.add_trace(
        go.Scatter(
            x=unique_ppm,
            y=[ppm_to_y[p] for p in unique_ppm],
            mode="markers+text",
            text=[f"{p:.3f}" for p in unique_ppm],
            textposition="top center",
            marker=dict(size=12),
            name="Experimental peaks",
            hovertemplate="δH %{x:.4f} ppm<extra></extra>",
        )
    )

    # Edges
    for _, row in matched_selected.iterrows():

        x1 = row["anchor_delta_H_unknown"]
        x2 = row["target_delta_H_unknown"]

        y1 = ppm_to_y[x1]
        y2 = ppm_to_y[x2]

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(width=2),
                showlegend=False,
                customdata=[
                    [
                        row["hash_value"],
                        row["anchor_delta_H_reference"],
                        row["target_delta_H_reference"],
                    ],
                    [
                        row["hash_value"],
                        row["anchor_delta_H_reference"],
                        row["target_delta_H_reference"],
                    ],
                ],
                hovertemplate=(
                    "Hash: %{customdata[0]}<br>"
                    "Reference: %{customdata[1]:.4f} → %{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(
        autorange="reversed",
        title="¹H chemical shift / ppm",
    )

    fig.update_yaxes(
        visible=False,
    )

    fig.update_layout(
        title=f"Matched constellation network — {selected_hit}",
        height=700,
        hovermode="closest",
    )

    return fig

def plot_experimental_spectrum_with_reference_peaks(
    unknown_peaks: pd.DataFrame,
    matched_selected: pd.DataFrame,
    selected_hit: str,
) -> go.Figure:

    u = unknown_peaks.copy()

    max_int = u["intensity"].max()
    if max_int <= 0:
        max_int = 1.0

    u["intensity_norm"] = u["intensity"] / max_int

    matched_ppm = pd.unique(
        matched_selected[
            ["anchor_delta_H_unknown", "target_delta_H_unknown"]
        ].values.ravel()
    )

    matched_ppm_set = set(np.round(matched_ppm.astype(float), 4))

    u["belongs_to_reference"] = (
        np.round(u["delta_H"].astype(float), 4).isin(matched_ppm_set)
    )

    fig = go.Figure()

    # All experimental peaks
    for _, row in u.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["delta_H"], row["delta_H"]],
                y=[0, row["intensity_norm"]],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Background experimental peaks
    bg = u[~u["belongs_to_reference"]]
    fig.add_trace(
        go.Scatter(
            x=bg["delta_H"],
            y=bg["intensity_norm"],
            mode="markers",
            marker=dict(size=1, symbol="circle-open"),
            name="Mixture peaks",
            hovertemplate=(
                "Mixture peak<br>"
                "δH: %{x:.4f} ppm<br>"
                "Normalized intensity: %{y:.3f}<extra></extra>"
            ),
        )
    )

    # Peaks assigned to selected reference
    ref = u[u["belongs_to_reference"]]
    fig.add_trace(
        go.Scatter(
            x=ref["delta_H"],
            y=ref["intensity_norm"],
            mode="markers",
            marker=dict(size=13, symbol="circle"),
            name=f"Matched peaks for {selected_hit}",
            hovertemplate=(
                "Matched reference peak in mixture<br>"
                "δH: %{x:.4f} ppm<br>"
                "Normalized intensity: %{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(
        autorange="reversed",
        title="¹H chemical shift / ppm",
    )

    fig.update_yaxes(
        title="Normalized experimental intensity",
    )

    fig.update_layout(
        title=f"Experimental mixture spectrum with peaks matched to {selected_hit}",
        height=550,
        hovermode="closest",
    )

    return fig

# ============================================================
# Streamlit UI
# ============================================================

st.title("1H NMR Relational Fingerprinting")
st.caption("A Shazam-inspired fingerprinting prototype for ¹H NMR peak lists and mixtures")

# LOGOS ======================================================
from PIL import Image
STATIC_DIR = Path(__file__).parent / "static"
LOGO_PATH = STATIC_DIR / "LAABio.png"
try:
    logo = Image.open(LOGO_PATH)  # raises if missing
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/LAABio.png")

st.markdown("by Ricardo M Borges (IPPN-UFRJ)")
 

 
with st.sidebar:
    st.divider()

    st.link_button(
        "📖 Documentation / Tutorial",
        "https://github.com/RicardoMBorges/NMR-Relational-Fingerprinting/blob/main/README.md",
        use_container_width=True,
    )
    
    st.divider()
    st.header("Fingerprint parameters")

    # ============================================================
    # Fingerprint mode
    # ============================================================

    st.subheader("Fingerprint mode")

    fingerprint_mode = st.selectbox(
        "Fingerprint mode",
        [
            "Geometry only",
            "Geometry + intensity",
            "Geometry + area",
            "Full relational",
            "Multiplet aware",
        ],
        index=0,
        help="""
Select which information will be used to build the relational fingerprints.
Geometry-only mode is recommended for complex mixtures and experimental peak lists.
"""
    )

    # ============================================================
    # Fingerprint mode presets
    # ============================================================

    if fingerprint_mode == "Geometry only":

        include_intensity_ratio = False
        include_area_ratio = False
        include_width_ratio = False
        include_multiplicity = False

    elif fingerprint_mode == "Geometry + intensity":

        include_intensity_ratio = True
        include_area_ratio = False
        include_width_ratio = False
        include_multiplicity = False

    elif fingerprint_mode == "Geometry + area":

        include_intensity_ratio = False
        include_area_ratio = True
        include_width_ratio = False
        include_multiplicity = False

    elif fingerprint_mode == "Full relational":

        include_intensity_ratio = True
        include_area_ratio = True
        include_width_ratio = True
        include_multiplicity = False

    elif fingerprint_mode == "Multiplet aware":

        include_intensity_ratio = True
        include_area_ratio = True
        include_width_ratio = True
        include_multiplicity = True

    with st.expander("About fingerprint modes"):

        st.markdown("""
### Geometry only
Uses only ΔδH relationships between peaks.

Best for:
- mixtures
- overlapped spectra
- noisy experimental data

### Geometry + intensity
Adds relative intensity relationships.

Useful for:
- cleaner spectra
- semi-quantitative patterns

### Geometry + area
Adds integrated area relationships.

Useful for:
- quantitative-like spectra
- integrated peak lists

### Full relational
Uses:
- ΔδH
- intensity
- area
- width

Best for high-quality processed spectra.

### Multiplet aware
Also includes multiplicity relationships.

Best for curated reference databases.
""")

    h_bin_size = st.number_input(
        "¹H spacing bin size / ppm",
        min_value=0.001,
        max_value=0.100,
        value=0.010,
        step=0.001,
        format="%.3f",
        help="Controls the tolerance used when converting peak spacing into hash bins.",
    )

    min_delta_h = st.number_input(
        "Minimum ΔδH / ppm",
        min_value=0.001,
        max_value=1.0,
        value=0.020,
        step=0.001,
        format="%.3f",
        help="Rejects very small peak spacings that generate non-specific false matches.",
    )

    max_delta_h = st.number_input(
        "Maximum ΔδH / ppm",
        min_value=0.05,
        max_value=15.0,
        value=10.0,
        step=0.05,
    )

    max_neighbors_per_peak = st.slider(
        "Maximum neighbors per peak",
        min_value=2,
        max_value=200,
        value=50,
        step=1,
    )

    min_intensity_quantile = st.slider(
        "Remove weakest peaks by intensity quantile",
        min_value=0.0,
        max_value=0.9,
        value=0.0,
        step=0.05,
    )
    
    min_relative_intensity = st.slider(
        "Minimum relative intensity",
        min_value=0.0,
        max_value=0.20,
        value=0.02,
        step=0.005,
        format="%.3f",
        help="Removes peaks below this fraction of the most intense peak. Example: 0.02 keeps peaks >= 2% of max intensity.",
    )
    
    match_shift_tolerance = st.number_input(
        "Absolute δH match tolerance / ppm",
        min_value=0.001,
        max_value=0.500,
        value=0.050,
        step=0.001,
        format="%.3f",
        help="Requires unknown and reference matched peaks to be close in absolute chemical shift.",
    )

    st.divider()
    #st.subheader("Hash content")

    #include_intensity_ratio = st.checkbox("Include intensity ratio", value=True)
    intensity_ratio_bin_size = st.number_input(
        "Intensity ratio bin size, log10 scale",
        min_value=0.05,
        max_value=2.0,
        value=0.25,
        step=0.05,
    )

    #include_area_ratio = st.checkbox("Include area ratio", value=False)
    area_ratio_bin_size = st.number_input(
        "Area ratio bin size, log10 scale",
        min_value=0.05,
        max_value=2.0,
        value=0.25,
        step=0.05,
    )

    #include_width_ratio = st.checkbox("Include width ratio", value=False)
    width_ratio_bin_size = st.number_input(
        "Width ratio bin size, log10 scale",
        min_value=0.05,
        max_value=2.0,
        value=0.25,
        step=0.05,
    )

    #include_multiplicity = st.checkbox("Include multiplicity pair", value=False)

    use_absolute_pair_direction = st.checkbox(
        "Canonical pair direction",
        value=True,
        help="If active, A→B and B→A are treated as the same relationship.",
    )

    params = ProtonHashParameters(
        h_bin_size=h_bin_size,
        min_delta_h=min_delta_h,
        max_delta_h=max_delta_h,
        max_neighbors_per_peak=max_neighbors_per_peak,
        include_intensity_ratio=include_intensity_ratio,
        intensity_ratio_bin_size=intensity_ratio_bin_size,
        include_area_ratio=include_area_ratio,
        area_ratio_bin_size=area_ratio_bin_size,
        include_width_ratio=include_width_ratio,
        width_ratio_bin_size=width_ratio_bin_size,
        include_multiplicity=include_multiplicity,
        use_absolute_pair_direction=use_absolute_pair_direction,
        min_intensity_quantile=min_intensity_quantile,
        match_shift_tolerance=match_shift_tolerance,
        min_relative_intensity=min_relative_intensity,
    )





tab_library, tab_hashes, tab_search, tab_notes = st.tabs(
    [
        "1. Build database",
        "2. Generate fingerprints",
        "3. Search unknown / mixture",
        "Notes",
    ]
)


with tab_library:
    st.subheader("1. Build the ¹H NMR reference database")

    st.markdown(
        """
        Upload ¹H NMR peak lists as CSV, TSV, TXT, or ZIP.

        Minimum required column:
        - `delta_H`

        Recommended columns:
        - `compound_id`
        - `compound_name`
        - `peak_id`
        - `intensity`
        - `area`
        - `width`
        - `multiplicity`

        Optional HMDB structure mapping CSV:
        - `HMDB_ID`
        - `SMILES`
        - `INCHI_KEY`
        - `GENERIC_NAME`

        For BMRB, the first practical route is to export or convert the relevant ¹H chemical shift tables
        into CSV/TSV files. Direct NMR-STAR parsing can be added after we inspect the exact BMRB files.
        """
    )

    uploaded_library_files = st.file_uploader(
        "Upload reference ¹H NMR peak lists",
        type=["csv", "tsv", "txt", "zip", "str", "star"],
        accept_multiple_files=True,
        key="library_upload",
        help="Ex: hmdb_1h_peak_database_standardized",
    )

    uploaded_structure_file = st.file_uploader(
        "Optional: upload structure mapping CSV",
        type=["csv", "tsv", "txt"],
        accept_multiple_files=False,
        key="structure_mapping_upload",
        help="Expected columns: HMDB_ID, SMILES, INCHI_KEY, GENERIC_NAME. HMDB_ID will be matched to compound_id. Ex: hmdb_structures_mapping ",
    )

    structure_mapping = None
    if uploaded_structure_file is not None:
        try:
            structure_mapping = standardize_structure_mapping(read_table_from_upload(uploaded_structure_file))
            st.session_state["hmdb_structure_mapping"] = structure_mapping
            st.success(f"Loaded structure mapping for {len(structure_mapping)} HMDB IDs.")
        except Exception as e:
            st.error(f"Could not load structure mapping: {e}")

    if uploaded_library_files:
        try:
            peak_library = load_peak_tables_from_uploads(uploaded_library_files)
            structure_mapping = st.session_state.get("hmdb_structure_mapping", structure_mapping)
            peak_library = merge_structure_mapping(peak_library, structure_mapping)
            st.session_state["proton_peak_library"] = peak_library
            st.success(
                f"Loaded {len(peak_library)} peaks from "
                f"{peak_library['compound_id'].nunique()} reference compounds."
            )

            summary = (
                peak_library.groupby(["compound_id", "compound_name"])
                .agg(
                    n_peaks=("delta_H", "count"),
                    min_delta_H=("delta_H", "min"),
                    max_delta_H=("delta_H", "max"),
                    source_file=("source_file", "first"),
                    SMILES=("SMILES", "first"),
                    INCHI_KEY=("INCHI_KEY", "first"),
                    GENERIC_NAME=("GENERIC_NAME", "first"),
                )
                .reset_index()
            )

            st.write("Reference compound summary")
            st.dataframe(summary, use_container_width=True)

            with st.expander("Show standardized peak table", expanded=False):
                st.dataframe(peak_library, use_container_width=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.download_button(
                    "Download standardized database as CSV",
                    data=peak_library.to_csv(index=False).encode("utf-8"),
                    file_name="proton_nmr_peak_database_standardized.csv",
                    mime="text/csv",
                )
            with col2:
                st.plotly_chart(plot_reference_library_overview(peak_library), use_container_width=True)

        except Exception as e:
            st.error(f"Could not load reference files: {e}")
    else:
        st.info("Upload reference peak lists to start building the ¹H NMR database.")


with tab_hashes:
    st.subheader("2. Generate relational fingerprints")

    uploaded_hash_file = st.file_uploader(
        "Optionally upload an existing reference hash database CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="upload_existing_hash_database",
        help="Ex: proton_nmr_relational_hash_database_27052026_HMDB",
    )

    if uploaded_hash_file is not None:
        try:
            uploaded_hashes = read_table_from_upload(uploaded_hash_file)

            required_hash_cols = ["compound_id", "compound_name", "hash_value"]
            missing_hash_cols = [c for c in required_hash_cols if c not in uploaded_hashes.columns]

            if missing_hash_cols:
                st.error(f"Uploaded hash database is missing columns: {missing_hash_cols}")
            else:
                st.session_state["proton_hash_library"] = uploaded_hashes

                rarity_weights = compute_hash_rarity_weights(uploaded_hashes)

                weighted_hash_library = uploaded_hashes.merge(
                    rarity_weights,
                    on="hash_value",
                    how="left")

                reference_hash_counts = (
                    weighted_hash_library.groupby(["compound_id", "compound_name"])
                    .agg(
                        reference_hashes=("hash_value", "nunique"),
                        reference_weighted_hashes=("rarity_weight", "sum"),)
                        .reset_index())

                st.session_state["proton_rarity_weights"] = rarity_weights
                st.session_state["proton_reference_hash_counts"] = reference_hash_counts

                st.success(
                        f"Loaded existing hash database with "
                        f"{len(uploaded_hashes)} hash records and "
                        f"{uploaded_hashes['compound_id'].nunique()} reference compounds.")

                st.dataframe(uploaded_hashes, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load hash database: {e}")

    peak_library = st.session_state.get("proton_peak_library")

    if peak_library is None or peak_library.empty:
        st.warning("Load a reference database first.")
    else:
        st.write(f"Reference compounds: {peak_library['compound_id'].nunique()}")
        st.write(f"Reference peaks: {len(peak_library)}")

        if st.button("Generate ¹H NMR hash database", type="primary"):
            hash_library = generate_proton_hash_library(peak_library, params)
            st.session_state["proton_hash_library"] = hash_library

            if hash_library.empty:
                st.warning("No hashes were generated. Check the number of peaks and parameter limits.")
            else:
                rarity_weights = compute_hash_rarity_weights(hash_library)
                weighted_hash_library = hash_library.merge(rarity_weights, on="hash_value", how="left")

                reference_hash_counts = (
                    weighted_hash_library.groupby(["compound_id", "compound_name"])
                    .agg(
                        reference_hashes=("hash_value", "nunique"),
                        reference_weighted_hashes=("rarity_weight", "sum"),
                    )
                    .reset_index()
                )

                st.session_state["proton_rarity_weights"] = rarity_weights
                st.session_state["proton_reference_hash_counts"] = reference_hash_counts

                st.success(
                    f"Generated {len(hash_library)} pair records and "
                    f"{hash_library['hash_value'].nunique()} unique hashes."
                )

        hash_library = st.session_state.get("proton_hash_library")
        reference_hash_counts = st.session_state.get("proton_reference_hash_counts")
        rarity_weights = st.session_state.get("proton_rarity_weights")

        if hash_library is not None and not hash_library.empty:
            st.write("Reference hash summary")
            st.dataframe(reference_hash_counts, use_container_width=True)

            with st.expander("Show hash database", expanded=False):
                st.dataframe(hash_library, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "Download hash database CSV",
                    data=hash_library.to_csv(index=False).encode("utf-8"),
                    file_name="proton_nmr_relational_hash_database.csv",
                    mime="text/csv",
                )
            with col2:
                st.download_button(
                    "Download hash counts CSV",
                    data=reference_hash_counts.to_csv(index=False).encode("utf-8"),
                    file_name="proton_nmr_reference_hash_counts.csv",
                    mime="text/csv",
                )
            with col3:
                st.download_button(
                    "Download rarity weights CSV",
                    data=rarity_weights.to_csv(index=False).encode("utf-8"),
                    file_name="proton_nmr_hash_rarity_weights.csv",
                    mime="text/csv",
                )

            st.plotly_chart(plot_spacing_distribution(hash_library), use_container_width=True)

            compound_options = peak_library["compound_id"].drop_duplicates().tolist()
            selected_compound = st.selectbox("Visualize reference spectrum", compound_options)
            compound_df = peak_library[peak_library["compound_id"] == selected_compound]
            st.plotly_chart(
                plot_proton_peak_map(compound_df, f"Reference ¹H NMR peaks: {selected_compound}"),
                use_container_width=True,
            )


with tab_search:
    st.subheader("3. Search unknown ¹H NMR spectrum or mixture")

    hash_library = st.session_state.get("proton_hash_library")
    reference_hash_counts = st.session_state.get("proton_reference_hash_counts")
    rarity_weights = st.session_state.get("proton_rarity_weights")

    if hash_library is None or reference_hash_counts is None or rarity_weights is None:
        st.warning("Generate the reference hash database first.")
    else:
        st.markdown(
            """
            Upload an unknown ¹H NMR peak list. For mixtures, the file may contain many peaks from several compounds.
            The score reports which reference compounds have relational fingerprints represented in the unknown spectrum.
            """
        )

        unknown_file = st.file_uploader(
            "Upload unknown ¹H NMR peak list",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
            key="unknown_upload",
            help="Ex: 0001_SAND_peakList.csv",
        )

        if unknown_file is not None:
            try:
                raw_unknown = read_table_from_upload(unknown_file)
                unknown_peaks = standardize_proton_peak_table(
                    raw_unknown,
                    default_compound_id="unknown",
                    default_compound_name="unknown",
                    source_file=unknown_file.name,
                )
                with st.expander("DEBUG — check chemical shift columns"):
                    st.write("Raw uploaded columns:")
                    st.write(raw_unknown.columns.tolist())

                    st.write("Standardized unknown columns:")
                    st.write(unknown_peaks.columns.tolist())

                    st.write("delta_H summary:")
                    st.write(unknown_peaks["delta_H"].describe())

                    st.write("First delta_H values:")
                    st.write(unknown_peaks[["peak_id", "delta_H", "intensity"]].head(20))

                    if "anchor_delta_H" in hash_library.columns:
                        st.write("Reference hash anchor_delta_H summary:")
                        st.write(hash_library["anchor_delta_H"].describe())

                    if "target_delta_H" in hash_library.columns:
                        st.write("Reference hash target_delta_H summary:")
                        st.write(hash_library["target_delta_H"].describe())
                
                st.subheader("Search region filter")

                exclude_low_ppm = st.checkbox(
                    "Exclude very low ppm region, e.g. TMS/noise",
                    value=True,
                    key="exclude_low_ppm_search",
                )

                min_match_ppm = st.number_input(
                    "Minimum δH used for matching",
                    min_value=-1.0,
                    max_value=15.0,
                    value=0.5,
                    step=0.1,
                    key="min_match_ppm_search",
                )

                unknown_peaks_for_search = unknown_peaks.copy()
                # Remove weak experimental peaks
                if (
                    hasattr(params, "min_relative_intensity")
                    and params.min_relative_intensity > 0
                    and "intensity" in unknown_peaks_for_search.columns
                ):

                    max_intensity = unknown_peaks_for_search["intensity"].max()

                    if pd.notna(max_intensity) and max_intensity > 0:

                        intensity_cutoff = (
                            max_intensity * params.min_relative_intensity
                        )

                        unknown_peaks_for_search = unknown_peaks_for_search[
                            unknown_peaks_for_search["intensity"] >= intensity_cutoff
                        ].copy()

                if exclude_low_ppm:
                    unknown_peaks_for_search = unknown_peaks_for_search[
                        unknown_peaks_for_search["delta_H"] >= min_match_ppm
                    ].copy()

                st.write(f"Peaks used for search: {len(unknown_peaks_for_search)}")
                

                st.write(f"Unknown peaks: {len(unknown_peaks)}")
                st.dataframe(unknown_peaks, use_container_width=True)
                st.plotly_chart(plot_proton_peak_map(unknown_peaks, "Unknown ¹H NMR peak list"), use_container_width=True)

                if st.button("Search unknown against database", type="primary"):
                    scores = score_unknown_against_library(
                        unknown_peaks_for_search,
                        hash_library,
                        reference_hash_counts,
                        rarity_weights,
                        params,
                    )

                    st.session_state["last_unknown_peaks"] = unknown_peaks
                    st.session_state["last_unknown_peaks_for_search"] = unknown_peaks_for_search
                    st.session_state["last_search_scores"] = scores

                scores = st.session_state.get("last_search_scores")
                unknown_peaks_saved = st.session_state.get("last_unknown_peaks")
                unknown_peaks_for_search_saved = st.session_state.get("last_unknown_peaks_for_search")

                if unknown_peaks_for_search_saved is None and unknown_peaks_saved is not None:
                    unknown_peaks_for_search_saved = unknown_peaks_saved.copy()

                if (
                    scores is not None
                    and not scores.empty
                    and unknown_peaks_saved is not None
                    and unknown_peaks_for_search_saved is not None
                ):

                    st.success("Search completed.")
                    st.dataframe(scores, use_container_width=True)

                    fig = px.bar(
                        scores.head(25).sort_values("weighted_coverage_score"),
                        x="weighted_coverage_score",
                        y="compound_name",
                        orientation="h",
                        hover_data=[
                            "compound_id",
                            "matched_hashes",
                            "reference_hashes",
                            "coverage_score",
                            "weighted_matched_score",
                            "SMILES",
                            "INCHI_KEY",
                            "GENERIC_NAME",
                        ],
                        title="Top ¹H NMR relational fingerprint matches",
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        "Download search results CSV",
                        data=scores.to_csv(index=False).encode("utf-8"),
                        file_name="proton_nmr_search_results.csv",
                        mime="text/csv",
                    )

                    st.subheader("Matched relational hashes")

                    selected_hit = st.selectbox(
                        "Select reference compound to inspect matched hashes",
                        scores["compound_id"].astype(str).tolist(),
                        key="selected_hit_for_constellation",
                    )

                    max_plot_links = st.slider(
                        "Maximum matched relationships to load",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10,
                        key="max_plot_links_selected_hit",
                    )

                    matched_selected = get_matched_hash_records_for_compound(
                        unknown_peaks=unknown_peaks_for_search_saved,
                        library_hashes=hash_library,
                        selected_compound_id=selected_hit,
                        params=params,
                        max_records=max_plot_links,
                    )
                    display_cols = [
                        "compound_id_reference",
                        "compound_name_reference",
                        "anchor_delta_H_unknown",
                        "target_delta_H_unknown",
                        "anchor_delta_H_reference",
                        "target_delta_H_reference",
                        "delta_H_spacing_unknown",
                        "delta_H_spacing_reference",
                        "hash_value",
                    ]

                    display_cols = [
                        c for c in display_cols
                        if c in matched_selected.columns
                    ]

                    st.dataframe(
                        matched_selected[display_cols],
                        use_container_width=True,
                    )

                    if matched_selected.empty:
                        st.warning("No detailed matched hashes for this selected compound.")

                    else:
                        matched_selected["edge_label"] = (
                            matched_selected["hash_value"].astype(str)
                            + " | U: "
                            + matched_selected["anchor_delta_H_unknown"].round(3).astype(str)
                            + "→"
                            + matched_selected["target_delta_H_unknown"].round(3).astype(str)
                            + " ppm | R: "
                            + matched_selected["anchor_delta_H_reference"].round(3).astype(str)
                            + "→"
                            + matched_selected["target_delta_H_reference"].round(3).astype(str)
                        )


                        st.subheader("Filter constellation relationships")

                        available_edges = matched_selected["edge_label"].tolist()

                        selected_edges = st.multiselect(
                            "Select constellation edges to display",
                            available_edges,
                            default=available_edges[: min(20, len(available_edges))],
                            key=f"selected_constellation_edges_{selected_hit}",
                        )

                        matched_selected_plot = matched_selected[
                            matched_selected["edge_label"].isin(selected_edges)
                        ].copy()

                        #st.subheader("Matched constellation overlay on experimental spectrum")

                        st.subheader("Matched constellation network")



                        fig_network = plot_constellation_network(
                            matched_selected=matched_selected_plot,
                            selected_hit=selected_hit,
                        )

                        st.plotly_chart(fig_network, use_container_width=True)
                        with st.expander("How to interpret the matched constellation network"):
                            st.markdown("""
                        ### What this plot shows

                        This plot shows the **experimental peaks from the unknown spectrum** that participate in matched relational hashes for the selected reference compound.

                        Each blue point is an experimental peak.

                        Each line connects two experimental peaks whose relationship matched a relationship present in the reference compound.

                        ### What the lines mean

                        A line does **not** mean a chemical bond.

                        A line means:

                        ```text
                        this pair of experimental peaks has a spacing pattern compatible with the reference compound
                        """)
                        
                        st.subheader("Experimental spectrum with selected reference peaks marked")

                        fig_reference_peaks = plot_experimental_spectrum_with_reference_peaks(
                            unknown_peaks=unknown_peaks_saved,
                            matched_selected=matched_selected_plot,
                            selected_hit=selected_hit,
                        )

                        st.plotly_chart(fig_reference_peaks, use_container_width=True)

                elif scores is not None and scores.empty:
                    st.warning("No matches found with the current parameters.")


            except Exception as e:
                st.error(f"Could not process unknown file: {e}")


with tab_notes:
    st.subheader("Conceptual notes")

    st.markdown(
        """
        ## Why ¹H NMR is useful

        ¹H NMR is fast, cheap, and widely available. It is especially attractive for rapid screening
        and dereplication. The limitation is spectral overlap, especially in mixtures.

        ## Why intensity matters here

        In ¹H NMR, relative intensity or integrated area can add important information because peak
        ratios can reflect the number of equivalent protons. However, in mixtures, intensities can be
        distorted by concentration differences and overlap. For this reason, intensity should be useful
        but should not be the only matching criterion.

        ## Current fingerprint logic

        Each spectrum is transformed into many peak-pair relationships:

        ```text
        peak A → peak B = ΔδH + intensity ratio + optional area/width/multiplicity
        ```

        These relationships are quantized and converted into stable hashes.

        ## Mixture interpretation

        For mixtures, a compound does not need to have all reference hashes present. The search ranks
        compounds according to the fraction and rarity of reference hashes found in the unknown spectrum.

        ## Recommended first tests

        1. Compound pure vs compound pure.
        2. Same compound with simulated chemical-shift drift.
        3. Artificial mixture created by merging peak tables from 2–5 known compounds.
        4. Real mixture or extract.

        ## Next improvements

        1. Direct BMRB NMR-STAR parser.
        2. Simulated drift test module.
        3. Artificial mixture generator.
        4. Peak overlap handling.
        5. Multiplet-aware fingerprints.
        6. Joint scoring with HSQC fingerprints.
        """
    )
