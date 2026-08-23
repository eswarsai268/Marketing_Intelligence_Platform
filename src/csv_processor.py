"""
CSV Processor for Customer Segmentation

Responsibilities:
1. Read uploaded CSV files
2. Clean completely empty rows/columns
3. Analyze the uploaded data
4. Automatically map columns
5. Build the ML-ready customer features
6. Support different CSV structures
7. Prepare data for ml_pipeline.py

Important:
    - No minimum row restriction
    - Extra columns are allowed
    - low_review_flag is NOT used
    - The ML model ultimately receives 5 features
"""

from typing import Any, Dict

import pandas as pd

from .column_mapper import map_columns
from .feature_builder import (
    build_features,
    ML_FEATURES,
)


# ============================================================
# CUSTOM ERROR
# ============================================================

class CSVProcessorError(Exception):
    """
    Error raised when the CSV can be read but
    cannot be prepared for customer analysis.
    """

    pass


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(
    file_source,
) -> pd.DataFrame:
    """
    Read an uploaded CSV file.

    The function tries normal UTF-8 first and then
    falls back to latin1 if required.

    Parameters:
        file_source:
            Streamlit UploadedFile, file path,
            or another pandas-compatible source.

    Returns:
        Cleaned DataFrame.
    """

    try:

        # ----------------------------------------------------
        # First attempt: normal UTF-8 CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            file_source
        )

    except UnicodeDecodeError:

        # ----------------------------------------------------
        # If the file has a different encoding,
        # reset the file position when possible.
        # ----------------------------------------------------

        if hasattr(
            file_source,
            "seek",
        ):
            file_source.seek(0)

        try:

            df = pd.read_csv(
                file_source,
                encoding="latin1",
            )

        except Exception as exc:

            raise CSVProcessorError(
                "The CSV could not be read because "
                "its text encoding could not be processed."
            ) from exc

    except pd.errors.EmptyDataError as exc:

        raise CSVProcessorError(
            "The uploaded CSV contains no data."
        ) from exc

    except pd.errors.ParserError as exc:

        raise CSVProcessorError(
            "The CSV structure could not be parsed. "
            "Some rows may have a different number "
            "of fields than the others."
        ) from exc

    except Exception as exc:

        raise CSVProcessorError(
            f"The uploaded file could not be read: {exc}"
        ) from exc

    # ========================================================
    # BASIC CLEANING
    # ========================================================

    # Remove completely empty rows.
    df = df.dropna(
        axis=0,
        how="all",
    )

    # Remove completely empty columns.
    df = df.dropna(
        axis=1,
        how="all",
    )

    # Reset index after cleaning.
    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # No usable data
    # --------------------------------------------------------

    if df.empty:

        raise CSVProcessorError(
            "The CSV was read successfully, "
            "but it contains no usable records."
        )

    return df


# ============================================================
# DATASET PROFILE
# ============================================================

def profile_dataframe(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Analyze the uploaded CSV.

    This information can be displayed by Streamlit
    so the user can understand what happened to
    their uploaded data.
    """

    # Missing values
    missing_values = (
        df.isna()
        .sum()
        .to_dict()
    )

    # Duplicate rows
    duplicate_rows = int(
        df.duplicated().sum()
    )

    # Data types
    data_types = {
        column: str(dtype)
        for column, dtype
        in df.dtypes.items()
    }

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": (
            df.columns.tolist()
        ),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "data_types": data_types,
    }


# ============================================================
# CHECK RFM AVAILABILITY
# ============================================================

def _check_rfm_availability(
    mapping: Dict[str, str],
) -> None:
    """
    Check whether the information required to build
    RFM features is available.

    RFM means:

        Recency
        Frequency
        Monetary
    """

    required_rfm = [
        "Recency",
        "Frequency",
        "Monetary",
    ]

    missing_rfm = [
        feature
        for feature in required_rfm
        if feature not in mapping
    ]

    if missing_rfm:

        raise CSVProcessorError(
            "The uploaded data does not contain enough "
            "customer purchase information to build "
            "the required RFM features.\n\n"
            f"Missing: {missing_rfm}\n\n"
            "The system needs Recency, Frequency and "
            "Monetary information to perform customer "
            "segmentation."
        )


# ============================================================
# PROCESS CSV
# ============================================================

def process_csv(
    file_source,
) -> Dict[str, Any]:
    """
    Complete CSV processing pipeline.

    Flow:

        CSV
         ↓
        Load
         ↓
        Clean empty rows/columns
         ↓
        Profile dataset
         ↓
        Automatically map columns
         ↓
        Check RFM availability
         ↓
        Build 5 ML features
         ↓
        Return ML-ready data
    """

    # ========================================================
    # STEP 1: LOAD CSV
    # ========================================================

    original_df = load_csv(
        file_source
    )

    # ========================================================
    # STEP 2: PROFILE DATA
    # ========================================================

    profile = profile_dataframe(
        original_df
    )

    # ========================================================
    # STEP 3: AUTOMATIC COLUMN MAPPING
    # ========================================================

    try:

        mapping_result = map_columns(
            original_df
        )

    except Exception as exc:

        raise CSVProcessorError(
            "The CSV was read successfully, "
            "but its columns could not be analyzed."
        ) from exc

    mapping = mapping_result.get(
        "mapping",
        {}
    )

    confidence = mapping_result.get(
        "confidence",
        {}
    )

    # ========================================================
    # STEP 4: CHECK RFM
    # ========================================================

    _check_rfm_availability(
        mapping
    )

    # ========================================================
    # STEP 5: BUILD ML FEATURES
    # ========================================================

    try:

        ml_data = build_features(
            original_df,
            mapping,
        )

    except ValueError as exc:

        raise CSVProcessorError(
            "The CSV was read successfully, "
            "but the customer features could not "
            f"be prepared for the ML model.\n\n{exc}"
        ) from exc

    except Exception as exc:

        raise CSVProcessorError(
            "An unexpected problem occurred while "
            "preparing the customer features."
        ) from exc

    # ========================================================
    # STEP 6: FINAL FEATURE CHECK
    # ========================================================

    missing_ml_features = [
        feature
        for feature in ML_FEATURES
        if feature not in ml_data.columns
    ]

    if missing_ml_features:

        raise CSVProcessorError(
            "Feature preparation completed, "
            "but some ML features are missing:\n"
            f"{missing_ml_features}"
        )

    # ========================================================
    # STEP 7: RETURN EVERYTHING
    # ========================================================

    return {

        # Original uploaded data
        "original_data": original_df,

        # Dataset information
        "profile": profile,

        # Automatically detected columns
        "mapping": mapping,

        # Mapping confidence
        "confidence": confidence,

        # ML-ready customer-level data
        "ml_data": ml_data,

        # Exactly the features expected by ML pipeline
        "ml_features": ML_FEATURES,

        # Processing status
        "status": "ready",
    }