"""
ML Pipeline for Customer Segmentation

Responsibilities:
1. Load trained scaler and K-Means model
2. Predict a single customer
3. Predict customers from an uploaded/preprocessed DataFrame
4. Calculate dynamic dashboard KPIs

Important:
    low_review_flag has been REMOVED.

The trained ML model uses exactly these 5 features:

    Recency
    Frequency
    Monetary
    avg_review_score
    review_count

This module is designed to be imported by Streamlit.
"""

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

SCALER_PATH = MODEL_DIR / "scaler.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"


# ============================================================
# REQUIRED ML FEATURES
# ============================================================

# IMPORTANT:
# low_review_flag has been intentionally removed.

ML_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "avg_review_score",
    "review_count",
]


# ============================================================
# BUSINESS SEGMENT MAPPING
# ============================================================

SEGMENT_MAPPING = {

    1: {
        "name": "Champions / VIPs",
        "description": (
            "High-spending, frequent repeat customers "
            "with strong engagement."
        ),
    },

    2: {
        "name": "Recent One-Timers",
        "description": (
            "Recent customers with limited purchase history "
            "and strong review signals."
        ),
    },

    3: {
        "name": "At-Risk / Dissatisfied",
        "description": (
            "Customers showing signs of disengagement "
            "or poor experience."
        ),
    },

    0: {
        "name": "Churned / Lost",
        "description": (
            "Dormant customers with very low recent engagement."
        ),
    },
}


# ============================================================
# MODEL LOADING
# ============================================================

def _load_models():
    """
    Load the trained scaler and K-Means model.

    Returns:
        scaler
        kmeans_model

    Raises:
        FileNotFoundError if the model files do not exist.
    """

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}\n\n"
            "Make sure scaler.pkl exists inside the models folder."
        )

    if not KMEANS_PATH.exists():
        raise FileNotFoundError(
            f"K-Means model not found:\n{KMEANS_PATH}\n\n"
            "Make sure kmeans_model.pkl exists inside the models folder."
        )

    scaler = joblib.load(SCALER_PATH)
    kmeans_model = joblib.load(KMEANS_PATH)

    return scaler, kmeans_model


# ============================================================
# MODEL FEATURE COUNT CHECK
# ============================================================

def _validate_model_features(
    scaler,
    kmeans_model
) -> None:
    """
    Make sure the saved scaler and K-Means model
    were trained using exactly 5 features.

    This prevents an old 6-feature model containing
    low_review_flag from being used accidentally.
    """

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None
    )

    kmeans_features = getattr(
        kmeans_model,
        "n_features_in_",
        None
    )

    expected_features = len(ML_FEATURES)

    if (
        scaler_features is not None
        and scaler_features != expected_features
    ):
        raise ValueError(
            "The saved scaler was trained with "
            f"{scaler_features} features, but this pipeline "
            f"requires {expected_features} features.\n\n"
            "The model must be retrained after removing "
            "low_review_flag."
        )

    if (
        kmeans_features is not None
        and kmeans_features != expected_features
    ):
        raise ValueError(
            "The saved K-Means model was trained with "
            f"{kmeans_features} features, but this pipeline "
            f"requires {expected_features} features.\n\n"
            "The model must be retrained after removing "
            "low_review_flag."
        )


# ============================================================
# DATA VALIDATION
# ============================================================

def _validate_input_columns(
    df: pd.DataFrame
) -> None:
    """
    Check whether the DataFrame contains the
    five required ML features.

    low_review_flag is NOT required.
    """

    missing_columns = [
        column
        for column in ML_FEATURES
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Input data is missing required ML columns:\n"
            f"{missing_columns}\n\n"
            f"Required ML columns are:\n{ML_FEATURES}"
        )


def _validate_numeric_features(
    df: pd.DataFrame
) -> None:
    """
    Check that all ML features contain valid
    numeric values.

    NaN and infinite values are rejected.
    """

    for column in ML_FEATURES:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.isna().any():

            bad_count = int(
                converted.isna().sum()
            )

            raise ValueError(
                f"Column '{column}' contains "
                f"{bad_count} missing/non-numeric value(s)."
            )

        if np.isinf(
            converted.to_numpy()
        ).any():

            raise ValueError(
                f"Column '{column}' contains "
                "infinite values."
            )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def _prepare_model_input(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare exactly the five columns expected
    by the trained model.

    Extra CSV columns are ignored by the model.

    low_review_flag is intentionally ignored/removed.
    """

    _validate_input_columns(df)

    model_input = df[
        ML_FEATURES
    ].copy()

    # Convert everything to numeric
    for column in ML_FEATURES:

        model_input[column] = pd.to_numeric(
            model_input[column],
            errors="coerce"
        )

    _validate_numeric_features(
        model_input
    )

    return model_input


# ============================================================
# SEGMENT INFORMATION
# ============================================================

def _get_segment_info(
    cluster_id: int
) -> Dict[str, str]:
    """
    Convert numeric K-Means cluster ID into
    business-friendly information.
    """

    if cluster_id not in SEGMENT_MAPPING:

        return {
            "name": f"Cluster {cluster_id}",
            "description": (
                "Cluster produced by the trained "
                "K-Means model."
            ),
        }

    return SEGMENT_MAPPING[
        cluster_id
    ]


# ============================================================
# 1. SINGLE CUSTOMER INFERENCE
# ============================================================

def predict_single_customer(
    recency: float,
    frequency: int,
    monetary: float,
    avg_review_score: float = 3.0,
    review_count: int = 0,
) -> Dict[str, Any]:
    """
    Predict the segment for ONE customer.

    This function can be used by Streamlit
    What-If analysis.

    Parameters:
        recency:
            Days since customer's last order.

        frequency:
            Number of orders.

        monetary:
            Customer spending/value.

        avg_review_score:
            Average review score.

            Default = 3.0 when review information
            is unavailable.

        review_count:
            Number of reviews.

            Default = 0 when review information
            is unavailable.

    IMPORTANT:
        low_review_flag is no longer used.
    """

    # --------------------------------------------------------
    # Load trained assets
    # --------------------------------------------------------

    scaler, kmeans_model = _load_models()

    # --------------------------------------------------------
    # Check that the model is the new 5-feature model
    # --------------------------------------------------------

    _validate_model_features(
        scaler,
        kmeans_model
    )

    # --------------------------------------------------------
    # Create one customer row
    # --------------------------------------------------------

    customer_df = pd.DataFrame(
        [
            {
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "avg_review_score": avg_review_score,
                "review_count": review_count,
            }
        ]
    )

    # --------------------------------------------------------
    # Prepare exactly 5 model features
    # --------------------------------------------------------

    model_input = _prepare_model_input(
        customer_df
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use transform(), NOT fit_transform().
    # --------------------------------------------------------

    scaled_customer = scaler.transform(
        model_input
    )

    # --------------------------------------------------------
    # Predict cluster
    # --------------------------------------------------------

    prediction = kmeans_model.predict(
        scaled_customer
    )

    cluster_id = int(
        prediction[0]
    )

    # --------------------------------------------------------
    # Convert cluster to business information
    # --------------------------------------------------------

    segment_info = _get_segment_info(
        cluster_id
    )

    return {
        "cluster_id": cluster_id,
        "segment_name": segment_info["name"],
        "description": segment_info["description"],
    }


# ============================================================
# 2. BATCH CSV / DATAFRAME PREDICTION
# ============================================================

def batch_predict_csv(
    input_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict customer segments for an entire DataFrame.

    Expected ML columns:

        Recency
        Frequency
        Monetary
        avg_review_score
        review_count

    Extra columns are allowed.

    The function returns the original DataFrame
    plus:

        Predicted_Cluster_ID
        Segment_Name
    """

    # --------------------------------------------------------
    # Basic type validation
    # --------------------------------------------------------

    if not isinstance(
        input_df,
        pd.DataFrame
    ):

        raise TypeError(
            "input_df must be a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Empty file check
    # --------------------------------------------------------

    if input_df.empty:

        raise ValueError(
            "The uploaded DataFrame is empty."
        )

    # --------------------------------------------------------
    # Load trained assets
    # --------------------------------------------------------

    scaler, kmeans_model = _load_models()

    # --------------------------------------------------------
    # Make sure this is the new 5-feature model
    # --------------------------------------------------------

    _validate_model_features(
        scaler,
        kmeans_model
    )

    # --------------------------------------------------------
    # Prepare exactly the five model features
    # --------------------------------------------------------

    model_input = _prepare_model_input(
        input_df
    )

    # --------------------------------------------------------
    # Scale using the TRAINED scaler
    # --------------------------------------------------------

    scaled_data = scaler.transform(
        model_input
    )

    # --------------------------------------------------------
    # Predict all customers
    # --------------------------------------------------------

    predictions = kmeans_model.predict(
        scaled_data
    )

    # --------------------------------------------------------
    # Copy original data
    # --------------------------------------------------------

    result_df = input_df.copy()

    # --------------------------------------------------------
    # Add cluster ID
    # --------------------------------------------------------

    result_df[
        "Predicted_Cluster_ID"
    ] = predictions.astype(int)

    # --------------------------------------------------------
    # Add business segment
    # --------------------------------------------------------

    segment_name_map = {
        cluster_id: info["name"]
        for cluster_id, info
        in SEGMENT_MAPPING.items()
    }

    result_df[
        "Segment_Name"
    ] = (
        result_df[
            "Predicted_Cluster_ID"
        ]
        .map(segment_name_map)
        .fillna(
            result_df[
                "Predicted_Cluster_ID"
            ].apply(
                lambda x:
                f"Cluster {x}"
            )
        )
    )

    return result_df


# ============================================================
# 3. DYNAMIC KPI AGGREGATOR
# ============================================================

def get_dashboard_kpis(
    predicted_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate dashboard metrics from the DataFrame
    generated by batch_predict_csv().

    Returns:

        total_customers
        average_monetary
        segment_distribution
        segment_summary
    """

    # --------------------------------------------------------
    # Type validation
    # --------------------------------------------------------

    if not isinstance(
        predicted_df,
        pd.DataFrame
    ):

        raise TypeError(
            "predicted_df must be a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Empty DataFrame
    # --------------------------------------------------------

    if predicted_df.empty:

        raise ValueError(
            "Cannot calculate KPIs from an empty DataFrame."
        )

    # --------------------------------------------------------
    # Required columns for KPI calculation
    # --------------------------------------------------------

    required_columns = (
        ML_FEATURES
        + [
            "Predicted_Cluster_ID",
            "Segment_Name",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in predicted_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "predicted_df is missing required columns:\n"
            f"{missing_columns}\n\n"
            "Run batch_predict_csv() first."
        )

    # --------------------------------------------------------
    # Total customers
    # --------------------------------------------------------

    total_customers = len(
        predicted_df
    )

    # --------------------------------------------------------
    # Average monetary value
    # --------------------------------------------------------

    average_monetary = float(
        pd.to_numeric(
            predicted_df["Monetary"],
            errors="coerce"
        ).mean()
    )

    # --------------------------------------------------------
    # Segment distribution
    # --------------------------------------------------------

    segment_counts = (
        predicted_df[
            "Segment_Name"
        ]
        .value_counts()
    )

    segment_distribution = {}

    for (
        segment_name,
        count
    ) in segment_counts.items():

        percentage = (
            count
            / total_customers
        ) * 100

        segment_distribution[
            segment_name
        ] = {
            "count": int(count),
            "percentage": round(
                float(percentage),
                2
            ),
        }

    # --------------------------------------------------------
    # Segment-level RFM summary
    # --------------------------------------------------------

    segment_summary = (
        predicted_df
        .groupby(
            "Segment_Name"
        )[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ]
        .mean()
        .round(2)
        .reset_index()
    )

    # --------------------------------------------------------
    # Return dashboard data
    # --------------------------------------------------------

    return {
        "total_customers": total_customers,

        "average_monetary": round(
            average_monetary,
            2
        ),

        "segment_distribution":
            segment_distribution,

        "segment_summary":
            segment_summary,
    }