"""
ML Pipeline for Customer Segmentation

Responsibilities:
1. Load trained scaler and K-Means model
2. Predict a single customer
3. Predict customers from an uploaded DataFrame
4. Calculate dynamic dashboard KPIs

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

# Project root:
# dataset_training/
# ├── models/
# ├── data/
# └── src/
#
# Since this file is inside src/, parent.parent = project root.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

SCALER_PATH = MODEL_DIR / "scaler.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"


# ============================================================
# REQUIRED ML FEATURES
# ============================================================

ML_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "avg_review_score",
    "review_count",
    "low_review_flag",
]


# ============================================================
# BUSINESS SEGMENT MAPPING
# ============================================================

# K-Means cluster IDs are converted into
# business-friendly segment names.

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
        FileNotFoundError if the .pkl files do not exist.
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
# DATA VALIDATION
# ============================================================

def _validate_input_columns(df: pd.DataFrame) -> None:
    """
    Check whether the DataFrame contains all six
    required ML features.
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
            f"Required columns are:\n{ML_FEATURES}"
        )


def _validate_numeric_features(df: pd.DataFrame) -> None:
    """
    Check that all ML features contain numeric values
    and do not contain NaN or infinite values.
    """

    for column in ML_FEATURES:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.isna().any():

            bad_count = converted.isna().sum()

            raise ValueError(
                f"Column '{column}' contains "
                f"{bad_count} missing/non-numeric value(s). "
                "Please clean the data before prediction."
            )

        if np.isinf(converted.to_numpy()).any():

            raise ValueError(
                f"Column '{column}' contains infinite values."
            )


# ============================================================
# SEGMENT INFORMATION
# ============================================================

def _get_segment_info(cluster_id: int) -> Dict[str, str]:
    """
    Convert numeric K-Means cluster ID into
    business-friendly information.
    """

    if cluster_id not in SEGMENT_MAPPING:

        return {
            "name": f"Cluster {cluster_id}",
            "description": (
                "Cluster produced by the trained K-Means model."
            ),
        }

    return SEGMENT_MAPPING[cluster_id]


# ============================================================
# 1. SINGLE CUSTOMER INFERENCE
# ============================================================

def predict_single_customer(
    recency: float,
    frequency: int,
    monetary: float,
    avg_review_score: float,
    review_count: int,
    low_review_flag: int,
) -> Dict[str, Any]:
    """
    Predict the segment for ONE customer.

    This function is intended for Streamlit What-If sliders.

    Parameters:
        recency:
            Days since customer's last order.

        frequency:
            Number of orders.

        monetary:
            Customer spending/value.

        avg_review_score:
            Average review score.

        review_count:
            Number of reviews.

        low_review_flag:
            Binary low-review indicator.

    Returns:
        Dictionary containing:
            cluster_id
            segment_name
            description
    """

    # Load trained assets
    scaler, kmeans_model = _load_models()

    # Create one customer row
    customer_df = pd.DataFrame(
        [
            {
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "avg_review_score": avg_review_score,
                "review_count": review_count,
                "low_review_flag": low_review_flag,
            }
        ]
    )

    # Validate
    _validate_input_columns(customer_df)
    _validate_numeric_features(customer_df)

    # IMPORTANT:
    # Use transform(), NOT fit_transform().
    scaled_array = scaler.transform(
        customer_df[ML_FEATURES]
    )

    # Wrap it back into a DataFrame to silence the warning
    scaled_customer = pd.DataFrame(
        scaled_array, 
        columns=ML_FEATURES
    )

    # Predict cluster
    prediction = kmeans_model.predict(
        scaled_customer
    )

    cluster_id = int(prediction[0])

    # Convert cluster into business information
    segment_info = _get_segment_info(cluster_id)

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

    Required columns:

        Recency
        Frequency
        Monetary
        avg_review_score
        review_count
        low_review_flag

    Returns:
        Original DataFrame plus:

        Predicted_Cluster_ID
        Segment_Name
    """

    if not isinstance(input_df, pd.DataFrame):

        raise TypeError(
            "input_df must be a pandas DataFrame."
        )

    if input_df.empty:

        raise ValueError(
            "The uploaded DataFrame is empty."
        )

    # Load trained assets
    scaler, kmeans_model = _load_models()

    # Validate columns
    _validate_input_columns(input_df)

    # Validate numeric values
    _validate_numeric_features(input_df)

    # Make a copy so original data isn't changed
    result_df = input_df.copy()

    # Scale using the trained scaler
    scaled_data = scaler.transform(
        result_df[ML_FEATURES]
    )


    # Predict all customers
    predictions = kmeans_model.predict(
        scaled_data
    )

    # Add cluster ID
    result_df["Predicted_Cluster_ID"] = (
        predictions.astype(int)
    )

    # Add business segment
    result_df["Segment_Name"] = (
        result_df["Predicted_Cluster_ID"]
        .map(
            {
                cluster_id: info["name"]
                for cluster_id, info
                in SEGMENT_MAPPING.items()
            }
        )
        .fillna(
            result_df["Predicted_Cluster_ID"]
            .apply(
                lambda x: f"Cluster {x}"
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

    if not isinstance(predicted_df, pd.DataFrame):

        raise TypeError(
            "predicted_df must be a pandas DataFrame."
        )

    if predicted_df.empty:

        raise ValueError(
            "Cannot calculate KPIs from an empty DataFrame."
        )

    # Required columns
    required_columns = ML_FEATURES + [
        "Predicted_Cluster_ID",
        "Segment_Name",
    ]

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

    # Total customers
    total_customers = len(predicted_df)

    # Average monetary value
    average_monetary = float(
        predicted_df["Monetary"].mean()
    )

    # Segment distribution
    segment_counts = (
        predicted_df["Segment_Name"]
        .value_counts()
    )

    segment_distribution = {}

    for segment_name, count in segment_counts.items():

        percentage = (
            count / total_customers
        ) * 100

        segment_distribution[segment_name] = {
            "count": int(count),
            "percentage": round(
                float(percentage),
                2
            ),
        }

    # Segment-level RFM summary
    segment_summary = (
        predicted_df
        .groupby("Segment_Name")[
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

    return {
        "total_customers": total_customers,
        "average_monetary": average_monetary,
        "segment_distribution": segment_distribution,
        "segment_summary": segment_summary,
    }