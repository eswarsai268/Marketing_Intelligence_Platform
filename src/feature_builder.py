"""
Feature Builder for Olist E-commerce Customer Segmentation

Responsibilities:
1. Build customer-level features from cleaned Olist transaction data
2. Calculate RFM features:
   - Recency
   - Frequency
   - Monetary
3. Add useful behavioral features
4. Return a model-ready customer feature DataFrame

Expected input:
A merged transaction-level DataFrame containing at least:

    customer_unique_id
    order_id
    order_purchase_timestamp
    price
    freight_value

Optional columns:
    payment_value
    review_score
    product_id
    order_status
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS: List[str] = [
    "customer_unique_id",
    "order_id",
    "order_purchase_timestamp",
    "price",
    "freight_value",
]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Validate that all required columns are available.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feature builder is missing required columns: "
            f"{missing_columns}"
        )


# ============================================================
# DATE PREPARATION
# ============================================================

def prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert order purchase timestamp into datetime format.
    """

    df = df.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "customer_unique_id",
            "order_id",
            "order_purchase_timestamp",
        ]
    )

    return df


# ============================================================
# TRANSACTION VALUE
# ============================================================

def create_transaction_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total transaction value.

    total_value = product price + freight value

    If payment_value exists, it is retained separately.
    """

    df = df.copy()

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce",
    ).fillna(0)

    df["freight_value"] = pd.to_numeric(
        df["freight_value"],
        errors="coerce",
    ).fillna(0)

    df["total_item_value"] = (
        df["price"] + df["freight_value"]
    )

    return df


# ============================================================
# RFM FEATURES
# ============================================================

def build_rfm_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build customer-level RFM features.

    Recency:
        Number of days since customer's most recent order.

    Frequency:
        Number of unique orders.

    Monetary:
        Total amount spent.
    """

    if reference_date is None:
        reference_date = (
            df["order_purchase_timestamp"].max()
            + pd.Timedelta(days=1)
        )

    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            last_purchase_date=(
                "order_purchase_timestamp",
                "max",
            ),
            frequency=(
                "order_id",
                "nunique",
            ),
            monetary=(
                "total_item_value",
                "sum",
            ),
        )
        .reset_index()
    )

    rfm["recency"] = (
        reference_date
        - rfm["last_purchase_date"]
    ).dt.days

    return rfm


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

def build_behavior_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build additional customer behavior features.
    """

    grouped = df.groupby("customer_unique_id")

    behavior = grouped.agg(
        average_order_value=(
            "total_item_value",
            "mean",
        ),
        total_items=(
            "product_id",
            "count",
        )
        if "product_id" in df.columns
        else (
            "order_id",
            "count",
        ),
    ).reset_index()

    # --------------------------------------------------------
    # Review score
    # --------------------------------------------------------

    if "review_score" in df.columns:

        review_df = (
            df.groupby("customer_unique_id")[
                "review_score"
            ]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "review_score": "average_review_score"
                }
            )
        )

        behavior = behavior.merge(
            review_df,
            on="customer_unique_id",
            how="left",
        )

    # --------------------------------------------------------
    # Payment behavior
    # --------------------------------------------------------

    if "payment_value" in df.columns:

        payment_df = (
            df.groupby("customer_unique_id")[
                "payment_value"
            ]
            .sum()
            .reset_index()
            .rename(
                columns={
                    "payment_value": "total_payment_value"
                }
            )
        )

        behavior = behavior.merge(
            payment_df,
            on="customer_unique_id",
            how="left",
        )

    return behavior


# ============================================================
# CUSTOMER FEATURE BUILDER
# ============================================================

def build_customer_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Main feature-building function.

    Converts transaction-level data into
    one row per customer.
    """

    if df.empty:
        raise ValueError(
            "Cannot build features from an empty DataFrame."
        )

    validate_input_columns(df)

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    df = prepare_dates(df)

    df = create_transaction_value(df)

    if df.empty:
        raise ValueError(
            "No valid transaction records remain after cleaning."
        )

    # --------------------------------------------------------
    # Build RFM
    # --------------------------------------------------------

    rfm = build_rfm_features(
        df,
        reference_date=reference_date,
    )

    # --------------------------------------------------------
    # Build behavior features
    # --------------------------------------------------------

    behavior = build_behavior_features(df)

    # --------------------------------------------------------
    # Merge features
    # --------------------------------------------------------

    features = rfm.merge(
        behavior,
        on="customer_unique_id",
        how="left",
    )

    # --------------------------------------------------------
    # Clean numerical values
    # --------------------------------------------------------

    numeric_columns = features.select_dtypes(
        include=np.number
    ).columns

    features[numeric_columns] = (
        features[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    features[numeric_columns] = (
        features[numeric_columns]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Remove unnecessary date column
    # --------------------------------------------------------

    if "last_purchase_date" in features.columns:
        features = features.drop(
            columns=["last_purchase_date"]
        )

    return features


# ============================================================
# MODEL FEATURE SELECTION
# ============================================================

def get_model_features(
    customer_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select numerical features used by the ML model.

    customer_unique_id is retained separately
    and is NOT passed into the model.
    """

    excluded_columns = {
        "customer_unique_id",
    }

    model_features = customer_features.drop(
        columns=[
            column
            for column in excluded_columns
            if column in customer_features.columns
        ],
        errors="ignore",
    )

    # Keep numerical columns only
    model_features = model_features.select_dtypes(
        include=np.number
    )

    return model_features


# ============================================================
# FEATURE SUMMARY
# ============================================================

def get_feature_names(
    customer_features: pd.DataFrame,
) -> List[str]:
    """
    Return the names of numerical model features.
    """

    return list(
        get_model_features(customer_features).columns
    )