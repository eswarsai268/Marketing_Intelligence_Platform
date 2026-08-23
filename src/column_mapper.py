"""
Column Mapper for Olist E-commerce Dataset

Responsibilities:
1. Standardize raw Olist CSV column names
2. Map dataset-specific columns to common internal names
3. Keep mappings centralized for the entire ML pipeline
4. Validate required columns
5. Support the CSV processor and downstream ML pipeline
"""

from typing import Dict, List
import pandas as pd


# ============================================================
# COLUMN MAPPINGS
# ============================================================

COLUMN_MAPPINGS: Dict[str, Dict[str, str]] = {

    # --------------------------------------------------------
    # Customers
    # olist_customers_dataset.csv
    # --------------------------------------------------------
    "customers": {
        "customer_id": "customer_id",
        "customer_unique_id": "customer_unique_id",
        "customer_zip_code_prefix": "customer_zip_code",
        "customer_city": "customer_city",
        "customer_state": "customer_state",
    },

    # --------------------------------------------------------
    # Orders
    # olist_orders_dataset.csv
    # --------------------------------------------------------
    "orders": {
        "order_id": "order_id",
        "customer_id": "customer_id",
        "order_status": "order_status",
        "order_purchase_timestamp": "order_purchase_timestamp",
        "order_approved_at": "order_approved_at",
        "order_delivered_carrier_date": "order_delivered_carrier_date",
        "order_delivered_customer_date": "order_delivered_customer_date",
        "order_estimated_delivery_date": "order_estimated_delivery_date",
    },

    # --------------------------------------------------------
    # Order Items
    # olist_order_items_dataset.csv
    # --------------------------------------------------------
    "order_items": {
        "order_id": "order_id",
        "order_item_id": "order_item_id",
        "product_id": "product_id",
        "seller_id": "seller_id",
        "shipping_limit_date": "shipping_limit_date",
        "price": "price",
        "freight_value": "freight_value",
    },

    # --------------------------------------------------------
    # Payments
    # olist_order_payments_dataset.csv
    # --------------------------------------------------------
    "payments": {
        "order_id": "order_id",
        "payment_sequential": "payment_sequential",
        "payment_type": "payment_type",
        "payment_installments": "payment_installments",
        "payment_value": "payment_value",
    },

    # --------------------------------------------------------
    # Reviews
    # olist_order_reviews_dataset.csv
    # --------------------------------------------------------
    "reviews": {
        "review_id": "review_id",
        "order_id": "order_id",
        "review_score": "review_score",
        "review_comment_title": "review_comment_title",
        "review_comment_message": "review_comment_message",
        "review_creation_date": "review_creation_date",
        "review_answer_timestamp": "review_answer_timestamp",
    },

    # --------------------------------------------------------
    # Products
    # olist_products_dataset.csv
    # --------------------------------------------------------
    "products": {
        "product_id": "product_id",
        "product_category_name": "product_category",
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
        "product_photos_qty": "product_photos_count",
        "product_weight_g": "product_weight_g",
        "product_length_cm": "product_length_cm",
        "product_height_cm": "product_height_cm",
        "product_width_cm": "product_width_cm",
    },

    # --------------------------------------------------------
    # Sellers
    # olist_sellers_dataset.csv
    # --------------------------------------------------------
    "sellers": {
        "seller_id": "seller_id",
        "seller_zip_code_prefix": "seller_zip_code",
        "seller_city": "seller_city",
        "seller_state": "seller_state",
    },

    # --------------------------------------------------------
    # Product Category Translation
    # product_category_name_translation.csv
    # --------------------------------------------------------
    "category_translation": {
        "product_category_name": "product_category",
        "product_category_name_english": "product_category_english",
    },

    # --------------------------------------------------------
    # Geolocation
    # olist_geolocation_dataset.csv
    # --------------------------------------------------------
    "geolocation": {
        "geolocation_zip_code_prefix": "zip_code",
        "geolocation_lat": "latitude",
        "geolocation_lng": "longitude",
        "geolocation_city": "city",
        "geolocation_state": "state",
    },
}


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS: Dict[str, List[str]] = {

    "customers": [
        "customer_id",
        "customer_unique_id",
    ],

    "orders": [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
    ],

    "order_items": [
        "order_id",
        "product_id",
        "price",
        "freight_value",
    ],

    "payments": [
        "order_id",
        "payment_value",
    ],

    "reviews": [
        "order_id",
        "review_score",
    ],

    "products": [
        "product_id",
        "product_category_name",
    ],
}


# ============================================================
# FUNCTIONS
# ============================================================

def get_column_mapping(dataset_name: str) -> Dict[str, str]:
    """
    Return the column mapping for a specific Olist dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier such as:
        customers, orders, order_items, payments, etc.

    Returns
    -------
    Dict[str, str]
        Raw column -> standardized column mapping.
    """

    dataset_name = dataset_name.lower().strip()

    if dataset_name not in COLUMN_MAPPINGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {list(COLUMN_MAPPINGS.keys())}"
        )

    return COLUMN_MAPPINGS[dataset_name].copy()


def map_columns(
    df: pd.DataFrame,
    dataset_name: str,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Rename raw Olist columns into standardized internal names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    dataset_name : str
        Dataset identifier.

    validate : bool
        Whether to validate required columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """

    mapping = get_column_mapping(dataset_name)

    # Only rename columns that actually exist
    existing_mapping = {
        old: new
        for old, new in mapping.items()
        if old in df.columns
    }

    mapped_df = df.rename(columns=existing_mapping)

    if validate:
        validate_columns(mapped_df, dataset_name)

    return mapped_df


def validate_columns(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Validate that required standardized columns exist.
    """

    required = REQUIRED_COLUMNS.get(dataset_name, [])

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns for '{dataset_name}': "
            f"{missing}"
        )


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic normalization for unexpected CSV column formatting.

    Example:
        ' Order_ID ' -> 'order_id'
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def get_available_datasets() -> List[str]:
    """
    Return all supported Olist dataset names.
    """

    return list(COLUMN_MAPPINGS.keys())