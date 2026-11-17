"""Generate a reproducible, fictional shopping transaction dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/ecommerce.csv"
DEFAULT_SEED = 20251117
IMAGE_SHAPE = (8, 8)
CATEGORIES = ("의류", "신발", "가방", "생활용품", "전자기기")

# Each profile specifies order count, day offsets, and product price preference.
PROFILES = (
    ((14, 19), (0, 365), "high"),
    ((7, 11), (0, 365), "middle"),
    ((1, 3), (335, 365), "any"),
    ((4, 7), (0, 245), "any"),
)


def make_catalog(rng):
    """Create 100 fictional SKUs and encode one grayscale array per SKU."""
    categories = np.repeat(CATEGORIES, 20)
    minimum_prices = np.repeat([15000, 30000, 25000, 5000, 50000], 20)
    maximum_prices = np.repeat([90000, 140000, 125000, 40000, 250000], 20)
    prices = rng.integers(minimum_prices // 1000, maximum_prices // 1000) * 1000

    y, x = np.indices(IMAGE_SHAPE)
    patterns = np.stack(
        [x > 2, y > 2, x + y > 7, (x - 3) ** 2 + (y - 3) ** 2 < 10, (x + y) % 2]
    )
    pixels = 45 + 145 * np.repeat(patterns, 20, axis=0)
    pixels = np.clip(pixels + rng.integers(-20, 21, size=pixels.shape), 0, 255)

    return pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(1, 101)],
        "product_name": [
            f"가상 {category} {i:03d}" + (" 기본형" if i % 3 else "")
            for i, category in enumerate(categories, 1)
        ],
        "category": categories,
        "price": prices,
        "image_array": [json.dumps(image.tolist(), separators=(",", ":")) for image in pixels],
    })


def generate_transactions(seed=DEFAULT_SEED):
    """Return one row per order line with a fixed catalog and 500 customers."""
    rng = np.random.default_rng(seed)
    catalog = make_catalog(rng)
    sorted_products = np.argsort(catalog["price"].to_numpy())
    product_pools = {
        "high": sorted_products[-45:],
        "middle": sorted_products[15:75],
        "any": sorted_products,
    }
    start = pd.Timestamp("2025-01-01")
    profiles = rng.permutation(np.repeat(np.arange(len(PROFILES)), 125))
    rows = []
    order_number = 0

    for customer_number, profile_index in enumerate(profiles, 1):
        order_range, day_range, price_preference = PROFILES[profile_index]
        order_count = rng.integers(*order_range)
        days = np.sort(rng.integers(*day_range, size=order_count))
        if price_preference in ("high", "middle"):
            days[-1] = rng.integers(320, 365)

        for day in days:
            order_number += 1
            product_indices = rng.choice(
                product_pools[price_preference], size=rng.integers(1, 4), replace=False
            )
            for product_index in product_indices:
                rows.append((
                    f"C{customer_number:04d}",
                    f"O{order_number:06d}",
                    start + pd.Timedelta(days=int(day)),
                    product_index,
                    rng.integers(1, 4),
                ))

    transactions = pd.DataFrame(
        rows, columns=["customer_id", "order_id", "order_date", "sku", "quantity"]
    ).join(catalog, on="sku").drop(columns="sku")
    # Reserve the observation endpoints even when the cohort assignment changes.
    earliest_order = transactions.loc[transactions["order_date"].idxmin(), "order_id"]
    transactions.loc[transactions["order_id"].eq(earliest_order), "order_date"] = start
    latest_order = transactions.loc[transactions["order_date"].idxmax(), "order_id"]
    transactions.loc[transactions["order_id"].eq(latest_order), "order_date"] = (
        start + pd.Timedelta(days=364)
    )

    missing_rows = rng.choice(transactions.index, size=100, replace=False)
    bulk_rows = rng.choice(transactions.index, size=24, replace=False)
    transactions.loc[missing_rows, "product_name"] = pd.NA
    transactions.loc[bulk_rows, "quantity"] = rng.integers(20, 61, size=len(bulk_rows))
    transactions["amount"] = transactions["price"] * transactions["quantity"]
    return transactions[[
        "customer_id", "order_id", "order_date", "product_id", "product_name", "category",
        "price", "quantity", "amount", "image_array",
    ]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    arguments = parser.parse_args()
    transactions = generate_transactions(arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(arguments.output, index=False, date_format="%Y-%m-%d")
    print(f"{len(transactions):,} lines, {transactions['order_id'].nunique():,} orders")
    print(arguments.output)


if __name__ == "__main__":
    main()
