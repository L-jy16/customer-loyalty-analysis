"""Generate a reproducible, fictional shopping transaction dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# 어느 폴더에서 실행해도 데이터를 찾을 수 있도록 프로젝트 위치를 정합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/ecommerce.csv"
DEFAULT_SEED = 20251117
IMAGE_SHAPE = (8, 8)
CATEGORIES = ("의류", "신발", "가방", "생활용품", "전자기기")

# 고객마다 구매 횟수, 구매 시기, 선호하는 가격대가 다르도록 설정합니다.
# 예를 들어 (14, 19)는 14~18회 주문한다는 뜻입니다. 마지막 숫자 19는 포함하지 않습니다.
PROFILES = (
    ((14, 19), (0, 365), "high"),
    ((7, 11), (0, 365), "middle"),
    ((1, 3), (335, 365), "any"),
    ((4, 7), (0, 245), "any"),
)


def make_catalog(rng):
    """Create 100 fictional SKUs and encode one grayscale array per SKU."""
    # 의류, 신발 등 5개 종류에 상품을 20개씩 만들어 총 100개를 준비합니다.
    categories = np.repeat(CATEGORIES, 20)
    minimum_prices = np.repeat([15000, 30000, 25000, 5000, 50000], 20)
    maximum_prices = np.repeat([90000, 140000, 125000, 40000, 250000], 20)
    prices = rng.integers(minimum_prices // 1000, maximum_prices // 1000) * 1000

    # 작은 칸마다 밝기 숫자를 넣어 도형 이미지를 만듭니다. 밝기에 무작위 차이를 주어 이미지를 조금씩 다르게 만듭니다.
    y, x = np.indices(IMAGE_SHAPE)
    patterns = np.stack(
        [x > 2, y > 2, x + y > 7, (x - 3) ** 2 + (y - 3) ** 2 < 10, (x + y) % 2]
    )
    pixels = 45 + 145 * np.repeat(patterns, 20, axis=0)
    pixels = np.clip(pixels + rng.integers(-20, 21, size=pixels.shape), 0, 255)

    # 이미지 숫자 목록을 JSON 형식의 글자로 바꿔 CSV에 저장합니다. 분석할 때 다시 숫자 목록으로 읽습니다.
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
    # 시드는 무작위 숫자의 출발점입니다. 같은 시드를 쓰면 같은 데이터를 다시 만들 수 있습니다.
    rng = np.random.default_rng(seed)
    catalog = make_catalog(rng)
    # 상품을 가격순으로 정리하고, 고객 유형에 따라 구매할 상품의 가격대를 다르게 정합니다.
    sorted_products = np.argsort(catalog["price"].to_numpy())
    product_pools = {
        "high": sorted_products[-45:],
        "middle": sorted_products[15:75],
        "any": sorted_products,
    }
    start = pd.Timestamp("2025-01-01")
    # 4가지 구매 패턴에 고객을 125명씩 나눕니다. 고객 번호순으로 패턴이 반복되지 않도록 섞습니다.
    profiles = rng.permutation(np.repeat(np.arange(len(PROFILES)), 125))
    rows = []
    order_number = 0

    for customer_number, profile_index in enumerate(profiles, 1):
        order_range, day_range, price_preference = PROFILES[profile_index]
        order_count = rng.integers(*order_range)
        days = np.sort(rng.integers(*day_range, size=order_count))
        if price_preference in ("high", "middle"):
            days[-1] = rng.integers(320, 365)

        # 한 번 주문할 때 서로 다른 상품을 1~3종 고릅니다. 상품 한 종류가 데이터 한 행이 됩니다.
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

    # 주문에서 고른 상품 번호를 이용해 상품명, 가격, 이미지 정보를 붙입니다.
    transactions = pd.DataFrame(
        rows, columns=["customer_id", "order_id", "order_date", "sku", "quantity"]
    ).join(catalog, on="sku").drop(columns="sku")
    # 분석 기간이 1월 1일부터 12월 31일까지가 되도록 맨 처음과 맨 마지막 주문 날짜를 맞춥니다.
    earliest_order = transactions.loc[transactions["order_date"].idxmin(), "order_id"]
    transactions.loc[transactions["order_id"].eq(earliest_order), "order_date"] = start
    latest_order = transactions.loc[transactions["order_date"].idxmax(), "order_id"]
    transactions.loc[transactions["order_id"].eq(latest_order), "order_date"] = (
        start + pd.Timedelta(days=364)
    )

    # 빈칸 처리와 이상치 분석을 연습하도록 상품명 100개를 비우고, 24행에는 많은 구매 수량을 넣습니다.
    missing_rows = rng.choice(transactions.index, size=100, replace=False)
    bulk_rows = rng.choice(transactions.index, size=24, replace=False)
    transactions.loc[missing_rows, "product_name"] = pd.NA
    transactions.loc[bulk_rows, "quantity"] = rng.integers(20, 61, size=len(bulk_rows))
    # 최종 수량에 상품 가격을 곱해 실제로 분석할 구매금액을 계산합니다.
    transactions["amount"] = transactions["price"] * transactions["quantity"]
    return transactions[[
        "customer_id", "order_id", "order_date", "product_id", "product_name", "category",
        "price", "quantity", "amount", "image_array",
    ]]


def main():
    # 실행할 때 --seed로 난수 시드를, --output으로 CSV 저장 위치를 바꿀 수 있습니다.
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
