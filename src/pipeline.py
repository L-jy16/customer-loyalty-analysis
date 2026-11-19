"""Reusable analysis for fictional e-commerce order lines."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class DataAnalyzer:
    """Load order lines and keep the source frame separate from working data."""

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.raw = None
        self.df = None

    def load_data(self):
        """Read the CSV, parse order dates, and reset the working data."""
        raw = pd.read_csv(self.data_path)
        raw["order_date"] = pd.to_datetime(raw["order_date"], errors="raise")
        # 불러온 원본은 self.raw에 보관하고, 전처리는 복사한 self.df에서 진행합니다.
        self.raw = raw
        self.df = raw.copy()
        # 여기서 돌려주는 표는 전처리 중에 바뀝니다. 처리 전 모습은 .copy(deep=True)로 따로 복사해 둡니다.
        return self.df

    def explore(self, rows=5):
        """Print schema and return a sample and an all-column summary."""
        self.df.info()
        return self.df.head(rows), self.df.describe(include="all")

    def missing_counts(self):
        """Return the number of missing values for each column."""
        return self.df.isna().sum()

    def handle_missing(self, column="product_name", group_col="product_id",
                       strategy="mode"):
        """Expose the evaluator's method name using the same imputation logic."""
        # 두 메서드 이름 중 어느 것을 호출해도 같은 빈칸 채우기 함수를 실행합니다.
        return self.handle_missing_values(column, group_col, strategy)

    def handle_missing_values(self, column="product_name", group_col="product_id",
                              strategy="mode"):
        """Fill missing values from their group, falling back to a global statistic."""
        original = self.df[column]
        if not original.isna().any():
            return original

        # 상품명 빈칸은 가장 자주 나온 이름으로 채웁니다. 숫자 열은 평균이나 가운데 값으로 채울 수도 있습니다.
        if strategy == "mode":
            def first_mode(values):
                modes = values.mode()
                return modes.iloc[0] if not modes.empty else np.nan

            aggregate = first_mode
        elif strategy in ("mean", "median"):
            aggregate = strategy
        else:
            raise ValueError("strategy must be mode, mean, or median")

        # 같은 상품끼리 묶어 대표값을 구하고, 각 행에 해당 상품의 대표값을 연결합니다.
        grouped = self.df.groupby(group_col)[column].transform(aggregate)
        # 어떤 상품의 값이 전부 비어 있으면 사용할 수 있도록 전체 데이터의 대표값도 구합니다.
        fallback = original.agg(aggregate)
        if pd.isna(fallback):
            raise ValueError(f"Cannot impute {column}: all values are missing")
        # 같은 상품의 대표값으로 빈칸을 채운 뒤, 그래도 남은 빈칸은 전체 대표값으로 채웁니다.
        self.df[column] = original.fillna(grouped).fillna(fallback)
        return self.df[column]

    def iqr_bounds(self, column, threshold=1.5):
        """Return the lower and upper IQR fences."""
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        q1, q3 = self.df[column].quantile([0.25, 0.75])
        if pd.isna(q1) or pd.isna(q3):
            raise ValueError(f"Cannot compute IQR for {column}: no values")
        # 작은 값부터 정렬했을 때 25%와 75% 위치의 차이가 IQR입니다. 이 범위로 이상치를 찾을 아래·위 경계를 만듭니다.
        width = q3 - q1
        return q1 - threshold * width, q3 + threshold * width

    def detect_outliers(self, column="quantity", threshold=1.5):
        """Return rows outside the IQR fences."""
        lower, upper = self.iqr_bounds(column, threshold)
        values = self.df[column]
        # 아래 경계보다 작거나 위 경계보다 큰 행을 골라 이상치 목록으로 보여줍니다.
        return self.df.loc[values.lt(lower) | values.gt(upper)].copy()

    def add_clipped_feature(self, column="quantity", threshold=1.5):
        """Store IQR-clipped values for EDA alongside observed values."""
        lower, upper = self.iqr_bounds(column, threshold)
        # 보정한 수량은 새 열에 넣습니다. 원래 수량과 비교할 때 사용합니다.
        feature = f"{column}_clipped"
        self.df[feature] = self.df[column].clip(lower, upper)
        return self.df[feature]

    def add_text_features(self, column="product_name"):
        """Count whitespace-delimited words in each name."""
        self.df["name_word_count"] = self.df[column].str.split().str.len()
        return self.df["name_word_count"]

    def add_image_features(self, column="image_array", image_shape=(8, 8)):
        """Decode fixed-size JSON arrays and vectorize pixel statistics."""
        encoded = self.df[column]
        # 같은 이미지가 여러 주문에 나와도 한 번만 읽고 계산해 중복 작업을 줄입니다.
        unique_images = pd.Index(encoded.unique())
        try:
            decoded = unique_images.map(json.loads).tolist()
            pixels = np.asarray(decoded)
        except (TypeError, ValueError) as error:
            raise ValueError("Images must be valid, consistently shaped JSON arrays") from error
        # 모든 이미지의 크기가 같은지, 픽셀이 계산 가능한 숫자이며 밝기 범위 0~255에 들어가는지 확인합니다.
        if pixels.shape != (len(unique_images), *image_shape):
            raise ValueError(f"Each image must have shape {image_shape}")
        if not np.issubdtype(pixels.dtype, np.number):
            raise ValueError("Image pixels must be numeric")
        if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 255)):
            raise ValueError("Image pixels must lie between 0 and 255")
        # 여러 이미지가 쌓인 배열에서 이미지별 픽셀을 한꺼번에 계산합니다. 이미지마다 반복문을 돌릴 필요가 없습니다.
        axes = tuple(range(1, pixels.ndim))
        # 한 번 계산한 이미지 결과를 그 이미지가 들어 있는 각 주문 행에 붙입니다.
        indices = unique_images.get_indexer(encoded)
        self.df["image_mean"] = pixels.mean(axis=axes)[indices]
        # 표준편차는 픽셀 밝기가 얼마나 다른지 보여줍니다. 여기서는 이미지의 모든 픽셀을 대상으로 계산합니다.
        self.df["image_std"] = pixels.std(axis=axes)[indices]
        return self.df[["image_mean", "image_std"]]

    def save_processed(self, output_path):
        """Write analysis features to a separate CSV."""
        destination = Path(output_path)
        if destination.resolve() == self.data_path.resolve():
            raise ValueError("Processed data must use a different path from the raw data")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(destination, index=False)

    def descriptive_statistics(self, columns=("price", "quantity", "amount")):
        """Return count, mean, sample standard deviation, quartiles and range."""
        # 평균, 가운데 값, 표준편차 등을 표로 정리합니다. 50%는 중앙값이고 std는 표본 표준편차입니다.
        return self.df[list(columns)].describe().T.rename(columns={"50%": "median"})

    def correlations(self, columns=("price", "quantity", "amount")):
        """Compute Pearson correlations for the selected columns."""
        return self.df[list(columns)].corr(method="pearson")

    def monthly_sales(self, date_col="order_date", amount_col="amount"):
        """Aggregate observed transaction amounts by calendar month."""
        # 주문 날짜에서 연도와 월을 뽑아 같은 달의 구매금액을 모두 더합니다.
        months = self.df[date_col].dt.to_period("M")
        return self.df.groupby(months)[amount_col].sum().sort_index()

    def calculate_rfm(self, customer_col="customer_id", date_col="order_date",
                      amount_col="amount", order_col="order_id", reference_date=None,
                      churn_days=90, new_days=30, vip_min_score=4):
        """Score customer behavior and assign one segment in priority order."""
        required = [customer_col, date_col, amount_col, order_col]
        if self.df.empty or self.df[required].isna().any().any():
            raise ValueError("RFM requires nonempty transactions with complete purchase fields")

        # 시각은 제외하고 날짜로 계산합니다. 기준일을 따로 주지 않으면 가장 마지막 주문일을 사용합니다.
        dates = pd.to_datetime(self.df[date_col]).dt.normalize()
        reference = dates.max() if reference_date is None else pd.Timestamp(reference_date).normalize()
        if pd.isna(reference) or reference < dates.max():
            raise ValueError("RFM reference date must be on or after the last transaction date")
        if not 0 <= new_days < churn_days or not 1 <= vip_min_score <= 5:
            raise ValueError("Require 0 <= new_days < churn_days and VIP score between 1 and 5")

        # 고객별로 묶어 첫·마지막 구매일, 주문 횟수, 총금액을 구합니다. 한 주문에 상품이 여러 개여도 주문은 한 번으로 셉니다.
        customers = self.df.assign(**{date_col: dates}).groupby(customer_col).agg(
            first_purchase=(date_col, "min"),
            last_purchase=(date_col, "max"),
            frequency=(order_col, "nunique"),
            monetary=(amount_col, "sum"),
        )
        customers["recency"] = (reference - customers["last_purchase"]).dt.days
        # 데이터에서 확인되는 첫 구매 이후 며칠이 지났는지 계산합니다. 실제 가입일은 이 데이터로 알 수 없습니다.
        observed_age = (reference - customers["first_purchase"]).dt.days

        # 최근에 샀을수록 R점수, 자주 샀을수록 F점수, 많이 썼을수록 M점수가 높아지게 합니다.
        for metric, ascending in (("recency", False), ("frequency", True), ("monetary", True)):
            # 고객의 상대적인 순위를 1~5점으로 바꿉니다. 값이 같은 고객은 같은 점수를 받습니다.
            percentiles = customers[metric].rank(method="average", ascending=ascending, pct=True)
            customers[f"{metric[0]}_score"] = np.ceil(percentiles * 5).astype(int)

        # 오랫동안 구매하지 않은 고객, 최근 첫 구매 고객, VIP 조건을 차례로 확인합니다.
        churned = customers["recency"].ge(churn_days)
        new = observed_age.le(new_days) & customers["frequency"].le(2)
        vip = customers["frequency"].ge(3) & customers[["r_score", "f_score", "m_score"]].ge(
            vip_min_score
        ).all(axis=1)
        # 여러 조건에 맞아도 먼저 맞은 그룹 하나만 선택합니다. 나머지 고객은 Loyal로 정합니다.
        customers["segment"] = np.select(
            [churned, new, vip], ["Churned", "New", "VIP"], default="Loyal"
        )
        return customers

    @staticmethod
    def summarize_segments(customers):
        """Report customer counts and mean RFM values by segment."""
        # 그룹마다 고객이 몇 명인지, 평균 구매금액과 주문 횟수는 얼마인지 정리합니다.
        return customers.groupby("segment").agg(
            customers=("segment", "size"),
            average_monetary=("monetary", "mean"),
            average_frequency=("frequency", "mean"),
            average_recency=("recency", "mean"),
            total_monetary=("monetary", "sum"),
        ).reindex(["VIP", "Loyal", "New", "Churned"]).assign(
            customers=lambda frame: frame["customers"].fillna(0).astype(int),
            total_monetary=lambda frame: frame["total_monetary"].fillna(0),
        )
