import os
from datetime import date, timedelta
from google.cloud import bigquery
import pandas as pd
from decimal import Decimal
from typing import List, Dict, Union
from src.data_processing.queries.big_query import build_query
from src.data_processing.abis.uniswap_v3_abis import SWAP_V3_JS_CODE
from src.utils.helper import sqrtPriceX96_to_price
from src.models.pool import Pool
import glob
from pathlib import Path
import json

V3_SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


class SwapDataCollector:
    def __init__(
        self,
        project_id: str,
        pool_address: str,
        token0: str,
        token1: str,
        decimals0: int,
        decimals1: int,
        tick_spacing: int,
        start_date: date,
        end_date: date,
    ):
        self.project_id = project_id
        self.pool_address = pool_address
        self.pool = Pool(token0, token1, decimals0, decimals1, tick_spacing)
        self.start_date = start_date
        self.end_date = end_date
        self.client = bigquery.Client(project=project_id)

        self.project_root = Path(__file__).resolve().parents[2]

        self.base_dir = self.project_root / "data" / "uniswap" / f"{token0}-{token1}"
        self.swaps_dir = self.base_dir / "swaps"
        self.blocks_dir = self.base_dir / "blocks"

        # Create directories if they don't exist
        self.swaps_dir.mkdir(parents=True, exist_ok=True)
        self.blocks_dir.mkdir(parents=True, exist_ok=True)

    def get_swap_data(
        self, start_date: date, end_date: date, block_interval: int = 100
    ) -> pd.DataFrame:
        query = build_query(SWAP_V3_JS_CODE, start_date, end_date, V3_SWAP_TOPIC, self.pool_address)
        df = self.client.query(query).to_dataframe()
        return df

    def collect_and_store_swap_data(self, time_step: int = 10, block_interval: int = 100):
        current_date = self.start_date
        while current_date <= self.end_date:
            end_of_period = min(current_date + timedelta(days=time_step - 1), self.end_date)
            print(
                f"Processing swaps for period: {current_date.isoformat()} to {end_of_period.isoformat()}"
            )
            swaps_df = self.get_swap_data(current_date, end_of_period, block_interval)
            if not swaps_df.empty:
                filename = (
                    self.swaps_dir
                    / f"swaps_{current_date.isoformat()}_to_{end_of_period.isoformat()}.parquet"
                )
                swaps_df.to_parquet(filename, compression="snappy")
                print(f"Swap data saved to {filename}")
            else:
                print(
                    f"No swap data collected for period: {current_date.isoformat()} to {end_of_period.isoformat()}"
                )
            current_date = end_of_period + timedelta(days=1)

    def load_swap_data(self) -> pd.DataFrame:
        all_files = list(self.swaps_dir.glob("swaps_*.parquet"))
        if not all_files:
            print(f"No swap data files found in {self.swaps_dir}")
            return pd.DataFrame()

        dfs = [pd.read_parquet(file) for file in all_files]
        return pd.concat(dfs, ignore_index=True)

    def process_swap_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price"] = df["sqrtPriceX96"].apply(
            lambda x: sqrtPriceX96_to_price(int(x), self.pool.decimals0, self.pool.decimals1)
        )
        # df['adjusted_price'] = df['price'].apply(self.pool.adjust_price)
        df["invert_price"] = df["price"].apply(self.pool.should_invert_price)
        df["final_price"] = df.apply(
            lambda row: 1 / row["price"] if row["invert_price"] else row["price"], axis=1
        )
        # Convert raw amounts to real amounts
        df["real_amount0"] = df["amount0"].apply(
            lambda x: Decimal(x) / Decimal(10**self.pool.decimals0)
        )
        df["real_amount1"] = df["amount1"].apply(
            lambda x: Decimal(x) / Decimal(10**self.pool.decimals1)
        )

        # Determine which token is being sold and the amounts
        df["token_sold"] = df.apply(
            lambda row: self.pool.token0 if row["real_amount0"] < 0 else self.pool.token1, axis=1
        )
        df["amount_sold"] = df.apply(
            lambda row: (
                abs(row["real_amount0"])
                if row["token_sold"] == self.pool.token0
                else abs(row["real_amount1"])
            ),
            axis=1,
        )
        df["amount_bought"] = df.apply(
            lambda row: (
                abs(row["real_amount1"])
                if row["token_sold"] == self.pool.token0
                else abs(row["real_amount0"])
            ),
            axis=1,
        )

        # Calculate the dollar value
        df["dollar_value"] = df.apply(self.calculate_dollar_value, axis=1)

        return df

    def calculate_dollar_value(self, row):
        if row["token_sold"] == self.pool.token0:
            return abs(row["real_amount0"])
        else:
            return abs(row["real_amount1"]) * row["final_price"]

    def collect_and_sample_blocks_bars(
        self,
        time_step: int = 10,
        block_interval: int = 100,
        method: str = "volume",
        threshold: float = 1000000,
    ):
        self.collect_and_store_swap_data(time_step, block_interval)
        all_swaps = self.load_swap_data()

        if not all_swaps.empty:
            processed_swaps = self.process_swap_data(all_swaps)

            if method == "volume":
                sampled_blocks = self.sample_blocks_volume_bars(processed_swaps, threshold)
            elif method == "dollar":
                sampled_blocks = self.sample_blocks_dollar_bars(processed_swaps, threshold)
            else:
                raise ValueError("Invalid method. Choose 'volume' or 'dollar'.")

            sampled_filename = self.blocks_dir / f"{method}_bar_sampled_blocks.json"
            with open(sampled_filename, "w") as f:
                json.dump(sampled_blocks, f, indent=2)

            print(f"{method.capitalize()} bar sampled block data saved to {sampled_filename}")
        else:
            print("No swap data available for sampling.")

    def sample_blocks_volume_bars(
        self, df: pd.DataFrame, threshold_volume: float
    ) -> List[Dict[str, Union[int, float]]]:
        df = df.sort_values("block_timestamp")
        df["cumulative_volume"] = (df["amount0"].abs() + df["amount1"].abs()).cumsum()

        sampled_blocks = []
        last_bar_end = 0

        for _, row in df.iterrows():
            if row["cumulative_volume"] - last_bar_end >= threshold_volume:
                sampled_blocks.append(
                    {
                        "block_number": int(row["block_number"]),
                        "block_timestamp": int(row["block_timestamp"].timestamp()),
                        "cumulative_volume": float(row["cumulative_volume"]),
                    }
                )
                last_bar_end = row["cumulative_volume"]

        return sampled_blocks

    def sample_blocks_dollar_bars(
        self, df: pd.DataFrame, threshold_dollars: float
    ) -> List[Dict[str, Union[int, float]]]:
        df = df.sort_values("block_timestamp")
        df["cumulative_dollar_volume"] = df["dollar_value"].cumsum()

        sampled_blocks = []
        last_bar_end = 0

        for _, row in df.iterrows():
            if row["cumulative_dollar_volume"] - last_bar_end >= threshold_dollars:
                sampled_blocks.append(
                    {
                        "block_number": int(row["block_number"]),
                        "block_timestamp": int(row["block_timestamp"].timestamp()),
                        "cumulative_dollar_volume": float(row["cumulative_dollar_volume"]),
                    }
                )
                last_bar_end = row["cumulative_dollar_volume"]

        return sampled_blocks

    def load_sampled_blocks_bars(self, method: str = "dollar") -> List[Dict[str, int]]:
        filename = self.blocks_dir / f"{method}_bar_sampled_blocks.json"
        if filename.exists():
            with open(filename, "r") as f:
                return json.load(f)
        else:
            print(f"Sampled block data file not found: {filename}")
            return []
