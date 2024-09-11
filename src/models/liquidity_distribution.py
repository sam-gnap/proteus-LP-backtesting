from typing import Dict, List, Tuple, NamedTuple
import math
import pandas as pd
from datetime import date
from decimal import Decimal
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from src.models.pool import Pool
from src.data_processing.fetch_data import fetch_oku_liquidity
from src.utils.helper import organize_tick_data
from pathlib import Path
from src.data_processing.swap_data_collector import SwapDataCollector
import logging


class LiquidityAnalyzer:
    def __init__(
        self,
        pool_address: str,
        token0: str,
        token1: str,
        decimals0: int,
        decimals1: int,
        sqrt_price_x96: int,
        tick_spacing: int,
        start_date: date,
        end_date: date,
    ):

        self.pool_address = pool_address
        self.pool = Pool(token0, token1, decimals0, decimals1, tick_spacing)
        self.start_date = start_date
        self.end_date = end_date

        self.project_root = Path(__file__).resolve().parents[2]
        self.base_dir = self.project_root / "data" / "uniswap" / f"{token0}-{token1}"
        self.liquidity_dir = self.base_dir / "liquidity"
        self.blocks_dir = self.base_dir / "blocks"
        # Create directories if they don't exist
        self.liquidity_dir.mkdir(parents=True, exist_ok=True)
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=f"{self.base_dir}/liquidity_analysis.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.sampled_blocks = self.load_sampled_blocks()

    def load_sampled_blocks(self) -> pd.DataFrame:
        collector = SwapDataCollector(
            project_id="uniswap-v3-analytics",
            pool_address=self.pool_address,
            token0=self.pool.token0,
            token1=self.pool.token1,
            decimals0=self.pool.decimals0,
            decimals1=self.pool.decimals1,
            tick_spacing=self.pool.tick_spacing,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        # Check if we have existing data
        existing_data = collector.load_sampled_blocks_bars(method="dollar")

        if existing_data:
            df = pd.DataFrame(existing_data)
            df["datetime"] = pd.to_datetime(df["block_timestamp"], unit="s")
            df = df.sort_values("datetime")

            # Check if we need to collect additional data
            if (
                df["datetime"].min().date() > self.start_date
                or df["datetime"].max().date() < self.end_date
            ):
                self.logger.info("Collecting additional data for new date range")
                collector.collect_and_sample_blocks_bars(method="dollar", threshold=1000000)

                # Reload the data after collection
                updated_data = collector.load_sampled_blocks_bars(method="dollar")
                df = pd.DataFrame(updated_data)
                df["datetime"] = pd.to_datetime(df["block_timestamp"], unit="s")
                df = df.sort_values("datetime")
        else:
            self.logger.info("No existing data found. Collecting new data.")
            collector.collect_and_sample_blocks_bars(method="dollar", threshold=1000000)

            # Load the newly collected data
            new_data = collector.load_sampled_blocks_bars(method="dollar")
            df = pd.DataFrame(new_data)
            df["datetime"] = pd.to_datetime(df["block_timestamp"], unit="s")
            df = df.sort_values("datetime")

        return df

    def analyze_liquidity(self, block_number: int) -> Tuple[pd.DataFrame, Decimal, Decimal]:
        data = fetch_oku_liquidity(pool_address=self.pool_address, block_number=block_number)
        tick_mapping = organize_tick_data(tick_data=data["ticks"])
        current_tick = int(data["current_pool_tick"])
        self.distribution = LiquidityDistribution(self.pool, tick_mapping, current_tick)
        result, total_amount0, total_amount1 = self.distribution.get_distribution()

        # Convert result to DataFrame and save
        df = pd.DataFrame([r._asdict() for r in result])
        self.save_liquidity_distribution(block_number, df)

        return df, total_amount0, total_amount1

    def save_liquidity_distribution(self, block_number: int, distribution: pd.DataFrame):
        filename = self.liquidity_dir / f"liquidity_distribution_{block_number}.parquet"

        # Convert Decimal columns to float with limited precision
        for column in distribution.select_dtypes(include=[Decimal]).columns:
            distribution[column] = distribution[column].astype(float).round(10)

        # Convert int64 to int32 if necessary
        for column in distribution.select_dtypes(include=["int64"]).columns:
            distribution[column] = distribution[column].astype("int32")

        try:
            distribution.to_parquet(filename, compression="snappy")
        except Exception as e:
            print(f"Error saving liquidity distribution for block {block_number}: {str(e)}")

    def load_liquidity_distribution(self, block_number: int) -> pd.DataFrame:
        filename = self.liquidity_dir / f"liquidity_distribution_{block_number}.parquet"
        if filename.exists():
            return pd.read_parquet(filename)
        else:
            raise FileNotFoundError(f"Liquidity distribution for block {block_number} not found.")

    def fetch_and_store_liquidity(self, start_date: date, end_date: date):
        blocks = self.select_blocks_by_date_range(start_date, end_date)
        for block in blocks:
            if not (self.liquidity_dir / f"liquidity_distribution_{block}.parquet").exists():
                self.analyze_liquidity(block)
            else:
                print(f"Liquidity distribution for block {block} already exists. Skipping.")

        self.logger.info("Finished fetching and storing liquidity data")

    def analyze_liquidity_changes(self, start_date: date, end_date: date) -> pd.DataFrame:
        blocks = self.select_blocks_by_date_range(start_date, end_date)

        # First, ensure all required liquidity data is fetched and stored
        self.fetch_and_store_liquidity(start_date, end_date)

        liquidity_data = []
        for block in blocks:
            df, total_amount0, total_amount1 = self.analyze_liquidity(block)

        return pd.DataFrame(liquidity_data)

    def plot_liquidity_over_time(self, start: date, end: date, num_samples: int = 5):
        blocks = self.select_blocks_by_date_range(start, end)
        sampled_blocks = blocks[:num_samples]  # Take first num_samples blocks for simplicity

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

        for block in sampled_blocks:
            result, _, _ = self.analyze_liquidity(block)
            timestamp = self.sampled_blocks.loc[
                self.sampled_blocks["block_number"] == block, "datetime"
            ].iloc[0]

            ticks = [info.tick for info in result]
            liquidities = [info.liquidity for info in result]

            ax1.plot(ticks, liquidities, label=f"Block {block} ({timestamp:%Y-%m-%d %H:%M})")
            ax2.plot(
                [info.price for info in result],
                liquidities,
                label=f"Block {block} ({timestamp:%Y-%m-%d %H:%M})",
            )

        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Liquidity")
        ax1.set_title("Liquidity Distribution over Ticks")
        ax1.legend()

        ax2.set_xlabel("Price")
        ax2.set_ylabel("Liquidity")
        ax2.set_title(f"Liquidity Distribution over Price ({self.pool.token1}/{self.pool.token0})")
        ax2.set_xscale("log")
        ax2.legend()

        plt.tight_layout()
        plt.show()


class TickInfo(NamedTuple):
    tick: int
    liquidity: int
    price: Decimal
    amount0: Decimal
    amount1: Decimal
    amount0_real: Decimal
    amount1_real: Decimal


class LiquidityDistribution:
    def __init__(self, pool: Pool, tick_mapping: Dict[int, int], current_tick: int):
        self.pool = pool
        self.tick_mapping = tick_mapping
        self.current_tick = current_tick
        self.current_sqrt_price = self.pool.tick_to_price(current_tick // 2)
        self.current_price = self.pool.adjust_price(self.current_sqrt_price**2)
        self.invert_price = self.pool.should_invert_price(self.current_price)
        self.current_range_bottom_tick = (
            math.floor(current_tick / pool.tick_spacing) * pool.tick_spacing
        )

    def calculate_amounts(
        self, liquidity: int, sa: Decimal, sb: Decimal
    ) -> Tuple[Decimal, Decimal]:
        amount1 = Decimal(liquidity) * (sb - sa)
        amount0 = amount1 / (sb * sa)
        return amount0, amount1

    def process_tick(self, tick: int, liquidity: int) -> Tuple[TickInfo, Decimal, Decimal]:
        bottom_tick = tick
        top_tick = bottom_tick + self.pool.tick_spacing
        sa = self.pool.tick_to_price(bottom_tick // 2)
        sb = self.pool.tick_to_price(top_tick // 2)

        if tick < self.current_range_bottom_tick:
            amount1 = liquidity * (sb - sa)
            amount0 = amount1 / (sb * sa)
            adjusted_amount0 = amount0 / 10**self.pool.decimals0
            adjusted_amount1 = amount1 / 10**self.pool.decimals1
            total_amount1 = amount1
            total_amount0 = Decimal("0")
        elif tick == self.current_range_bottom_tick:
            amount0 = liquidity * (sb - self.current_sqrt_price) / (self.current_sqrt_price * sb)
            amount1 = liquidity * (self.current_sqrt_price - sa)
            adjusted_amount0 = amount0 / 10**self.pool.decimals0
            adjusted_amount1 = amount1 / 10**self.pool.decimals1
            total_amount0 = amount0
            total_amount1 = amount1
        else:
            amount1 = liquidity * (sb - sa)
            amount0 = amount1 / (sb * sa)
            adjusted_amount0 = amount0 / 10**self.pool.decimals0
            adjusted_amount1 = amount1 / 10**self.pool.decimals1
            total_amount0 = amount0
            total_amount1 = Decimal("0")

        price = self.pool.adjust_price(self.pool.tick_to_price(tick))
        if self.invert_price:
            price = Decimal("1") / price

        return (
            TickInfo(tick, liquidity, price, amount0, amount1, adjusted_amount0, adjusted_amount1),
            total_amount0,
            total_amount1,
        )

    def get_distribution(self) -> Tuple[List[TickInfo], Decimal, Decimal]:
        distribution = []
        liquidity = 0
        total_amount0 = Decimal("0")
        total_amount1 = Decimal("0")
        min_tick = min(self.tick_mapping.keys())
        max_tick = max(self.tick_mapping.keys())

        for tick in range(min_tick, max_tick + self.pool.tick_spacing, self.pool.tick_spacing):
            liquidity_delta = self.tick_mapping.get(tick, 0)
            liquidity += liquidity_delta
            tick_info, amount0, amount1 = self.process_tick(tick, liquidity)
            distribution.append(tick_info)
            total_amount0 += amount0
            total_amount1 += amount1

        return distribution, total_amount0, total_amount1


# if __name__ == "main":
#     POOL_ADDRESS = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
#     START_DATE = date(2024, 8, 1)
#     END_DATE = date(2024, 8, 31)

#     analyzer = LiquidityAnalyzer(
#         pool_address=POOL_ADDRESS,
#         token0="WETH",
#         token1="USDC",
#         decimals0=18,
#         decimals1=6,
#         tick_spacing=60,
#         start_date=START_DATE,
#         end_date=END_DATE,
#     )
#     data = fetch_oku_liquidity(pool_address=POOL_ADDRESS, block_number=20530985)
#     current_tick = int(data["current_pool_tick"])
#     pool = Pool(token0="WETH", token1="USDC", decimals0=18, decimals1=6, tick_spacing=60)
#     liq = LiquidityDistribution(pool, data, current_tick)
