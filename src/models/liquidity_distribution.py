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
        self, pool: Pool, start_date: date, end_date: date, threshold_dollar_volume_swaps: int
    ):
        self.pool = pool
        self.start_date = start_date
        self.end_date = end_date
        self.threshold = threshold_dollar_volume_swaps
        self.project_root = Path(__file__).resolve().parents[2]
        self.base_dir = (
            self.project_root
            / "data"
            / "uniswap"
            / f"{self.pool.token0}-{self.pool.token1}"
            / f"fee_{self.pool.fee_tier}"
        )
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
        self.liquidity_changes = self.fetch_liquidity()

    def load_sampled_blocks(self) -> pd.DataFrame:
        collector = SwapDataCollector(
            project_id="uniswap-v3-analytics",
            pool=self.pool,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        # Load existing data
        existing_data = collector.load_sampled_blocks_bars(method="dollar")

        if existing_data:
            # Convert the dictionary of lists into a list of dictionaries
            flattened_data = [
                {**block, "date": date}
                for date, blocks in existing_data.items()
                for block in blocks
            ]
            df = pd.DataFrame(flattened_data)
            df["datetime"] = pd.to_datetime(df["block_timestamp"], unit="s")
            df = df.sort_values("datetime")

            # Check if we need to collect additional data
            if (
                df["datetime"].min().date() > self.start_date
                or df["datetime"].max().date() < self.end_date
            ):
                self.logger.info("Collecting additional data for new date range")
                collector.collect_and_sample_blocks_bars(method="dollar", threshold=self.threshold)
                # Reload the data after collection
                updated_data = collector.load_sampled_blocks_bars(method="dollar")
                flattened_updated_data = [
                    {**block, "date": date}
                    for date, blocks in updated_data.items()
                    for block in blocks
                ]
                df = pd.DataFrame(flattened_updated_data)
                df["datetime"] = pd.to_datetime(df["block_timestamp"], unit="s")
                df = df.sort_values("datetime")
        else:
            self.logger.info("No existing data found. Collecting new data.")
            collector.collect_and_sample_blocks_bars(method="dollar", threshold=self.threshold)
            # Load the newly collected data
            new_data = collector.load_sampled_blocks_bars(method="dollar")
            flattened_new_data = [
                {**block, "date": date} for date, blocks in new_data.items() for block in blocks
            ]
            df = pd.DataFrame(flattened_new_data)
            df["datetime"] = pd.to_datetime(df["block_timestamp"], unit="s")
            df = df.sort_values("datetime")

        return df

    def save_liquidity_distribution(self, block_number: int, distribution: pd.DataFrame):
        filename = self.liquidity_dir / f"liquidity_distribution_{block_number}.parquet"

        for column in distribution.select_dtypes(include=[Decimal]).columns:
            distribution[column] = distribution[column].astype(float).round(10)

        for column in distribution.select_dtypes(include=["int64"]).columns:
            distribution[column] = distribution[column].astype("int32")

        try:
            distribution.to_parquet(filename, compression="snappy")
            self.logger.info(f"Liquidity distribution for block {block_number} saved successfully")
        except Exception as e:
            print(f"Error saving liquidity distribution for block {block_number}: {str(e)}")

    def load_liquidity_distribution(self, block_number: int) -> pd.DataFrame:
        filename = self.liquidity_dir / f"liquidity_distribution_{block_number}.parquet"
        if filename.exists():
            return pd.read_parquet(filename)
        else:
            raise FileNotFoundError(f"Liquidity distribution for block {block_number} not found.")

    def fetch_liquidity(self):
        self.df_liquidity = pd.DataFrame()
        for block in self.sampled_blocks.block_number:
            if not (self.liquidity_dir / f"liquidity_distribution_{block}.parquet").exists():
                df_liquidity_block, _, _ = self.get_liquidity_distribution(block)
            else:
                df_liquidity_block = self.load_liquidity_distribution(block)
            df_liquidity_block = df_liquidity_block.loc[
                (df_liquidity_block["price"] > 1500) & (df_liquidity_block["price"] < 4500)
            ]
            print(f"Loading liquidity for block number {block}")
            self.df_liquidity = pd.concat([self.df_liquidity, df_liquidity_block])
        self.logger.info("Finished fetching and storing liquidity data")

    def get_liquidity_distribution(
        self, block_number: int
    ) -> Tuple[pd.DataFrame, Decimal, Decimal]:
        data = fetch_oku_liquidity(pool_address=self.pool.pool_address, block_number=block_number)
        tick_mapping = organize_tick_data(tick_data=data["ticks"])
        current_tick = int(data["current_pool_tick"])
        self.distribution = LiquidityDistribution(
            self.pool, tick_mapping, current_tick, block_number
        )
        result, _, _ = self.distribution.get_distribution()
        df = pd.DataFrame([r._asdict() for r in result])
        df = df.loc[(df["price"] > 1500) & (df["price"] < 4500)]
        self.save_liquidity_distribution(block_number, df)

        return df, _, _


class TickInfo(NamedTuple):
    block: int
    tick: int
    active_tick: int
    liquidity: int
    price: Decimal
    amount0: Decimal
    amount1: Decimal
    amount0_real: Decimal
    amount1_real: Decimal


class LiquidityDistribution:
    def __init__(self, pool: Pool, tick_mapping: Dict[int, int], current_tick: int, block: int):
        self.pool = pool
        self.block = block
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
            TickInfo(
                self.block,
                self.current_tick,
                tick,
                liquidity,
                price,
                amount0,
                amount1,
                adjusted_amount0,
                adjusted_amount1,
            ),
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


class UniswapV3LiquidityCalculator:
    Q96 = 2**96

    def __init__(self, pool):
        self.pool = pool

    @staticmethod
    def get_tick_at_sqrt_ratio(sqrt_ratio):
        return math.floor(math.log((sqrt_ratio / UniswapV3LiquidityCalculator.Q96) ** 2, 1.0001))

    @staticmethod
    def get_sqrt_ratio_at_tick(tick):
        return int((1.0001 ** (tick / 2)) * UniswapV3LiquidityCalculator.Q96)

    @staticmethod
    def get_liquidity_for_amounts(
        sqrt_price_x96, sqrt_price_a_x96, sqrt_price_b_x96, amount0, amount1
    ):
        if sqrt_price_a_x96 > sqrt_price_b_x96:
            sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96

        liquidity = 0
        if sqrt_price_x96 <= sqrt_price_a_x96:
            liquidity = UniswapV3LiquidityCalculator.get_liquidity_for_amount0(
                sqrt_price_a_x96, sqrt_price_b_x96, amount0
            )
        elif sqrt_price_x96 < sqrt_price_b_x96:
            liquidity0 = UniswapV3LiquidityCalculator.get_liquidity_for_amount0(
                sqrt_price_x96, sqrt_price_b_x96, amount0
            )
            liquidity1 = UniswapV3LiquidityCalculator.get_liquidity_for_amount1(
                sqrt_price_a_x96, sqrt_price_x96, amount1
            )
            liquidity = min(liquidity0, liquidity1)
        else:
            liquidity = UniswapV3LiquidityCalculator.get_liquidity_for_amount1(
                sqrt_price_a_x96, sqrt_price_b_x96, amount1
            )

        return liquidity

    @staticmethod
    def get_liquidity_for_amount0(sqrt_price_a_x96, sqrt_price_b_x96, amount0):
        intermediate = sqrt_price_a_x96 * sqrt_price_b_x96 // UniswapV3LiquidityCalculator.Q96
        return (amount0 * intermediate) // (sqrt_price_b_x96 - sqrt_price_a_x96)

    @staticmethod
    def get_liquidity_for_amount1(sqrt_price_a_x96, sqrt_price_b_x96, amount1):
        return (amount1 * UniswapV3LiquidityCalculator.Q96) // (sqrt_price_b_x96 - sqrt_price_a_x96)

    def calculate_liquidity_with_tick_spacing(
        self, amount0, amount1, current_price, lower_price, upper_price
    ):
        # Convert prices to ticks
        current_tick = self.pool.price_to_tick(current_price)
        lower_tick = self.pool.price_to_tick(lower_price)
        upper_tick = self.pool.price_to_tick(upper_price)

        # Adjust ticks to be multiples of tick_spacing
        lower_tick = math.ceil(lower_tick / self.pool.tick_spacing) * self.pool.tick_spacing
        upper_tick = math.floor(upper_tick / self.pool.tick_spacing) * self.pool.tick_spacing

        # Convert adjusted ticks back to sqrt prices
        sqrt_price_x96 = self.get_sqrt_ratio_at_tick(current_tick)
        sqrt_price_a_x96 = self.get_sqrt_ratio_at_tick(lower_tick)
        sqrt_price_b_x96 = self.get_sqrt_ratio_at_tick(upper_tick)

        # Calculate liquidity
        liquidity = self.get_liquidity_for_amounts(
            sqrt_price_x96, sqrt_price_a_x96, sqrt_price_b_x96, amount0, amount1
        )

        return liquidity


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
