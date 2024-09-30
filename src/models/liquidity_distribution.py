from typing import Dict, List, Tuple, NamedTuple
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
import numpy as np
from decimal import Decimal, getcontext
import math


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


# Set a high precision to handle the large numbers
getcontext().prec = 80


class LiquidityPeriphery:
    Q96 = 2**96
    MIN_TICK = -887272
    MAX_TICK = -MIN_TICK
    MIN_SQRT_RATIO = 4295128739
    MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

    def __init__(self, pool):
        self.pool = pool

    def mul_div(self, a, b, denominator):
        return (a * b) // denominator

    def get_sqrt_ratio_at_tick(self, tick):
        abs_tick = abs(tick)
        if abs_tick > LiquidityPeriphery.MAX_TICK:
            raise ValueError("Tick must be between MIN_TICK and MAX_TICK")

        ratio = (
            0xFFFCB933BD6FAD37AA2D162D1A594001
            if abs_tick & 0x1 != 0
            else 0x100000000000000000000000000000000
        )

        if abs_tick & 0x2 != 0:
            ratio = (ratio * 0xFFF97272373D413259A46990580E213A) >> 128
        if abs_tick & 0x4 != 0:
            ratio = (ratio * 0xFFF2E50F5F656932EF12357CF3C7FDCC) >> 128
        if abs_tick & 0x8 != 0:
            ratio = (ratio * 0xFFE5CACA7E10E4E61C3624EAA0941CD0) >> 128
        if abs_tick & 0x10 != 0:
            ratio = (ratio * 0xFFCB9843D60F6159C9DB58835C926644) >> 128
        if abs_tick & 0x20 != 0:
            ratio = (ratio * 0xFF973B41FA98C081472E6896DFB254C0) >> 128
        if abs_tick & 0x40 != 0:
            ratio = (ratio * 0xFF2EA16466C96A3843EC78B326B52861) >> 128
        if abs_tick & 0x80 != 0:
            ratio = (ratio * 0xFE5DEE046A99A2A811C461F1969C3053) >> 128
        if abs_tick & 0x100 != 0:
            ratio = (ratio * 0xFCBE86C7900A88AEDCFFC83B479AA3A4) >> 128
        if abs_tick & 0x200 != 0:
            ratio = (ratio * 0xF987A7253AC413176F2B074CF7815E54) >> 128
        if abs_tick & 0x400 != 0:
            ratio = (ratio * 0xF3392B0822B70005940C7A398E4B70F3) >> 128
        if abs_tick & 0x800 != 0:
            ratio = (ratio * 0xE7159475A2C29B7443B29C7FA6E889D9) >> 128
        if abs_tick & 0x1000 != 0:
            ratio = (ratio * 0xD097F3BDFD2022B8845AD8F792AA5825) >> 128
        if abs_tick & 0x2000 != 0:
            ratio = (ratio * 0xA9F746462D870FDF8A65DC1F90E061E5) >> 128
        if abs_tick & 0x4000 != 0:
            ratio = (ratio * 0x70D869A156D2A1B890BB3DF62BAF32F7) >> 128
        if abs_tick & 0x8000 != 0:
            ratio = (ratio * 0x31BE135F97D08FD981231505542FCFA6) >> 128
        if abs_tick & 0x10000 != 0:
            ratio = (ratio * 0x9AA508B5B7A84E1C677DE54F3E99BC9) >> 128
        if abs_tick & 0x20000 != 0:
            ratio = (ratio * 0x5D6AF8DEDB81196699C329225EE604) >> 128
        if abs_tick & 0x40000 != 0:
            ratio = (ratio * 0x2216E584F5FA1EA926041BEDFE98) >> 128
        if abs_tick & 0x80000 != 0:
            ratio = (ratio * 0x48A170391F7DC42444E8FA2) >> 128

        if tick > 0:
            ratio = (2**256 - 1) // ratio

        return int((ratio >> 32) + (0 if ratio % (1 << 32) == 0 else 1))

    def price_to_tick(self, price: float, token0_decimals: int, token1_decimals: int) -> int:
        decimals_adjustment = 10 ** (token0_decimals - token1_decimals)
        price_adjusted = price * decimals_adjustment
        sqrt_price_x96 = int((math.sqrt(price_adjusted) * (2**96)))
        return self.get_tick_at_sqrt_ratio(sqrt_price_x96)

    def get_liquidity_for_amounts(
        self, sqrt_ratio_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount0, amount1
    ):
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

        if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
            liquidity = self.get_liquidity_for_amount0(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount0)
        elif sqrt_ratio_x96 < sqrt_ratio_b_x96:
            liquidity0 = self.get_liquidity_for_amount0(sqrt_ratio_x96, sqrt_ratio_b_x96, amount0)
            liquidity1 = self.get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_x96, amount1)
            liquidity = min(liquidity0, liquidity1)
        else:
            liquidity = self.get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount1)

        return sqrt_ratio_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount0, amount1, liquidity

    def get_liquidity_for_amount0(self, sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount0):
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

        intermediate = self.mul_div(sqrt_ratio_a_x96, sqrt_ratio_b_x96, self.Q96)
        return self.mul_div(amount0, intermediate, sqrt_ratio_b_x96 - sqrt_ratio_a_x96)

    def get_liquidity_for_amount1(self, sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount1):
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

        return self.mul_div(amount1, self.Q96, sqrt_ratio_b_x96 - sqrt_ratio_a_x96)

    def calculate_liquidity(self, amount0, amount1, lower_tick, upper_tick, current_tick):
        sqrt_ratio_x96 = self.get_sqrt_ratio_at_tick(current_tick)
        sqrt_ratio_a_x96 = self.get_sqrt_ratio_at_tick(lower_tick)
        sqrt_ratio_b_x96 = self.get_sqrt_ratio_at_tick(upper_tick)
        amount0 = amount0 * (10**self.pool.decimals0)
        amount1 = amount1 * (10**self.pool.decimals1)

        return self.get_liquidity_for_amounts(
            sqrt_ratio_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount0, amount1
        )

    def get_amounts_for_liquidity(self, current_tick, lower_tick, upper_tick, liquidity):
        sqrt_ratio_x96 = self.get_sqrt_ratio_at_tick(current_tick)
        sqrt_ratio_a_x96 = self.get_sqrt_ratio_at_tick(lower_tick)
        sqrt_ratio_b_x96 = self.get_sqrt_ratio_at_tick(upper_tick)
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

        if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
            amount0 = self.get_amount0_for_liquidity(sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity)
            amount1 = 0
        elif sqrt_ratio_x96 < sqrt_ratio_b_x96:
            amount0 = self.get_amount0_for_liquidity(sqrt_ratio_x96, sqrt_ratio_b_x96, liquidity)
            amount1 = self.get_amount1_for_liquidity(sqrt_ratio_a_x96, sqrt_ratio_x96, liquidity)
        else:
            amount0 = 0
            amount1 = self.get_amount1_for_liquidity(sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity)
        amount0 = amount0 / (10**self.pool.decimals0)
        amount1 = amount1 / (10**self.pool.decimals1)
        return amount0, amount1

    def get_amount0_for_liquidity(self, sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity):
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

        return (
            self.mul_div(liquidity << 96, sqrt_ratio_b_x96 - sqrt_ratio_a_x96, sqrt_ratio_b_x96)
            // sqrt_ratio_a_x96
        )

    def get_amount1_for_liquidity(self, sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity):
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

        return self.mul_div(liquidity, sqrt_ratio_b_x96 - sqrt_ratio_a_x96, self.Q96)

    def print_debug_info(self, amount0, amount1, current_tick, lower_tick, upper_tick):
        sqrt_ratio_x96 = self.get_sqrt_ratio_at_tick(current_tick)
        sqrt_ratio_a_x96 = self.get_sqrt_ratio_at_tick(lower_tick)
        sqrt_ratio_b_x96 = self.get_sqrt_ratio_at_tick(upper_tick)

        print(f"sqrt_ratio_x96: {sqrt_ratio_x96}")
        print(f"sqrt_ratio_a_x96: {sqrt_ratio_a_x96}")
        print(f"sqrt_ratio_b_x96: {sqrt_ratio_b_x96}")
        print(f"amount0 (scaled): {amount0 * Decimal(10**self.pool.decimals0)}")
        print(f"amount1 (scaled): {amount1 * Decimal(10**self.pool.decimals1)}")
