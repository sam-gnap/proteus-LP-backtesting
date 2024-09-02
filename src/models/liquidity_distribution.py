from typing import Dict, List, Tuple, NamedTuple
from decimal import Decimal
import math
from .pool import Pool


class TickInfo(NamedTuple):
    tick: int
    liquidity: int
    price: Decimal
    amount0: Decimal
    amount1: Decimal


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
            total_amount1 = amount1
            total_amount0 = Decimal("0")
        elif tick == self.current_range_bottom_tick:
            amount0 = liquidity * (sb - self.current_sqrt_price) / (self.current_sqrt_price * sb)
            amount1 = liquidity * (self.current_sqrt_price - sa)
            total_amount0 = amount0
            total_amount1 = amount1
        else:
            amount1 = liquidity * (sb - sa)
            amount0 = amount1 / (sb * sa)
            total_amount0 = amount0
            total_amount1 = Decimal("0")

        price = self.pool.adjust_price(self.pool.tick_to_price(tick))
        if self.invert_price:
            price = Decimal("1") / price

        return TickInfo(tick, liquidity, price, amount0, amount1), total_amount0, total_amount1

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
