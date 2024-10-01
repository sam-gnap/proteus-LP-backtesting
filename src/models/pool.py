from decimal import Decimal, getcontext
import numpy as np
from math import sqrt


class Pool:
    def __init__(
        self,
        pool_address: str,
        token0: str,
        token1: str,
        decimals0: int,
        decimals1: int,
        fee_tier: int,
        sqrt_price_x96: int,
    ):
        self.pool_address = pool_address
        self.token0 = token0
        self.token1 = token1
        self.decimals0 = int(decimals0)
        self.decimals1 = int(decimals1)
        self.fee_tier = int(fee_tier)
        self.tick_spacing = self.fee_tier_to_tick_spacing()
        self.TICK_BASE = Decimal("1.0001")
        self.sqrt_price_x96 = sqrt_price_x96
        self.q96 = 2**96
        getcontext().prec = 50

    def to_wei(self, amount, decimals) -> int:
        return int(amount * 10**decimals)

    def price_to_sqrt_price_x96(
        self, price: float, token0_decimals: int, token1_decimals: int
    ) -> int:
        """
        Convert a price (token1/token0) to sqrtPriceX96 format with high precision
        """
        price_adjusted = price * (10 ** (token0_decimals - token1_decimals))
        return int(np.sqrt(price_adjusted) * (2**96))

    def sqrt_price_x96_to_price(
        self, sqrt_price_x96: int, token0_decimals: int, token1_decimals: int
    ) -> float:
        """
        Convert a sqrtPriceX96 to price (token1/token0)
        """
        price = (sqrt_price_x96 / self.q96) ** 2
        return price / (10 ** (token1_decimals - token0_decimals))

    def price_to_sqrtp(self, p):
        return int(np.sqrt(p) * self.q96)

    def tick_to_price(self, tick: int) -> Decimal:
        return self.TICK_BASE**tick

    def price_to_tick(self, price: Decimal) -> int:
        log_price = np.log(np.float64(price))
        log_tick_base = np.log(np.float64(self.TICK_BASE))
        tick = log_price / log_tick_base
        return round(tick)

    def adjust_price(self, price: Decimal) -> Decimal:
        if self.should_invert_price(price):
            return 1 / (price / Decimal(10 ** (self.decimals1 - self.decimals0)))
        return price / Decimal(10 ** (self.decimals1 - self.decimals0))

    def should_invert_price(self, current_price: Decimal) -> bool:
        self.stablecoins = ["USDC", "DAI", "USDT", "TUSD", "LUSD", "BUSD", "GUSD", "UST"]
        if self.token0 in self.stablecoins and self.token1 not in self.stablecoins:
            return True
        return current_price < Decimal("1.0")

    def fee_tier_to_tick_spacing(self):
        return {100: 1, 500: 10, 3000: 60, 10000: 200}.get(self.fee_tier, 60)
