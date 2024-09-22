from decimal import Decimal
import numpy as np


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
        self.sqrt_price, self.price = self.sqrtPriceX96_to_price()

    def tick_to_price(self, tick: int) -> Decimal:
        return self.TICK_BASE**tick

    def price_to_tick(self, price: Decimal) -> int:
        log_price = np.log(np.float64(price))
        log_tick_base = np.log(np.float64(self.TICK_BASE))
        tick = log_price / log_tick_base
        return round(tick)

    def adjust_price(self, price: Decimal) -> Decimal:
        return price / Decimal(10 ** (self.decimals1 - self.decimals0))

    def should_invert_price(self, current_price: Decimal) -> bool:
        self.stablecoins = ["USDC", "DAI", "USDT", "TUSD", "LUSD", "BUSD", "GUSD", "UST"]
        if self.token0 in self.stablecoins and self.token1 not in self.stablecoins:
            return True
        return current_price < Decimal("1.0")

    def fee_tier_to_tick_spacing(self):
        return {100: 1, 500: 10, 3000: 60, 10000: 200}.get(self.fee_tier, 60)

    def sqrtPriceX96_to_price(self):
        sqrt_price = Decimal(self.sqrt_price_x96) / Decimal(2**96)
        price = int(sqrt_price**2)
        price = price * Decimal(10 ** (self.decimals0 - self.decimals1))
        return sqrt_price, price
