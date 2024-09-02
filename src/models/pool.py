from decimal import Decimal


class Pool:
    def __init__(self, token0: str, token1: str, decimals0: int, decimals1: int, tick_spacing: int):
        self.token0 = token0
        self.token1 = token1
        self.decimals0 = decimals0
        self.decimals1 = decimals1
        self.tick_spacing = tick_spacing
        self.TICK_BASE = Decimal("1.0001")

    def tick_to_price(self, tick: int) -> Decimal:
        return self.TICK_BASE**tick

    def adjust_price(self, price: Decimal) -> Decimal:
        return price / Decimal(10 ** (self.decimals1 - self.decimals0))

    def should_invert_price(self, current_price: Decimal) -> bool:
        stablecoins = ["USDC", "DAI", "USDT", "TUSD", "LUSD", "BUSD", "GUSD", "UST"]
        if self.token0 in stablecoins and self.token1 not in stablecoins:
            return True
        return current_price < Decimal("1.0")
