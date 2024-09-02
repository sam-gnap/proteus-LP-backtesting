import matplotlib.pyplot as plt
from decimal import Decimal
from typing import List
from ..models.liquidity_distribution import TickInfo
from ..models.pool import Pool


def plot_liquidity_distribution(distribution: List[TickInfo], current_tick: int, pool: Pool):
    current_price = pool.adjust_price(pool.tick_to_price(current_tick))
    min_price = current_price * Decimal("0.5")
    max_price = current_price * Decimal("1.5")

    filtered_distribution = [info for info in distribution if min_price <= info.price <= max_price]

    ticks = [info.tick for info in filtered_distribution]
    liquidities = [info.liquidity for info in filtered_distribution]
    prices = [float(info.price) for info in filtered_distribution]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot liquidity
    ax1.bar(ticks, liquidities, width=pool.tick_spacing, align="edge")
    ax1.set_ylabel("Liquidity")
    ax1.set_title(f"Liquidity Distribution (±50% of current price)")
    ax1.axvline(x=current_tick, color="r", linestyle="--", label="Current Tick")
    ax1.legend()

    # Plot price
    ax2.plot(ticks, prices)
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Price")
    ax2.set_title(f"Price Distribution ({pool.token1}/{pool.token0})")
    ax2.set_yscale("log")
    ax2.axvline(x=current_tick, color="r", linestyle="--", label="Current Tick")
    ax2.axhline(y=float(current_price), color="g", linestyle=":", label="Current Price")
    ax2.axhline(y=float(min_price), color="b", linestyle=":", label="Min Price (-50%)")
    ax2.axhline(y=float(max_price), color="b", linestyle=":", label="Max Price (+50%)")
    ax2.legend()

    plt.tight_layout()
    plt.show()
