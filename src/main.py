from decimal import Decimal
from src.data_processing.fetch_data import fetch_oku_liquidity, fetch_pool_data
from src.models.pool import Pool
from src.models.liquidity_distribution import LiquidityDistribution
from src.utils.helper import organize_tick_data, sqrtPriceX96_to_price
from src.visualization.plot_liquidity import plot_liquidity_distribution
from config.api_config import API_KEY


def main():
    POOL_ADDRESS = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    BLOCK_NUMBER = 20656195

    # Fetch pool data
    sqrt_price, tick_spacing, token0, token1, decimals0, decimals1, fee_tier = fetch_pool_data(
        POOL_ADDRESS, API_KEY
    )

    # Fetch liquidity data
    data = fetch_oku_liquidity(pool_address=POOL_ADDRESS, block_number=BLOCK_NUMBER)

    # Process data
    current_price = sqrtPriceX96_to_price(
        int(data["sqrt_price_x96"]), data["token0_decimals"], data["token1_decimals"]
    )
    current_price_inverted = 1 / current_price
    tick_mapping = organize_tick_data(tick_data=data["ticks"])

    current_tick = int(data["current_pool_tick"])

    # Create Pool and LiquidityDistribution instances
    pool = Pool(token0="WETH", token1="USDC", decimals0=18, decimals1=6, tick_spacing=60)
    distribution = LiquidityDistribution(pool, tick_mapping, current_tick)

    # Get distribution data
    result, total_amount0, total_amount1 = distribution.get_distribution()

    # Plot the distribution
    plot_liquidity_distribution(result, current_tick, pool)

    print(f"Total Amount 0: {total_amount0}")
    print(f"Total Amount 1: {total_amount1}")


if __name__ == "__main__":
    main()
