from datetime import date
from swap_data_collector import SwapDataCollector


def main():
    collector = SwapDataCollector(
        project_id="uniswap-v3-analytics",
        pool_address="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        token0="WETH",
        token1="USDC",
        decimals0=6,
        decimals1=18,
        tick_spacing=60,
        start_date=date(2024, 8, 1),
        end_date=date(2024, 8, 31),
    )

    # Collect and sample blocks
    collector.collect_and_sample_blocks_bars(method="dollar", threshold=100000)

    # Load all sampled blocks
    sampled_blocks = collector.load_sampled_blocks_adaptive()


if __name__ == "__main__":
    main()
