from datetime import date
from swap_data_collector import SwapDataCollector

def main():
    # Configuration
    project_id = 'uniswap-v3-analytics'
    pool_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"  # v3 USDC/ETH 0.3%
    start_date = date(2024, 8, 1)
    end_date = date(2024, 9, 1)
    time_step = 31 
    block_interval = 100

    # Initialize SwapDataCollector
    collector = SwapDataCollector(project_id, pool_address, start_date, end_date)

    # Collect and store swap data
    collector.collect_and_store_swap_data(time_step, block_interval)

    # Optional: Load and print some statistics about the collected data
    swap_data = collector.load_swap_data()
    if not swap_data.empty:
        print(f"Collected {len(swap_data)} swap events.")
        print(f"Date range: {swap_data['block_timestamp'].min()} to {swap_data['block_timestamp'].max()}")
        print(f"Block range: {swap_data['block_number'].min()} to {swap_data['block_number'].max()}")
    else:
        print("No swap data was collected.")

if __name__ == "__main__":
    main()
