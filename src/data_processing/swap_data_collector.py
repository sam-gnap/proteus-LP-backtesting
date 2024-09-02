import os
from datetime import date, timedelta
from google.cloud import bigquery
import pandas as pd
from typing import List, Tuple

# Import the necessary parts from your existing code
from queries.big_query import build_query_block_based_sampling
from abis.uniswap_v3_abis import SWAP_V3_JS_CODE

V3_SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

class SwapDataCollector:
    def __init__(self, project_id: str, pool_address: str, start_date: date, end_date: date):
        self.project_id = project_id
        self.pool_address = pool_address
        self.start_date = start_date
        self.end_date = end_date
        self.client = bigquery.Client(project=project_id)
        self.base_dir = os.path.join("data", "uniswap", "swaps")
        os.makedirs(self.base_dir, exist_ok=True)

    def get_swap_data(self, start_date: date, end_date: date, block_interval: int = 100) -> pd.DataFrame:
        query = build_query_block_based_sampling(
            SWAP_V3_JS_CODE, 
            start_date, 
            end_date, 
            V3_SWAP_TOPIC, 
            self.pool_address, 
            block_interval
        )
        df = self.client.query(query).to_dataframe()
        return df

    def collect_and_store_swap_data(self, time_step: int = 10, block_interval: int = 100):
        current_date = self.start_date
        all_swaps = []

        while current_date <= self.end_date:
            end_of_period = min(current_date + timedelta(days=time_step - 1), self.end_date)
            print(f"Processing swaps for period: {current_date.isoformat()} to {end_of_period.isoformat()}")
            
            swaps_df = self.get_swap_data(current_date, end_of_period, block_interval)
            if not swaps_df.empty:
                all_swaps.append(swaps_df)
            
            current_date = end_of_period + timedelta(days=1)

        if all_swaps:
            combined_swaps = pd.concat(all_swaps, ignore_index=True)
            filename = os.path.join(self.base_dir, f"swaps_{self.start_date.isoformat()}_to_{self.end_date.isoformat()}.parquet")
            combined_swaps.to_parquet(filename, compression='snappy')
            print(f"All swap data saved to {filename}")
        else:
            print("No swap data collected.")

    def load_swap_data(self) -> pd.DataFrame:
        filename = os.path.join(self.base_dir, f"swaps_{self.start_date.isoformat()}_to_{self.end_date.isoformat()}.parquet")
        if os.path.exists(filename):
            return pd.read_parquet(filename)
        else:
            print(f"Swap data file not found: {filename}")
            return pd.DataFrame()



