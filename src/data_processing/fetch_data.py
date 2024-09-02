import requests
from typing import Dict, Any
from ..utils.helper import fee_tier_to_tick_spacing, fetch_all_data

POOL_QUERY = """query get_pools($pool_id: ID!) {
  pools(where: {id: $pool_id}) {
    tick
    sqrtPrice
    liquidity
    feeTier
    token0 {
      symbol
      decimals
    }
    token1 {
      symbol
      decimals
    }
  }
}"""


def fetch_oku_liquidity(
    pool_address: str, block_number: int, chain: str = "ethereum"
) -> Dict[str, Any]:
    url = f"https://omni.icarus.tools/{chain}/cush/simulatePoolLiquidity"
    payload = {"params": [pool_address, block_number]}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()["result"]


def fetch_data_subgraph(
    api_key, query, variables=None, data_key=None, first_n=None, batch_size=1000
):
    # Not all ticks can be initialized. Tick spacing is determined by the pool's fee tier.
    url = f"https://gateway-arbitrum.network.thegraph.com/api/{api_key}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
    all_data = []
    skip = 0

    while True:
        # Update the skip variable in the query variables
        if variables is None:
            variables = {}
        variables["skip"] = skip

        response = requests.post(url, json={"query": query, "variables": variables})

        if response.status_code != 200:
            print(f"Query failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            break

        result = response.json()

        # If data_key is not provided, assume the data is at the top level
        data = result["data"][data_key] if data_key else result["data"]

        if not data:
            break

        all_data.extend(data)
        print(f"Fetched {len(all_data)} records")

        # Break if we've fetched the desired number of records
        if first_n and len(all_data) >= first_n:
            all_data = all_data[:first_n]
            break

        # Break if we've fetched less than the batch size (indicating we've reached the end)
        if len(data) < batch_size:
            break

        skip += batch_size

    return all_data


def fetch_pool_data(pool_address, API_KEY):
    variables = {"pool_id": pool_address}
    pool = fetch_data_subgraph(
        API_KEY, POOL_QUERY, variables, data_key="pools", first_n=100, batch_size=1000
    )[0]

    tick_spacing = fee_tier_to_tick_spacing(int(pool["feeTier"]))
    token0 = pool["token0"]["symbol"]
    token1 = pool["token1"]["symbol"]
    decimals0 = int(pool["token0"]["decimals"])
    decimals1 = int(pool["token1"]["decimals"])
    sqrt_price = int(pool["sqrtPrice"])
    fee_tier = int(pool["feeTier"])

    return sqrt_price, tick_spacing, token0, token1, decimals0, decimals1, fee_tier
