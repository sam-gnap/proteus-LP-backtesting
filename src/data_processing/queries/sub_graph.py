POOL_QUERY = """
query get_pools($pool_id: ID!) {
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
}
"""

TOP_POOLS_QUERY = """
query get_top_pools($first: Int!) {
  pools(first: $first, orderBy: totalValueLockedUSD, orderDirection: desc) {
    id
    token0 {
      symbol
    }
    token1 {
      symbol
    }
    feeTier
    totalValueLockedUSD
  }
}
"""
