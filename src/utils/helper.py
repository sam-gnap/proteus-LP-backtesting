from decimal import Decimal


def sqrtPriceX96_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> Decimal:
    sqrt_price = Decimal(sqrt_price_x96) / Decimal(2**96)
    price = sqrt_price**2
    price = price * Decimal(10 ** (decimals0 - decimals1))
    return price


def organize_tick_data(tick_data):
    tick_mapping = {}
    for item in tick_data:
        liquidity_net = int(item["liquidity_net"])
        if liquidity_net > 2**255:
            liquidity_net -= 2**256
        tick_mapping[int(item["tick_index"])] = liquidity_net
    return tick_mapping


def tick_to_price(TICK_BASE, tick):
    return TICK_BASE**tick


def fee_tier_to_tick_spacing(fee_tier):
    return {100: 1, 500: 10, 3000: 60, 10000: 200}.get(fee_tier, 60)


def fetch_all_data(API_KEY, query, variables, data_key, first_n, batch_size):
    # Implement the logic to fetch data from the API
    pass
