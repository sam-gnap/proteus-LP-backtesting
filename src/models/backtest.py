import numpy as np
import pandas as pd
from datetime import date
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from src.data_processing.fetch_data import fetch_data_subgraph
from src.models.liquidity_distribution import Pool, LiquidityPeriphery
import logging

pd.set_option("display.max_columns", 100)


class Position:
    def __init__(
        self,
        pool,
        current_tick: int,
        lower_tick: int,
        upper_tick: int,
        amount0: float,
        amount1: float,
    ):
        self.fees_earned_tvl = 0
        self.fees_earned_liquidity = 0
        self.pool = pool
        self.amount0 = amount0
        self.amount1 = amount1
        self.lower_tick = int(lower_tick)
        self.upper_tick = int(upper_tick)
        self.current_tick = int(current_tick)
        self.current_price = float(
            1 / self.pool.adjust_price(self.pool.tick_to_price(int(self.current_tick)))
        )
        self.initial_price = self.current_price
        self.capital_USD = self.amount0 * self.current_price + self.amount1
        self.capital_USD_liquidity = capital_USD

        self.capital_liquidity = self.calculate_liquidity(
            self.current_tick,
        )
        self.capital_liquidity = int(self.capital_liquidity)

    def calculate_liquidity(self, current_tick):
        liquidity = int(
            liquidity_calculator.calculate_liquidity(
                self.amount0, self.amount1, self.lower_tick, self.upper_tick, current_tick
            )[5]
        )
        return liquidity

    def update_position(self, current_tick):
        # try:
        self.amount0, self.amount1 = liquidity_calculator.get_amounts_for_liquidity(
            current_tick, self.lower_tick, self.upper_tick, self.capital_liquidity
        )
        self.capital_USD = self.amount0 * self.current_price + self.amount1
        # except Exception as e:
        #     print("This is error at line 59:", e)

    def is_in_range(self, current_tick: int) -> bool:
        return self.lower_tick <= current_tick <= self.upper_tick


from math import sqrt


class LiquidityProvisionBacktester:
    def __init__(self, pool, liquidity_calculator, log_file):
        self.pool = pool
        self.liquidity_calculator = liquidity_calculator
        self.logger = self.setup_logger(log_file)

    def setup_logger(self, log_file=None):
        logger = logging.getLogger("LiquidityProvisionBacktester")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if log_file is provided)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def log_rebalance(
        self,
        current_block,
        current_datetime,
        rebalance_count,
        i,
        current_price,
        current_tick,
        old_lower_tick,
        old_upper_tick,
        position,
        old_amount0,
        old_amount1,
    ):
        rebalance_info = [
            f"{'='*20} REBALANCE EVENT {'='*20}",
            f"Block: {current_block}",
            f"Timestamp: {current_datetime}",
            f"Rebalance count: {rebalance_count} within {i} blocks",
            f"Current price: {current_price:.6f}, Current tick: {current_tick}",
            f"Old range: [{old_lower_tick}, {old_upper_tick}], New range: [{position.lower_tick}, {position.upper_tick}]",
            f"Old amounts: amount0={old_amount0:.6f}, amount1={old_amount1:.6f}",
            f"New amounts: amount0={position.amount0:.6f}, amount1={position.amount1:.6f}",
            f"Fees earned since last rebalance: TVL={position.fees_earned_tvl:.6f}, Liquidity={position.fees_earned_liquidity:.6f}",
            f"Current position value: {position.capital_USD:.6f}",
            f"{'='*50}",
        ]
        for line in rebalance_info:
            self.logger.info(line)

    def simulate_swap(self, amount_in: float, is_token0: bool, current_price: float) -> float:
        # This is a simplified swap simulation. In reality, you'd use the pool's liquidity
        # curve and account for slippage. This simple version assumes no slippage.
        if is_token0:
            return amount_in * current_price
        else:
            return amount_in / current_price

    def rebalance_position(
        self, position: Position, current_tick: int, current_price: float, tick_range: int
    ):
        position.update_position(current_tick)
        # Calculate total position value in USD
        total_value_usd = position.amount0 * current_price + position.amount1

        # Calculate target amounts for a 50/50 split (adjust as needed for your strategy)
        target_value_each = total_value_usd / 2
        position.amount0 = target_value_each / current_price
        position.amount1 = target_value_each

        # Update position parameters
        position.current_tick = current_tick
        position.lower_tick = current_tick - tick_range
        position.upper_tick = current_tick + tick_range
        position.current_price = current_price

        # Recalculate liquidity with new balanced amounts
        position.capital_liquidity = position.calculate_liquidity(current_tick)

        return position

    def calculate_impermanent_loss(
        self, initial_price, current_price, initial_amount0, initial_amount1
    ):
        price_ratio = current_price / initial_price
        sqrt_price_ratio = sqrt(price_ratio)

        il_factor = 2 * sqrt_price_ratio / (1 + price_ratio) - 1
        portfolio_value_with_il = (
            initial_amount0 * current_price * sqrt_price_ratio + initial_amount1 / sqrt_price_ratio
        )
        portfolio_value_without_il = initial_amount0 * current_price + initial_amount1

        impermanent_loss = (
            portfolio_value_with_il - portfolio_value_without_il
        ) / portfolio_value_without_il
        return impermanent_loss

    def should_rebalance(self, position: Position, current_tick: int) -> bool:
        should_rebalance = current_tick < position.lower_tick or current_tick > position.upper_tick
        return should_rebalance

    def calculate_fees(
        self,
        current_block: pd.Series,
        position: Position,
        active_tvl_current_tick_range: float,
        active_liquidity_current_tick_range: float,
    ) -> float:
        volume = current_block["cumulative_dollar_volume"]
        fee_rate = self.pool.fee_tier / 1e6
        rebalance_cost = position.capital_USD * 0.0001
        total_fees = volume * fee_rate
        tvl_share = position.capital_USD / active_tvl_current_tick_range
        liquidity_share = position.capital_liquidity / active_liquidity_current_tick_range
        fees_earned_tvl = total_fees * tvl_share
        fees_earned_liquidity = total_fees * liquidity_share
        return fees_earned_tvl, fees_earned_liquidity

    def calc_active_tvl(
        self, df_liquidity: pd.DataFrame, current_block: int, next_block: int, tick_range: int
    ) -> tuple:
        current_data = df_liquidity[df_liquidity["block_number"] == current_block]
        next_data = df_liquidity[df_liquidity["block_number"] == next_block]

        if current_data.empty or next_data.empty:
            raise ValueError(f"No data for blocks {current_block} or {next_block}")

        # there are better ways to calculate this
        # we should look into volatility
        active_tick_t = current_data["active_tick_adjusted"].iloc[0]
        active_tick_t_1 = next_data["active_tick_adjusted"].iloc[0]
        tick_difference = abs(active_tick_t_1 - active_tick_t)

        active_tvl_current_tick_range = current_data[
            (current_data["tick"] >= (active_tick_t - tick_difference))
            & (current_data["tick"] <= (active_tick_t + tick_difference))
        ]["amount_locked"].sum()
        active_liquidity_current_tick_range = current_data[
            (current_data["tick"] >= (active_tick_t - tick_difference))
            & (current_data["tick"] <= (active_tick_t + tick_difference))
        ]["liquidity"].sum()

        return active_tvl_current_tick_range, active_liquidity_current_tick_range

    def backtest_liquidity_provision(
        self,
        df_liquidity: pd.DataFrame,
        capital_USD: float,
        amount0: float,
        amount1: float,
        tick_range: int,
    ) -> pd.DataFrame:
        results = []
        position = None
        hodl_capital = capital_USD
        rebalance_count = 0

        blocks = df_liquidity["block_number"].unique()
        total_blocks = len(blocks) - 1

        self.logger.info(f"Starting backtest with {total_blocks} blocks")

        for i in range(total_blocks):
            current_block = blocks[i]
            next_block = blocks[i + 1]

            try:
                current_data = df_liquidity[df_liquidity["block_number"] == current_block].iloc[0]
                current_datetime = current_data["datetime"]
                current_tick = int(current_data["active_tick_adjusted"])
                current_price = float(current_data["active_price_inverted"])

                if position is None:
                    position = Position(
                        self.pool,
                        current_tick,
                        current_tick - tick_range,
                        current_tick + tick_range,
                        amount0,
                        amount1,
                    )
                    self.logger.info(f"Initial position created at block {current_block}")
                    self.logger.info(
                        f"Initial price: {current_price}, Initial tick: {current_tick}"
                    )
                    self.logger.info(f"Initial amount0: {amount0}, Initial amount1: {amount1}")

                active_tvl_current_tick_range, active_liquidity_current_tick_range = (
                    self.calc_active_tvl(df_liquidity, current_block, next_block, tick_range)
                )
                fees_earned_tvl, fees_earned_liquidity = self.calculate_fees(
                    current_data,
                    position,
                    active_tvl_current_tick_range,
                    active_liquidity_current_tick_range,
                )

                position.fees_earned_tvl += fees_earned_tvl
                position.fees_earned_liquidity += fees_earned_liquidity

                rebalanced = False
                tick_outside = self.should_rebalance(position, current_tick)
                if tick_outside:
                    rebalance_count += 1
                    rebalance_fee = 0  # position.capital_USD * rebalance_cost
                    position.fees_earned_tvl -= rebalance_fee
                    position.fees_earned_liquidity -= rebalance_fee

                    old_lower_tick, old_upper_tick = position.lower_tick, position.upper_tick
                    old_amount0, old_amount1 = position.amount0, position.amount1

                    position.current_price = current_price
                    position = self.rebalance_position(
                        position, current_tick, current_price, tick_range
                    )
                    rebalanced = True
                    self.log_rebalance(
                        current_block,
                        current_datetime,
                        rebalance_count,
                        i,
                        current_price,
                        current_tick,
                        old_lower_tick,
                        old_upper_tick,
                        position,
                        old_amount0,
                        old_amount1,
                    )

                impermanent_loss = self.calculate_impermanent_loss(
                    position.initial_price, current_price, amount0, amount1
                )

                # Update hodl capital
                hodl_capital *= (
                    current_price
                    / df_liquidity[df_liquidity["block_number"] == blocks[0]]["price"].iloc[0]
                )

                results.append(
                    {
                        "block": current_block,
                        "datetime": current_data["datetime"],
                        "price": current_price,
                        "position_capital_USD": position.capital_USD,
                        "capital_USD_liquidity": position.capital_USD_liquidity,
                        "hodl_capital": hodl_capital,
                        "fees_earned_tvl": fees_earned_tvl,
                        "fees_earned_liquidity": fees_earned_liquidity,
                        "rebalanced": rebalanced,
                        "impermanent_loss": impermanent_loss,
                    }
                )

            except Exception as e:
                print(f"Error at block {current_block}: {str(e)}")
                continue

        return pd.DataFrame(results)


class BacktestResults:
    def __init__(self):
        self.data = []

    def add_result(
        self, block, datetime, price, position_value, hodl_value, fees_earned, impermanent_loss
    ):
        self.data.append(
            {
                "block": block,
                "datetime": datetime,
                "price": price,
                "position_value": position_value,
                "hodl_value": hodl_value,
                "fees_earned": fees_earned,
                "impermanent_loss": impermanent_loss,
            }
        )

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def calculate_metrics(self):
        df = self.to_dataframe()
        total_return = (df["position_value"].iloc[-1] - df["position_value"].iloc[0]) / df[
            "position_value"
        ].iloc[0]
        hodl_return = (df["hodl_value"].iloc[-1] - df["hodl_value"].iloc[0]) / df[
            "hodl_value"
        ].iloc[0]
        total_fees = df["fees_earned"].sum()
        max_drawdown = (df["position_value"] / df["position_value"].cummax() - 1).min()

        returns = df["position_value"].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365)  # Assuming daily data

        return {
            "total_return": total_return,
            "hodl_return": hodl_return,
            "total_fees": total_fees,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

    def plot_results(self):
        df = self.to_dataframe()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

        ax1.plot(df["datetime"], df["position_value"], label="Position Value")
        ax1.plot(df["datetime"], df["hodl_value"], label="HODL Value")
        ax1.set_ylabel("Value (USD)")
        ax1.legend()
        ax1.set_title("Position Value vs HODL Value")

        ax2.plot(df["datetime"], df["fees_earned"].cumsum())
        ax2.set_ylabel("Cumulative Fees Earned (USD)")
        ax2.set_title("Cumulative Fees Earned")

        ax3.plot(df["datetime"], df["impermanent_loss"])
        ax3.set_ylabel("Impermanent Loss")
        ax3.set_title("Impermanent Loss Over Time")

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    API_KEY = "893d7471304c5edf436c8ba60781762c"
    POOL_ADDRESS = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640".lower()
    pool_query = """query get_pools($pool_id: ID!) {
    pools(where: {id: $pool_id}) {
        tick
        sqrtPrice
        liquidity
        feeTier
        totalValueLockedUSD
        totalValueLockedETH
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
    variables = {"pool_id": POOL_ADDRESS}
    data_pool = fetch_data_subgraph(
        API_KEY, pool_query, variables, data_key="pools", first_n=100, batch_size=1000
    )[0]
    pool = Pool(
        pool_address=POOL_ADDRESS,
        token0=data_pool["token0"]["symbol"],
        token1=data_pool["token1"]["symbol"],
        decimals0=data_pool["token0"]["decimals"],
        decimals1=data_pool["token1"]["decimals"],
        fee_tier=data_pool["feeTier"],
        sqrt_price_x96=data_pool["sqrtPrice"],
    )
    df_liquidity = pd.read_parquet(
        "/Users/gnapsamuel/Documents/AMM/proteus-LP-backtesting/data/liquidity_data.parquet"
    )

    initial_capital_amount_0 = 1  # ETH
    initial_capital_amount_1 = float(df_liquidity.iloc[0]["active_price_inverted"])  # USDC
    capital_USD = (
        initial_capital_amount_0 * (initial_capital_amount_1 / initial_capital_amount_0)
        + initial_capital_amount_1
    )
    tick_range = 100
    rebalance_cost = 0.0001
    current_tick = int(df_liquidity.iloc[0]["active_tick_adjusted"])

    lower_tick = current_tick - tick_range
    upper_tick = current_tick + tick_range
    log_file = "backtest_logs.txt"
    liquidity_calculator = LiquidityPeriphery(pool)
    df_liquidity_s = df_liquidity.loc[
        df_liquidity["block_number"].isin(df_liquidity.block_number.unique()[:20])
    ]

    backtester = LiquidityProvisionBacktester(pool, liquidity_calculator, log_file=log_file)
    results_df = backtester.backtest_liquidity_provision(
        df_liquidity_s,
        capital_USD,
        initial_capital_amount_0,
        initial_capital_amount_1,
        tick_range,
    )
