from datetime import date
from typing import Optional


def build_query(
    js_code: str, start_date: date, end_date: date, topic: str, pool_address: Optional[str] = None
) -> str:
    address_filter = f"AND logs.address = '{pool_address}'" if pool_address else ""
    return f"""
    CREATE TEMP FUNCTION
    PARSE_LOG(data STRING, topics ARRAY<STRING>)
    RETURNS STRUCT<
        `sender` STRING,
        `recipient` STRING,
        `tickLower` STRING,
        `tickUpper` STRING,
        `amount` STRING,
        `amount0` STRING,
        `amount1` STRING,
        `sqrtPriceX96` STRING,
        `liquidity` STRING,
        `amount0Int` STRING,
        `amount1Int` STRING,
        `tick` STRING
    >
    LANGUAGE js AS \"\"\"
    {js_code}
    \"\"\"
    OPTIONS
    ( library="https://storage.googleapis.com/ethlab-183014.appspot.com/ethjs-abi.js" );

    WITH parsed_logs AS (
    SELECT
        logs.address,
        logs.block_timestamp,
        logs.block_number,
        logs.transaction_hash,
        logs.log_index,
        logs.topics[SAFE_OFFSET(0)] AS event_signature,
        PARSE_LOG(logs.data, logs.topics) AS parsed
    FROM `bigquery-public-data.crypto_ethereum.logs` AS logs
    WHERE DATE(logs.block_timestamp) BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
        {address_filter}
        AND logs.topics[SAFE_OFFSET(0)] = '{topic}'
    )

    SELECT
    block_timestamp,
    block_number,
    transaction_hash,
    address AS pool_address,
    event_signature,
    parsed.sender,
    parsed.recipient,
    parsed.tickLower,
    parsed.tickUpper,
    parsed.amount,
    parsed.amount0,
    parsed.amount1,
    parsed.sqrtPriceX96,
    parsed.liquidity,
    parsed.amount0Int,
    parsed.amount1Int,
    parsed.tick
    FROM parsed_logs
    ORDER BY block_timestamp, log_index ASC
    """


def build_query_block_based_sampling(
    js_code: str,
    start_date: date,
    end_date: date,
    topic: str,
    pool_address: Optional[str] = None,
    block_interval: int = 100,
) -> str:
    address_filter = f"AND address = '{pool_address}'" if pool_address else ""
    return f"""
    CREATE TEMP FUNCTION
    PARSE_LOG(data STRING, topics ARRAY<STRING>)
    RETURNS STRUCT<
        `sender` STRING,
        `recipient` STRING,
        `tickLower` STRING,
        `tickUpper` STRING,
        `amount` STRING,
        `amount0` STRING,
        `amount1` STRING,
        `sqrtPriceX96` STRING,
        `liquidity` STRING,
        `amount0Int` STRING,
        `amount1Int` STRING,
        `tick` STRING
    >
    LANGUAGE js AS \"\"\"
    {js_code}
    \"\"\"
    OPTIONS
    ( library="https://storage.googleapis.com/ethlab-183014.appspot.com/ethjs-abi.js" );

    WITH parsed_logs AS (
    SELECT
        address,
        block_timestamp,
        block_number,
        transaction_hash,
        log_index,
        topics[SAFE_OFFSET(0)] AS event_signature,
        PARSE_LOG(data, topics) AS parsed
    FROM `bigquery-public-data.crypto_ethereum.logs`
    WHERE DATE(block_timestamp) BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
        {address_filter}
        AND topics[SAFE_OFFSET(0)] = '{topic}'
    ),
    block_groups AS (
    SELECT 
        *,
        DIV(block_number, {block_interval}) AS block_group
    FROM parsed_logs
    ),
    sampled_logs AS (
    SELECT 
        block_group,
        ARRAY_AGG(block_groups ORDER BY block_number, log_index LIMIT 1)[OFFSET(0)] AS log_data
    FROM block_groups
    GROUP BY block_group
    )

    SELECT
    log_data.block_timestamp,
    log_data.block_number,
    log_data.transaction_hash,
    log_data.address AS pool_address,
    log_data.event_signature,
    log_data.parsed.sender,
    log_data.parsed.recipient,
    log_data.parsed.tickLower,
    log_data.parsed.tickUpper,
    log_data.parsed.amount,
    log_data.parsed.amount0,
    log_data.parsed.amount1,
    log_data.parsed.sqrtPriceX96,
    log_data.parsed.liquidity,
    log_data.parsed.amount0Int,
    log_data.parsed.amount1Int,
    log_data.parsed.tick
    FROM sampled_logs
    ORDER BY log_data.block_number, log_data.log_index ASC
    """
