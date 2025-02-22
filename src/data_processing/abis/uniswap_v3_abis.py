MINT_V3_JS_CODE = """
    let parsedEvent = {
        "anonymous": false,
        "inputs": [{
            "indexed": false,
            "internalType": "address",
            "name": "sender",
            "type": "address"
        }, {
            "indexed": true,
            "internalType": "address",
            "name": "owner",
            "type": "address"
        }, {
            "indexed": true,
            "internalType": "int24",
            "name": "tickLower",
            "type": "int24"
        }, {
            "indexed": true,
            "internalType": "int24",
            "name": "tickUpper",
            "type": "int24"
        }, {
            "indexed": false,
            "internalType": "uint128",
            "name": "amount",
            "type": "uint128"
        }, {
            "indexed": false,
            "internalType": "uint256",
            "name": "amount0",
            "type": "uint256"
        }, {
            "indexed": false,
            "internalType": "uint256",
            "name": "amount1",
            "type": "uint256"
        }],
        "name": "Mint",
        "type": "event"
    }
    return abi.decodeEvent(parsedEvent, data, topics, false);
"""

BURN_V3_JS_CODE = """
    let parsedEvent = {
        "anonymous": false,
        "inputs": [{
            "indexed": true,
            "internalType": "address",
            "name": "owner",
            "type": "address"
        }, {
            "indexed": true,
            "internalType": "int24",
            "name": "tickLower",
            "type": "int24"
        }, {
            "indexed": true,
            "internalType": "int24",
            "name": "tickUpper",
            "type": "int24"
        }, {
            "indexed": false,
            "internalType": "uint128",
            "name": "amount",
            "type": "uint128"
        }, {
            "indexed": false,
            "internalType": "uint256",
            "name": "amount0",
            "type": "uint256"
        }, {
            "indexed": false,
            "internalType": "uint256",
            "name": "amount1",
            "type": "uint256"
        }],
        "name": "Burn",
        "type": "event"
    }
    return abi.decodeEvent(parsedEvent, data, topics, false);
"""

COLLECT_V3_JS_CODE = """
    let parsedEvent = {
        "anonymous": false,
        "inputs": [{
            "indexed": true,
            "internalType": "address",
            "name": "owner",
            "type": "address"
        }, {
            "indexed": false,
            "internalType": "address",
            "name": "recipient",
            "type": "address"
        }, {
            "indexed": true,
            "internalType": "int24",
            "name": "tickLower",
            "type": "int24"
        }, {
            "indexed": true,
            "internalType": "int24",
            "name": "tickUpper",
            "type": "int24"
        }, {
            "indexed": false,
            "internalType": "uint128",
            "name": "amount0",
            "type": "uint128"
        }, {
            "indexed": false,
            "internalType": "uint128",
            "name": "amount1",
            "type": "uint128"
        }],
        "name": "Collect",
        "type": "event"
    }
    return abi.decodeEvent(parsedEvent, data, topics, false);
"""

SWAP_V3_JS_CODE = """
    let parsedEvent = {
        "anonymous": false,
        "inputs": [{
            "indexed": true,
            "internalType": "address",
            "name": "sender",
            "type": "address"
        }, {
            "indexed": true,
            "internalType": "address",
            "name": "recipient",
            "type": "address"
        }, {
            "indexed": false,
            "internalType": "int256",
            "name": "amount0",
            "type": "int256"
        }, {
            "indexed": false,
            "internalType": "int256",
            "name": "amount1",
            "type": "int256"
        }, {
            "indexed": false,
            "internalType": "uint160",
            "name": "sqrtPriceX96",
            "type": "uint160"
        }, {
            "indexed": false,
            "internalType": "uint128",
            "name": "liquidity",
            "type": "uint128"
        }, {
            "indexed": false,
            "internalType": "int24",
            "name": "tick",
            "type": "int24"
        }],
        "name": "Swap",
        "type": "event"
    }
    return abi.decodeEvent(parsedEvent, data, topics, false);
"""
