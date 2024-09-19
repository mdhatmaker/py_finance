import json
from web3 import Web3
import pandas as pd

"""
This code connects to the Binance Smart Chain (BSC) and Ethereum networks to fetch token prices
and volume data from PancakeSwap and Uniswap. It calculates the average price and volume with
their respective standard deviations to identify breakouts.

Please note:

Replace YOUR_INFURA_PROJECT_ID with your Infura project ID.
Replace YOUR_ROUTER_ABI_JSON with the ABI JSON of the PancakeSwap/Uniswap router.
Replace YOUR_TOKEN_ADDRESS with the token address you are interested in.
"""


# Connect to BSC and Ethereum
bsc = Web3(Web3.HTTPProvider('https://bsc-dataseed.binance.org/'))
eth = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID'))

# PancakeSwap and Uniswap router contract addresses
pancake_router_address = Web3.toChecksumAddress('0x10ED43C718714eb63d5aA57B78B54704E256024E')
uniswap_router_address = Web3.toChecksumAddress('0x7a250d5630b4cf539739df2c5dAcb4c659F2488D')

# Router ABI
router_abi = json.loads('YOUR_ROUTER_ABI_JSON')

# Token addresses
token_address = Web3.toChecksumAddress('YOUR_TOKEN_ADDRESS')
weth_address = Web3.toChecksumAddress('0xC02aaA39b223FE8D0a0e5C4F27eAD9083C756Cc2') # WETH for Uniswap
wbnb_address = Web3.toChecksumAddress('0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c') # WBNB for PancakeSwap

# Initialize router contracts
pancake_router = bsc.eth.contract(address=pancake_router_address, abi=router_abi)
uniswap_router = eth.eth.contract(address=uniswap_router_address, abi=router_abi)

def get_price(router, token, base_token):
    amounts_out = router.functions.getAmountsOut(1 * 10**18, [token, base_token]).call()
    return amounts_out[1] / 10**18

def get_volume(router, token, base_token, start_block, end_block, provider):
    volume = 0
    for block in range(start_block, end_block):
        events = router.events.Swap().getLogs(fromBlock=block, toBlock=block)
        for event in events:
            if event['args']['path'][0] == token and event['args']['path'][-1] == base_token:
                volume += event['args']['amount0Out'] / 10**18
            elif event['args']['path'][0] == base_token and event['args']['path'][-1] == token:
                volume += event['args']['amount1Out'] / 10**18
    return volume

def identify_breakouts(df, price_threshold, volume_threshold):
    price_breakouts = df[df['price'] > price_threshold]
    volume_breakouts = df[df['volume'] > volume_threshold]
    return price_breakouts, volume_breakouts

# Define the block range to analyze
start_block_bsc = 10000000
end_block_bsc = 10000100
start_block_eth = 12000000
end_block_eth = 12000100

# Get price and volume data
data = []
for block in range(start_block_bsc, end_block_bsc):
    price = get_price(pancake_router, token_address, wbnb_address)
    volume = get_volume(pancake_router, token_address, wbnb_address, block, block + 1, bsc)
    data.append({'block': block, 'price': price, 'volume': volume})

for block in range(start_block_eth, end_block_eth):
    price = get_price(uniswap_router, token_address, weth_address)
    volume = get_volume(uniswap_router, token_address, weth_address, block, block + 1, eth)
    data.append({'block': block, 'price': price, 'volume': volume})

df = pd.DataFrame(data)

# Identify breakouts
price_threshold = df['price'].mean() + df['price'].std() * 2
volume_threshold = df['volume'].mean() + df['volume'].std() * 2

price_breakouts, volume_breakouts = identify_breakouts(df, price_threshold, volume_threshold)

print("Price Breakouts:\n", price_breakouts)
print("Volume Breakouts:\n", volume_breakouts)



