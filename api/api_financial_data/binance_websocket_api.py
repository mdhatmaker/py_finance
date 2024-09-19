import json
import websockets
import asyncio
from datetime import datetime
from typing import List

# Binance API docs
# https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Connect
# https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Live-Subscribing-Unsubscribing-to-streams
# https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Kline-Candlestick-Streams
"""
Base Url: wss://fstream.binance.com
Raw Streams: /ws/<streamName>
Combined Streams: /stream/streams=<streamName1>/<streamName2>/<streamName3>
example: wss://fstream.binance.com/ws/bnbusdt@aggTrade
example: wss://fstream.binance.com/ws/bnbusdt@depth
example: wss://fstream.binance.com/ws/bnbusdt@kline_<interval>
where intervals like: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
example: wss://fstream.binance.com/stream?streams=bnbusdt@aggTrade/btcusdt@markPrice
example: headers {"method": "SUBSCRIBE", "params": ["btcusdt@kline_1m"], "id": 1}
         url wss://fstream.binance.com/stream?streams=btcusdt@kline_1m
"""

# {"method": "SUBSCRIBE", "params": ["btcusdt@kline_1m"], "id": 1}
# {"method": "SUBSCRIBE", "params": ["btcusdt@kline_1m", "ethusdt@kline_1m"], "id": 1}


async def run_websocket(symbols: List[str], id: int):
    url = "wss://fstream.binance.com/stream?streams=btcusdt@kline_1m"
    async with websockets.connect(url, ping_timeout=None, max_size=10000000) as websocket:
        # headers = {"method": "SUBSCRIBE", "params": ["btcusdt@kline_1m"], "id": 1}
        headers = {"method": "SUBSCRIBE", "params": symbols, "id": id}
        await websocket.send(json.dumps(headers))
        while True:
            # msg = await websocket.recv()
            # data = json.loads(msg)
            # print(data)
            msg = await websocket.recv()
            raw_data = json.loads(msg)
            if 'result' in raw_data:
                continue
            data = raw_data['data']
            pair = data['s']
            candle = data['k']
            dt_format = '%Y-%m-%d %H:%M:%S:%f'
            event_time = datetime.fromtimestamp(data['E'] / 1000).strftime(dt_format)
            minute = datetime.fromtimestamp(candle['t'] / 1000).strftime('%H:%M')
            price_stub = "{:.2f}"
            col_width = 8
            open_price = f"{price_stub.format(float(candle['o']))}{' ' * (col_width - len(price_stub.format(float(candle['o']))))}"
            close_price = f"{price_stub.format(float(candle['c']))}{' ' * (col_width - len(price_stub.format(float(candle['c']))))}"
            high_price = f"{price_stub.format(float(candle['h']))}{' ' * (col_width - len(price_stub.format(float(candle['h']))))}"
            low_price = f"{price_stub.format(float(candle['l']))}{' ' * (col_width - len(price_stub.format(float(candle['l']))))}"
            volume = candle['v']
            print(f'[{event_time}] {pair} - minute: {minute} | open: {open_price} | close: {close_price} | '
                  f'high: {high_price} | low: {low_price} | volume: {volume}')


if __name__ == '__main__':
    asyncio.run(run_websocket())



"""
Example of single response:
{
    "stream": "btcusdt@kline_1m",
    "data": {
        "e": "kline",
        "E": 1714226603449,
        "s": "BTCUSDT",
        "k": {
            "t": 1714226580000, //start time in epoch format
            "T": 1714226639999, //end time in epoch format
            "s": "BTCUSDT",
            "i": "1m", //interval
            "f": 4942154251,
            "L": 4942155024,
            "o": "63127.00", //open price
            "c": "63144.80", //close price
            "h": "63152.80", //high price
            "l": "63126.90", //low price
            "v": "48.194", //volume
            "n": 774, //number of trades
            "x": False, //is the kline closed?
            "q": "3042797.35400",
            "V": "35.753",
            "Q": "2257302.61500",
            "B": "0"
        }
    }
}
"""
