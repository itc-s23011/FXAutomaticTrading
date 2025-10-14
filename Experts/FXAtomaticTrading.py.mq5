"""
MT5 Python FX 自動売買サンプル
ファイル名: mt5_python_fx_robot.py

概要:
- MetaTrader5 Python モジュールを使い、MT5ターミナルへ接続して注文を出します。
- シンプルだが実用的な戦略:
    - EMAクロス (短期EMA 20, 長期EMA 50)
    - ATR を使ったボラティリティベースのSL/TP
    - RSI フィルターでノイズを減らす
- リスク管理:
    - 口座残高の割合でロットサイズ計算（リスク/トレード）
    - 最大同時ポジション数、取引間隔の制限
- 実行: ライブ／デモでの使用前にバックテスト・デモ検証を必ず行ってください。

前提:
- Python 3.8+
- pip install MetaTrader5 pandas numpy ta
- MT5 ターミナルを起動し、取引アカウントにログインしておく

使い方:
- 設定部分(config)を自分の環境に合わせて編集
- dry_run=True にして戦略を紙トレード（ログ確認）
- 動作確認後、dry_run=False にして実運用

重要: 自動売買は損失を出す可能性があります。資金を失っても自己責任です。
"""

import time
import math
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta  # technical analysis helpers (pip install ta)

# ---------------------- Configuration ----------------------
CONFIG = {
    'mt5_path': None,  # 例: r"C:\Program Files\MetaTrader 5\terminal64.exe" (通常は不要)
    'login': None,     # 数字のアカウントID （MT5にすでにログイン済みならNoneでも可）
    'password': None,
    'server': None,

    'symbol': 'USDJPY',
    'timeframe': mt5.TIMEFRAME_M15,  # M5, M15, H1など
    'history_bars': 2000,

    # Strategy parameters
    'ema_fast': 20,
    'ema_slow': 50,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'atr_period': 14,
    'atr_multiplier_sl': 1.5,
    'atr_multiplier_tp': 2.5,

    # Risk management
    'risk_per_trade': 0.01,  # 口座残高の割合（例: 0.01 = 1%）
    'max_positions': 3,
    'min_seconds_between_trades': 60 * 30,  # 30分

    'magic': 123456,
    'deviation': 10,  # スリッページ（ポイント）

    'dry_run': True,  # Trueなら発注せずログのみ
}

# ---------------------- Utility functions ----------------------

def connect_mt5():
    if CONFIG['mt5_path']:
        mt5.initialize(path=CONFIG['mt5_path'])
    else:
        mt5.initialize()

    if CONFIG['login']:
        authorized = mt5.login(CONFIG['login'], password=CONFIG['password'], server=CONFIG['server'])
        if not authorized:
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    # check symbol availability
    if not mt5.symbol_select(CONFIG['symbol'], True):
        raise RuntimeError(f"Failed to select symbol {CONFIG['symbol']}")


def disconnect_mt5():
    mt5.shutdown()


def fetch_ohlc(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise RuntimeError(f"Failed to fetch rates: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df


def calc_indicators(df):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=CONFIG['ema_fast'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=CONFIG['ema_slow'], adjust=False).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], CONFIG['rsi_period']).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], CONFIG['atr_period']).average_true_range()
    return df


def get_account_info():
    return mt5.account_info()


def calc_lot_size(symbol, stop_loss_price):
    # リスク金額 = 口座残高 * risk_per_trade
    acc = get_account_info()
    if acc is None:
        raise RuntimeError('Failed to get account info')
    balance = acc.balance
    risk_amount = balance * CONFIG['risk_per_trade']

    # pip value per lot estimation
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError('Failed to get symbol info')

    point = symbol_info.point
    contract_size = symbol_info.trade_contract_size

    # price distance in points
    # 注意: JPYペアは小数点が異なることがある
    # distance (in price terms): abs(entry - SL)
    # We'll compute lot by: lot = risk_amount / (distance * contract_size)
    # convert distance to price units
    # Here caller should pass correct price distance
    return balance, risk_amount, point, contract_size


def estimate_lots_by_sl(symbol, entry_price, sl_price):
    # 決定ロットを推定する：スリッページや証拠金はブローカーによる
    acc = get_account_info()
    if acc is None:
        raise RuntimeError('Failed to get account info')
    balance = acc.balance
    risk_amount = balance * CONFIG['risk_per_trade']

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError('Failed to get symbol info')

    point = symbol_info.point
    contract_size = symbol_info.trade_contract_size

    # price distance in price units
    distance = abs(entry_price - sl_price)
    if distance <= 0:
        return 0.0

    # For FX pairs pip is price change; approximate money loss per lot = distance * contract_size
    # If symbol uses 2-5 decimal places, contract_size accounts for units per lot
    loss_per_lot = distance * contract_size
    lots = risk_amount / loss_per_lot

    # Round down to broker-allowed step
    lot_step = symbol_info.volume_step
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lots = math.floor(lots / lot_step) * lot_step
    lots = max(min_lot, min(lots, max_lot))
    return float(round(lots, 2))


# ---------------------- Trading logic ----------------------

last_trade_time = None


def generate_signal(df):
    # シグナル: EMAクロス + RSIフィルタ
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # cross up
    buy = (prev['ema_fast'] <= prev['ema_slow']) and (last['ema_fast'] > last['ema_slow']) and (last['rsi'] < CONFIG['rsi_overbought'])
    sell = (prev['ema_fast'] >= prev['ema_slow']) and (last['ema_fast'] < last['ema_slow']) and (last['rsi'] > CONFIG['rsi_oversold'])
    return 'buy' if buy else ('sell' if sell else None)


def count_open_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len(positions)


def place_order(symbol, order_type, lots, price, sl_price, tp_price):
    if CONFIG['dry_run']:
        print(f"[DRY RUN] Would place {order_type} {lots} lots {symbol} entry {price} SL {sl_price} TP {tp_price}")
        return None

    symbol_info_tick = mt5.symbol_info_tick(symbol)
    if symbol_info_tick is None:
        print('Failed to get tick')
        return None

    if order_type == 'buy':
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': lots,
            'type': mt5.ORDER_TYPE_BUY,
            'price': price,
            'sl': sl_price,
            'tp': tp_price,
            'deviation': CONFIG['deviation'],
            'magic': CONFIG['magic'],
            'comment': 'auto_ema_atr'
        }
    else:
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': lots,
            'type': mt5.ORDER_TYPE_SELL,
            'price': price,
            'sl': sl_price,
            'tp': tp_price,
            'deviation': CONFIG['deviation'],
            'magic': CONFIG['magic'],
            'comment': 'auto_ema_atr'
        }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode} - {result.comment}")
    else:
        print(f"Order placed: ticket={result.order} volume={lots} price={price}")
    return result


# ---------------------- Main loop ----------------------

def main_loop():
    global last_trade_time
    connect_mt5()
    print('Connected to MT5')

    try:
        while True:
            df = fetch_ohlc(CONFIG['symbol'], CONFIG['timeframe'], CONFIG['history_bars'])
            df = calc_indicators(df)

            signal = generate_signal(df)
            print(datetime.now(), f"Signal: {signal}")

            # throttle
            now = datetime.now()
            if last_trade_time and (now - last_trade_time).total_seconds() < CONFIG['min_seconds_between_trades']:
                print('Waiting due to min_seconds_between_trades')
                time.sleep(5)
                continue

            # check open positions limit
            open_pos = count_open_positions(CONFIG['symbol'])
            if open_pos >= CONFIG['max_positions']:
                print('Max positions reached, skip')
                time.sleep(5)
                continue

            if signal is not None:
                last_bar = df.iloc[-1]
                atr = last_bar['atr']
                close = last_bar['close']

                if np.isnan(atr) or atr <= 0:
                    print('Invalid ATR, skipping')
                    time.sleep(5)
                    continue

                if signal == 'buy':
                    entry_price = mt5.symbol_info_tick(CONFIG['symbol']).ask
                    sl_price = entry_price - CONFIG['atr_multiplier_sl'] * atr
                    tp_price = entry_price + CONFIG['atr_multiplier_tp'] * atr
                else:
                    entry_price = mt5.symbol_info_tick(CONFIG['symbol']).bid
                    sl_price = entry_price + CONFIG['atr_multiplier_sl'] * atr
                    tp_price = entry_price - CONFIG['atr_multiplier_tp'] * atr

                # estimate lots
                lots = estimate_lots_by_sl(CONFIG['symbol'], entry_price, sl_price)
                if lots <= 0:
                    print('Calculated lots <= 0, skip')
                    time.sleep(5)
                    continue

                # place order
                res = place_order(CONFIG['symbol'], signal, lots, entry_price, sl_price, tp_price)
                last_trade_time = datetime.now()

            # sleep until next candle roughly
            time.sleep(10)
    except KeyboardInterrupt:import time
result = mt5.order_send(request)
if result.retcode != mt5.TRADE_RETCODE_DONE:
print(f"Order failed: {result.retcode} - {result.comment}")
else:
print(f"Order placed: ticket={result.order} volume={lots} price={price}")
return result




# ---------------------- Main loop ----------------------


def main_loop():
global last_trade_time
connect_mt5()
print('Connected to MT5')


try:
while True:
df = fetch_ohlc(CONFIG['symbol'], CONFIG['timeframe'], CONFIG['history_bars'])
df = calc_indicators(df)


signal = generate_signal(df)
print(datetime.now(), f"Signal: {signal}")


# throttle
now = datetime.now()
if last_trade_time and (now - last_trade_time).total_seconds() < CONFIG['min_seconds_between_trades']:
print('Waiting due to min_seconds_between_trades')
time.sleep(5)
continue


# check open positions limit
open_pos = count_open_positions(CONFIG['symbol'])
if open_pos >= CONFIG['max_positions']:
print('Max positions reached, skip')
time.sleep(5)
continue


if signal is not None:
last_bar = df.iloc[-1]
atr = last_bar['atr']
close = last_bar['close']


if np.isnan(atr) or atr <= 0:
print('Invalid ATR, skipping')
time.sleep(5)
continue


if signal == 'buy':
entry_price = mt5.symbol_info_tick(CONFIG['symbol']).ask
sl_price = entry_price - CONFIG['atr_multiplier_sl'] * atr
tp_price = entry_price + CONFIG['atr_multiplier_tp'] * atr
else:
entry_price = mt5.symbol_info_tick(CONFIG['symbol']).bid
sl_price = entry_price + CONFIG['atr_multiplier_sl'] * atr
tp_price = entry_price - CONFIG['atr_multiplier_tp'] * atr


# estimate lots
lots = estimate_lots_by_sl(CONFIG['symbol'], entry_price, sl_price)
if lots <= 0:
print('Calculated lots <= 0, skip')
time.sleep(5)
continue


# place order
res = place_order(CONFIG['symbol'], signal, lots, entry_price, sl_price, tp_price)
last_trade_time = datetime.now()


# sleep until next candle roughly
time.sleep(10)
except KeyboardInterrupt:
print('Interrupted by user')
finally:
disconnect_mt5()
print('Disconnected')




if __name__ == '__main__':
print('Starting robot (dry_run={})'.format(CONFIG['dry_run']))
main_loop()
        print('Interrupted by user')
    finally:
        disconnect_mt5()
        print('Disconnected')


if __name__ == '__main__':
    print('Starting robot (dry_run={})'.format(CONFIG['dry_run']))
    main_loop()
