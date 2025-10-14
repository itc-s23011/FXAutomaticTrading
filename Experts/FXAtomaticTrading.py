import time
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