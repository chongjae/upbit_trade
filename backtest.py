import os
import time
import json
import sqlite3
import urllib.request
from urllib.parse import urlencode
from datetime import datetime, timedelta
from trade import (
    PaperAccount,
    VolatilityBreakoutBot,
    ConnorsRSIBot,
    BollingerBandBot,
    MACDDivergenceBot,
    DynamicSwitchBot,
    CoinSelector,
    LAST_TRADE_TIME
)
import trade # To access globals directly if needed

BACKTEST_INITIAL_BALANCE = 1_000_000

# --- Helper: Resampling ---
def resample_candles(minute_candles, interval_minutes):
    """
    Resample 1-minute candles into larger intervals (15m, 60m, 1440m/Daily).
    Returns list of dicts (Newest -> Oldest)
    minute_candles: List of dicts (Newest -> Oldest according to Upbit API usually, but here likely chronological?
                    Let's assume input is Chronological (Old->New) to make accumulation easy, then reverse for bot?)
    """
    if not minute_candles: return []

    # Ensure input is Chronological (Old -> New)
    # Our DB fetch returns chronological? BacktestDB.get_candles_by_time implies point-in-time access.
    # The calling loop provides a historical buffer.

    # We will simply group the buffer.

    resampled = []
    current_candle = None

    # Assume minute_candles is recent history [Oldest, ..., Newest]
    # We iterate and group.

    # To align with clock (e.g. 15m candles start at 00, 15, 30, 45), we need timestamp parsing.
    # Upbit Daily closes at 09:00 KST.

    # Simplified Resampling:
    # Just standard bucket logic.

    # Optimization: Callers in backtest loop usually need "The historical candles ending NOW".
    # So we take the last N minutes.

    pass

class BacktestAccount(PaperAccount):
    def __init__(self, initial_krw=1_000_000):
        self.filename = "backtest_account.json"
        self.db_file = ":memory:"
        self.init_db()
        self.balance = initial_krw
        self.positions = {}
        self.orders = []
        self.trades_log = []

    def check_orders(self, current_candle_map):
        filled_orders = []
        remaining_orders = []

        for order in self.orders:
            market = order['market']
            if market not in current_candle_map:
                remaining_orders.append(order)
                continue

            candle = current_candle_map[market]
            # Candle Format: {'trade_price': ..., 'high_price': ..., 'low_price': ...}

            low = candle['low_price']
            high = candle['high_price']
            side = order['side']
            price = order['price']
            filled = False

            # Slippage/Spread Simulation: 0.05%?

            if side == 'bid' and low <= price: # Buy Limit
                 self.update_position(market, order['volume'], price, 'bid')
                 curr_vol = order['volume']
                 fee = (price * curr_vol) * 0.0005
                 total = (price * curr_vol) + fee
                 self.balance -= total # Deduct Balance immediately? PaperAccount usually deducts on order *creation* (locked).
                 # Wait, PaperAccount logic: buy_limit locks balance?
                 # check trade.py: buy_limit -> locks balance.
                 # So here we just finalize.

                 self.log_trade(market, 'bid', price, order['volume'], total)
                 filled = True

            elif side == 'ask' and high >= price: # Sell Limit
                 curr_vol = order['volume']
                 earn = price * curr_vol
                 fee = (price * curr_vol) * 0.0005
                 total_earn = earn - fee

                 self.balance += total_earn
                 self.update_position(market, order['volume'], price, 'ask') # Updates position volume (removes it)

                 self.log_trade(market, 'ask', price, order['volume'], total_earn)
                 filled = True

            if filled:
                filled_orders.append(order)
            else:
                remaining_orders.append(order)

        self.orders = remaining_orders
        return filled_orders

    def log_trade(self, market, side, price, volume, total):
        self.trades_log.append({
            'timestamp': str(datetime.now()), # Mock time? Ideally passed in.
            'market': market,
            'side': side,
            'price': price,
            'volume': volume,
            'total': total
        })

    def save(self): pass

# --- DB & Data Fetching ---
class BacktestDB:
    def __init__(self, db_file="backtest_cache.db"):
        self.db_file = db_file
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS candles
                     (market TEXT,
                      timestamp_kst TEXT,
                      trade_price REAL,
                      high_price REAL,
                      low_price REAL,
                      opening_price REAL,
                      candle_acc_trade_volume REAL,
                      PRIMARY KEY (market, timestamp_kst))''')
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON candles (timestamp_kst)")
        conn.commit()
        conn.close()

    def save_bulk(self, candles):
        if not candles: return
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        data = []
        for cdl in candles:
            data.append((
                cdl['market'],
                cdl['candle_date_time_kst'],
                cdl['trade_price'],
                cdl['high_price'],
                cdl['low_price'],
                cdl['opening_price'],
                cdl['candle_acc_trade_volume']
            ))
        c.executemany("INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)", data)
        conn.commit()
        conn.close()

    def get_entries_range(self, market, start_ts, end_ts):
        """Get all 1-minute candles in range [start_ts, end_ts] inclusive (Strings)"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM candles WHERE market=? AND timestamp_kst >= ? AND timestamp_kst <= ? ORDER BY timestamp_kst ASC", (market, start_ts, end_ts))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_latest_timestamp(self, market):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("SELECT MAX(timestamp_kst) FROM candles WHERE market=?", (market,))
        res = c.fetchone()
        conn.close()
        return res[0] if res else None

import urllib.request
from urllib.parse import urlencode

# ... (Previous imports)

def fetch_minute_candles(market, to_datetime, count=200):
    url = "https://api.upbit.com/v1/candles/minutes/1"
    to_str = to_datetime.strftime("%Y-%m-%dT%H:%M:%S") + "+09:00"
    params = {'market': market, 'to': to_str, 'count': count}

    query_string = urlencode(params)
    full_url = f"{url}?{query_string}"

    req = urllib.request.Request(full_url)
    req.add_header("User-Agent", "Mozilla/5.0")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data

    except urllib.error.HTTPError as e:
        if e.code == 429:
             time.sleep(1)
             return []
        print(f"HTTP Error {e.code} for {market}: {e}")
        return []
    except Exception as e:
        print(f"Error fetching {market}: {e}")
        return []

def sync_data(market, days=7):
    db = BacktestDB()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    print(f"Syncing {market} for {days} days...")

    latest = db.get_latest_timestamp(market)

    # Simple logic: If no data or gap at end, fetch backwards from Now until hits Cache or Start
    curr = end_dt
    retries = 0

    while curr > start_dt:
        if latest and curr.strftime("%Y-%m-%dT%H:%M:%S") < latest:
            print("  Hit cached range.")
            break

        print(f"  Fetching from {curr}")
        batch = fetch_minute_candles(market, curr, 200)
        if not batch:
            retries += 1
            if retries > 3:
                print(f"  Max retries reached for {market} at {curr}. Skipping fetch.")
                break
            time.sleep(1)
            continue

        retries = 0 # Reset on success

        db.save_bulk(batch)
        last_ts = batch[-1]['candle_date_time_kst']
        last_dt = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%S")
        curr = last_dt - timedelta(minutes=1)
        time.sleep(0.12)

    return db


# --- Resampler Logic ---
def build_candles_from_minutes(minute_data, final_curr_price, timeframe_min):
    """
    Groups ordered 1-minute data into candles of `timeframe_min`.
    minute_data: list of dicts (Oldest -> Newest)
    Returns: list of dicts (Newest -> Oldest) as expected by Bot
    """
    if not minute_data: return []

    output = []

    # Grouping key?
    # Just simple chunking relative to first? No, alignment issues.
    # Align to clock?
    # 00, 15, 30, 45 for 15m.
    # 09:00 alignment for Daily.

    # Rough Simulation:
    # We iterate minutes.

    # We need REVERSE order output (Newest First).

    # Just construct ONE candle if we are simulating the "live" moment?
    # Bots usually need History (e.g. 20 candles).
    # So we need to reconstruct the last N candles.

    # Reverse input to work Newest -> Oldest?
    # minutes: [m1, m2, m3 ... mLatest]

    # Let's iterate backwards.
    minutes = minute_data[::-1] # Newest -> Oldest

    current_bucket = []

    # Daily (1440m) needs 09:00 KST alignment.
    # 15m needs HH:00, HH:15...

    for m in minutes:
        dt = datetime.strptime(m['timestamp_kst'], "%Y-%m-%dT%H:%M:%S")

        # Determine Bucket Key
        if timeframe_min == 1440:
            # Daily: 09:00 boundary.
            # If time < 09:00, it belongs to "Yesterday's" trading day (which ends today 09:00)
            # Upbit date logic:
            # Shift -9 hours? No.
            # Just use date() logic shifted by 9H.
            # (dt - 9h).date() is the unique key.
            key = (dt - timedelta(hours=9)).date()
        elif timeframe_min == 60:
             key = dt.replace(minute=0, second=0)
        elif timeframe_min == 15:
             minute_bucket = (dt.minute // 15) * 15
             key = dt.replace(minute=minute_bucket, second=0)
        else:
             key = dt

        # Check if we moved to new bucket
        if output and output[-1]['key'] != key:
             # This m belongs to a older bucket.
             pass

        # Wait, efficient way:
        # We need a list of consolidated candles.

        # Helper to finalize bucket
        pass

    # Pandas is far better for this. But native python:
    # Re-sort to chronological [Old -> New]
    sorted_mins = sorted(minute_data, key=lambda x: x['timestamp_kst'])

    candles = []
    current_c = None
    last_key = None

    for m in sorted_mins:
        dt = datetime.strptime(m['timestamp_kst'], "%Y-%m-%dT%H:%M:%S")

        if timeframe_min == 1440:
             # Daily open at 09:00 KST. Close at 08:59:59 next day.
             # Bucket ID: The Date the candle OPENS on?
             # If dt < 09:00, it belongs to Previous Day (starts Prev 09:00).
             adj = dt - timedelta(hours=9)
             key = adj.date()
        else:
             # 15m / 60m
             total_mins = (dt.hour * 60) + dt.minute
             bucket_idx = total_mins // timeframe_min
             key = f"{dt.date()}-{bucket_idx}"

        if last_key != key:
            if current_c:
                 candles.append(current_c)

            # New Candle
            current_c = {
                'candle_date_time_kst': m['timestamp_kst'],  # Start time approximation
                'opening_price': m['opening_price'],
                'high_price': m['high_price'],
                'low_price': m['low_price'],
                'trade_price': m['trade_price'],
                'candle_acc_trade_volume': m['candle_acc_trade_volume']
            }
            last_key = key
        else:
            # Update Candle
            current_c['high_price'] = max(current_c['high_price'], m['high_price'])
            current_c['low_price'] = min(current_c['low_price'], m['low_price'])
            current_c['trade_price'] = m['trade_price'] # Close is latest
            current_c['candle_acc_trade_volume'] += m['candle_acc_trade_volume']

    if current_c:
        candles.append(current_c)

    # Return Newest -> Oldest
    return candles[::-1]


def run_backtest(days=5, coin_selection_mode='dynamic', strategy_mode='dynamic'):
    # selection_mode: 'dynamic' (volume), 'bluechip' (BTC, ETH, etc.)
    # strategy_mode: 'dynamic' (ADX based), 'vb' (Always trend), 'bb' (Always range)

    # 0. Global Reset for simulation cleanliness
    global LAST_TRADE_TIME
    LAST_TRADE_TIME.clear()

    db = BacktestDB()
    now = datetime.now()
    sync_days = max(days + 20, 30)

    cs = CoinSelector()

    if coin_selection_mode == 'bluechip':
        candidates = ['KRW-BTC', 'KRW-ETH']
    else:
        # volume based
        candidate_tickers = cs.get_market_candidates(limit=2)
        candidates = ['KRW-BTC', 'KRW-ETH']

    for m in candidates:
        sync_data(m, sync_days)

    account = BacktestAccount(BACKTEST_INITIAL_BALANCE)

    # Simulation Range
    start_dt = now - timedelta(days=days)
    start_ts = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_ts = now.strftime("%Y-%m-%d %H:%M:%S")

    # Timeline
    conn = sqlite3.connect("backtest_cache.db")
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT timestamp_kst FROM candles WHERE timestamp_kst >= ? ORDER BY timestamp_kst ASC", (start_ts,))
    timeline = [r[0] for r in cur.fetchall()]
    conn.close()

    if not timeline:
        print("No data in timeline.")
        return {}

    print(f"\n--- Simulation: {coin_selection_mode} / {strategy_mode} ({len(timeline)} steps) ---")

    # Pre-load ALL candidates
    market_data = {}
    for m in candidates:
        rows = db.get_entries_range(m, (start_dt - timedelta(days=20)).strftime("%Y-%m-%d %H:%M:%S"), end_ts)
        market_data[m] = {row['timestamp_kst']: row for row in rows}

    history_buffers = {m: [] for m in candidates}
    bots = {}
    target_markets = []
    last_rebalance_time = None

    step = 0
    total = len(timeline)

    for ts in timeline:
        step += 1
        current_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")

        # --- Hourly Rebalancing ---
        if last_rebalance_time is None or (current_dt - last_rebalance_time).total_seconds() >= 3600:
            if coin_selection_mode == 'bluechip':
                target_markets = candidates
            else:
                # Mock current prices for selection
                current_prices = {m: market_data[m][ts]['trade_price'] for m in candidates if ts in market_data[m]}
                # For simplicity in optimization, we use initial selection or refresh if possible
                if not target_markets:
                    target_markets = cs.select_targets(top_n=20, select_count=5)
                    print(f"\n[BACKTEST-DEBUG] Targets Selected: {target_markets}")

            last_rebalance_time = current_dt

        # 1. Update Market Prices
        current_tick_map = {}
        for m in candidates:
            if ts in market_data[m]:
                candle = market_data[m][ts]
                current_tick_map[m] = candle
                history_buffers[m].append(candle)
                if len(history_buffers[m]) > 10000: history_buffers[m] = history_buffers[m][-10000:]

        # DEBUG: Write state to file
        with open("debug_log.txt", "a") as f:
            f.write(f"[{ts}] Bal: {account.balance:.0f} | Pos: {list(account.positions.keys())}\n")

        # 2. Orders
        account.check_orders(current_tick_map)

        # 3. Bots Run
        if current_dt.minute % 5 == 0:
            active_markets = set(target_markets)
            for m, pos in account.positions.items():
                if pos.get('volume', 0) > 1e-8: active_markets.add(m)

            for m in active_markets:
                if m not in current_tick_map: continue

                # Bot Selection
                if m not in bots:
                    if strategy_mode == 'vb': bots[m] = VolatilityBreakoutBot(account, m)
                    elif strategy_mode == 'bb': bots[m] = BollingerBandBot(account, m)
                    else: bots[m] = DynamicSwitchBot(account, m)

                c_price = current_tick_map[m]['trade_price']
                hist = history_buffers[m]
                daily_cf = build_candles_from_minutes(hist, c_price, 1440)[:200]
                min15_cf = build_candles_from_minutes(hist, c_price, 15)[:60]

                allow_buy = m in target_markets

                if isinstance(bots[m], DynamicSwitchBot):
                    bots[m].run(c_price, {'daily': daily_cf, '15m': min15_cf}, current_time=current_dt, allow_buy=allow_buy)
                elif isinstance(bots[m], VolatilityBreakoutBot):
                    bots[m].run(c_price, daily_cf, current_time=current_dt, allow_buy=allow_buy)
                elif isinstance(bots[m], BollingerBandBot):
                    # BB needs 15m
                    bots[m].run(c_price, min15_cf, current_time=current_dt, allow_buy=allow_buy)

    import logging
    logging.info(f"\n[Final Breakdown]")
    logging.info(f" Cash: {account.balance:,.0f} KRW")

    pos_val = 0
    for m, p in account.positions.items():
        if p.get('volume', 0) > 1e-8:
            vol = p['volume']
            price = current_tick_map.get(m, {}).get('trade_price', 0)
            if price == 0 and m in history_buffers and history_buffers[m]:
                price = history_buffers[m][-1]['trade_price']

            val = vol * price
            pos_val += val
            logging.info(f"   - {m}: {vol:.4f} * {price:,.0f} = {val:,.0f} KRW")

    final_val = account.balance + pos_val
    pnl_pct = ((final_val - BACKTEST_INITIAL_BALANCE)/BACKTEST_INITIAL_BALANCE)*100
    logging.info(f" Total Position Value: {pos_val:,.0f} KRW")
    logging.info(f"\n[{coin_selection_mode}/{strategy_mode}] Final Account Value: {final_val:,.0f} | PnL: {pnl_pct:.2f}% | Trades: {len(account.trades_log)}")
    for t in account.trades_log:
        logging.info(f"  {t}")

    return {
        'selection': coin_selection_mode,
        'strategy': strategy_mode,
        'final_balance': final_val,
        'pnl_pct': pnl_pct,
        'trades': len(account.trades_log)
    }

if __name__ == "__main__":
    run_backtest(days=7, coin_selection_mode='dynamic', strategy_mode='dynamic')
