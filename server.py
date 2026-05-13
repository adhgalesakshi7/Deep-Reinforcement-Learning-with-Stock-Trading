import json
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "web"
CACHE_DIR = PROJECT_DIR / "live_cache"
CACHE_DIR.mkdir(exist_ok=True)

HOST = "127.0.0.1"
PORT = 8765

INDIAN_STOCKS = OrderedDict(
    [
        ("Reliance Industries", "RELIANCE.NS"),
        ("Tata Consultancy Services", "TCS.NS"),
        ("Infosys", "INFY.NS"),
        ("HDFC Bank", "HDFCBANK.NS"),
        ("ICICI Bank", "ICICIBANK.NS"),
        ("State Bank of India", "SBIN.NS"),
        ("ITC", "ITC.NS"),
        ("Larsen & Toubro", "LT.NS"),
        ("Bharti Airtel", "BHARTIARTL.NS"),
        ("Asian Paints", "ASIANPAINT.NS"),
    ]
)

INDICES = OrderedDict(
    [
        ("Nifty 50", "^NSEI"),
        ("Sensex", "^BSESN"),
        ("Bank Nifty", "^NSEBANK"),
    ]
)

INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "1d": 24 * 60}
ALLOWED_INTERVALS = {"5m", "15m", "1h", "1d"}
ALLOWED_RANGES = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"}
SIGNAL_LOOKBACK = {"5m": "1mo", "15m": "3mo", "1h": "6mo", "1d": "2y"}


def cache_path_for_symbol(symbol: str) -> Path:
    safe_name = symbol.replace("^", "IDX_").replace(".", "_")
    return CACHE_DIR / f"{safe_name}.json"


def market_status() -> str:
    ist = ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return "Closed for weekend"
    open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if open_time <= now <= close_time:
        return "Market open"
    return "Market closed"


def read_cache(symbol: str) -> list[dict]:
    cache_path = cache_path_for_symbol(symbol)
    if not cache_path.exists():
        return []
    return json.loads(cache_path.read_text(encoding="utf-8"))


def write_cache(symbol: str, rows: list[dict]) -> None:
    cache_path = cache_path_for_symbol(symbol)
    cache_path.write_text(json.dumps(rows), encoding="utf-8")


def yahoo_chart_request(symbol: str, chart_range: str, interval: str) -> dict:
    encoded_symbol = quote(symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?interval={interval}&range={chart_range}&includePrePost=false&events=div%2Csplits"
    )
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_rows(payload: dict) -> list[dict]:
    result = payload.get("chart", {}).get("result", [])
    if not result:
        return []

    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quotes = (chart.get("indicators") or {}).get("quote") or []
    if not quotes:
        return []

    quote_data = quotes[0]
    rows = []
    for idx, ts in enumerate(timestamps):
        open_price = quote_data.get("open", [None])[idx]
        high_price = quote_data.get("high", [None])[idx]
        low_price = quote_data.get("low", [None])[idx]
        close_price = quote_data.get("close", [None])[idx]
        volume = quote_data.get("volume", [None])[idx]

        if None in (open_price, high_price, low_price, close_price):
            continue

        timestamp = datetime.fromtimestamp(ts, ZoneInfo("Asia/Kolkata")).isoformat()
        rows.append(
            {
                "timestamp": timestamp,
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": int(volume or 0),
            }
        )
    return rows


def fetch_history(symbol: str, chart_range: str, interval: str) -> list[dict]:
    try:
        payload = yahoo_chart_request(symbol, chart_range, interval)
        rows = normalize_rows(payload)
        if rows:
            write_cache(symbol, rows)
            return rows
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        pass

    return read_cache(symbol)


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["sma20"] = data["close"].rolling(20).mean()
    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
    data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["returns"] = data["close"].pct_change()
    data["volatility20"] = data["returns"].rolling(20).std() * np.sqrt(252)

    delta = data["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))

    data["ema12"] = data["close"].ewm(span=12, adjust=False).mean()
    data["ema26"] = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema12"] - data["ema26"]
    data["signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["signal"]
    data["volume_change"] = data["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return data.dropna().reset_index(drop=True)


def predict_signal(rows: list[dict], interval: str) -> dict:
    frame = rows_to_frame(rows)
    enriched = add_indicators(frame)
    if len(enriched) < 80:
        latest_price = float(frame["close"].iloc[-1]) if not frame.empty else 0.0
        return {
            "signal": "HOLD",
            "confidence": 0.50,
            "predictedReturnPct": 0.0,
            "predictedPrice": latest_price,
            "accuracyPct": 0.0,
            "reasoning": ["Not enough rows yet for a stable forecast."],
        }

    features = pd.DataFrame(
        {
            "ret_1": enriched["close"].pct_change(1),
            "ret_3": enriched["close"].pct_change(3),
            "ret_5": enriched["close"].pct_change(5),
            "price_vs_sma20": enriched["close"] / enriched["sma20"] - 1,
            "price_vs_ema50": enriched["close"] / enriched["ema50"] - 1,
            "rsi_scaled": enriched["rsi"] / 100.0,
            "macd_gap": enriched["macd_hist"],
            "range_pct": (enriched["high"] - enriched["low"]) / enriched["close"],
            "volume_change": enriched["volume_change"],
            "volatility_20": enriched["volatility20"].bfill(),
        }
    )
    target = enriched["close"].shift(-1) / enriched["close"] - 1
    dataset = features.join(target.rename("target")).replace([np.inf, -np.inf], np.nan).dropna()
    if len(dataset) < 40:
        latest_price = float(enriched["close"].iloc[-1])
        return {
            "signal": "HOLD",
            "confidence": 0.50,
            "predictedReturnPct": 0.0,
            "predictedPrice": latest_price,
            "accuracyPct": 0.0,
            "reasoning": ["Training window is too small for a reliable signal."],
        }

    split_index = max(30, int(len(dataset) * 0.75))
    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    x_train = train.drop(columns="target").to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=float)
    x_test = test.drop(columns="target").to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=float)
    x_last = dataset.drop(columns="target").iloc[-1].to_numpy(dtype=float)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0

    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std
    x_last_scaled = (x_last - mean) / std

    coeff = np.linalg.lstsq(np.c_[np.ones(len(x_train_scaled)), x_train_scaled], y_train, rcond=None)[0]

    def predict(array: np.ndarray) -> np.ndarray:
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return np.c_[np.ones(len(array)), array] @ coeff

    latest = enriched.iloc[-1]
    predicted_return = float(predict(x_last_scaled)[0])
    predicted_price = float(latest["close"] * (1 + predicted_return))
    test_predictions = predict(x_test_scaled)
    accuracy = float(np.mean(np.sign(test_predictions) == np.sign(y_test))) if len(y_test) else 0.0

    score = 0.0
    score += 0.30 if latest["close"] > latest["ema20"] else -0.30
    score += 0.20 if latest["ema20"] > latest["ema50"] else -0.20
    score += 0.25 if latest["macd"] > latest["signal"] else -0.25
    score += 0.15 if latest["rsi"] < 65 else -0.15
    score += 0.10 if predicted_return > 0 else -0.10

    move_threshold = 0.006 if interval in {"5m", "15m", "1h"} else 0.012
    if predicted_return >= move_threshold and score > 0.15:
        signal = "BUY"
    elif predicted_return <= -move_threshold and score < -0.15:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = max(
        0.45,
        min(
            0.95,
            0.52 + (accuracy * 0.25) + min(abs(predicted_return) / (move_threshold * 3), 1.0) * 0.18,
        ),
    )

    return {
        "signal": signal,
        "confidence": round(confidence * 100, 1),
        "predictedReturnPct": round(predicted_return * 100, 2),
        "predictedPrice": round(predicted_price, 2),
        "accuracyPct": round(accuracy * 100, 1),
        "reasoning": [
            f"Forecasted next move: {predicted_return * 100:.2f}%.",
            f"RSI: {latest['rsi']:.1f}, MACD spread: {latest['macd_hist']:.3f}.",
            f"Recent directional accuracy: {accuracy * 100:.1f}%.",
        ],
    }


def analyse_symbol(symbol: str, chart_range: str, interval: str) -> dict:
    rows = fetch_history(symbol, chart_range, interval)
    if not rows:
        return {"error": "No data available for this symbol."}

    signal_rows = fetch_history(symbol, SIGNAL_LOOKBACK.get(interval, chart_range), interval)
    signal = predict_signal(signal_rows if signal_rows else rows, interval)
    latest = rows[-1]
    previous_close = rows[-2]["close"] if len(rows) > 1 else latest["close"]
    live_change_pct = ((latest["close"] / previous_close) - 1) * 100 if previous_close else 0.0
    next_ts = (
        pd.to_datetime(latest["timestamp"])
        + timedelta(minutes=INTERVAL_MINUTES.get(interval, 24 * 60))
    ).isoformat()

    return {
        "symbol": symbol,
        "marketStatus": market_status(),
        "lastUpdated": latest["timestamp"],
        "lastPrice": round(latest["close"], 2),
        "liveChangePct": round(live_change_pct, 2),
        "predictedPoint": {"timestamp": next_ts, "price": signal["predictedPrice"]},
        "signal": signal,
        "history": rows,
    }


def market_overview(symbols: list[str], chart_range: str, interval: str) -> dict:
    watchlist = []
    for symbol in symbols:
        summary = analyse_symbol(symbol, chart_range, interval)
        if "error" in summary:
            continue
        watchlist.append(
            {
                "symbol": symbol,
                "lastPrice": summary["lastPrice"],
                "liveChangePct": summary["liveChangePct"],
                "predictedReturnPct": summary["signal"]["predictedReturnPct"],
                "signal": summary["signal"]["signal"],
                "confidence": summary["signal"]["confidence"],
            }
        )

    index_rows = []
    for name, symbol in INDICES.items():
        rows = fetch_history(symbol, "1mo", "1d")
        if not rows:
            continue
        last_price = rows[-1]["close"]
        prev_price = rows[-2]["close"] if len(rows) > 1 else last_price
        index_rows.append(
            {
                "name": name,
                "symbol": symbol,
                "value": round(last_price, 2),
                "movePct": round(((last_price / prev_price) - 1) * 100, 2),
            }
        )

    return {"indices": index_rows, "watchlist": watchlist}


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            self.send_json(
                {
                    "stocks": [{"name": name, "symbol": symbol} for name, symbol in INDIAN_STOCKS.items()],
                    "indices": [{"name": name, "symbol": symbol} for name, symbol in INDICES.items()],
                    "ranges": sorted(ALLOWED_RANGES),
                    "intervals": sorted(ALLOWED_INTERVALS, key=lambda item: INTERVAL_MINUTES[item]),
                    "marketStatus": market_status(),
                }
            )
            return

        if parsed.path == "/api/analyze":
            params = parse_qs(parsed.query)
            symbol = params.get("symbol", ["RELIANCE.NS"])[0]
            chart_range = params.get("range", ["1mo"])[0]
            interval = params.get("interval", ["1h"])[0]

            if interval not in ALLOWED_INTERVALS or chart_range not in ALLOWED_RANGES:
                self.send_json({"error": "Unsupported range or interval."}, status=400)
                return

            payload = analyse_symbol(symbol, chart_range, interval)
            if "error" in payload:
                self.send_json(payload, status=502)
                return
            self.send_json(payload)
            return

        if parsed.path == "/api/overview":
            params = parse_qs(parsed.query)
            chart_range = params.get("range", ["1mo"])[0]
            interval = params.get("interval", ["1h"])[0]
            symbols = params.get("symbols", [])
            if not symbols:
                symbols = list(INDIAN_STOCKS.values())[:5]

            payload = market_overview(symbols, chart_range, interval)
            self.send_json(payload)
            return

        if parsed.path in {"/", ""}:
            self.path = "/index.html"

        super().do_GET()


def main():
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Serving Indian stock dashboard at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
