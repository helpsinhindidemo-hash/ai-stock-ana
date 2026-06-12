import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import yfinance as yf
from groq import Groq

logger = logging.getLogger("stock_analyzer")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)

GROQ_API_KEY = os.getenv("GROQAPIKEY", "").strip()
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

APP_VERSION = "stock_analyzer_v10_pro"

# Global caching variables for prices
_batch_history_df = None
_last_batch_fetch_time = None
BATCH_CACHE_EXPIRY = 600  # 10 minutes cache for daily prices during market hours

# Persistent metadata cache
METADATA_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata_cache.json")
_metadata_cache = {}

def load_metadata_cache():
    global _metadata_cache
    if os.path.exists(METADATA_CACHE_FILE):
        try:
            with open(METADATA_CACHE_FILE, "r") as f:
                _metadata_cache = json.load(f)
            logger.info(f"Loaded {len(_metadata_cache)} cached stock metadata items.")
        except Exception as e:
            logger.error(f"Error loading metadata cache: {e}")

def save_metadata_cache():
    try:
        with open(METADATA_CACHE_FILE, "w") as f:
            json.dump(_metadata_cache, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving metadata cache: {e}")

# Load metadata cache immediately on startup
load_metadata_cache()

STOCKS_TO_ANALYZE = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "MARICO.NS",
    "SBIN.NS",
    "WIPRO.NS",
    "BAJFINANCE.NS",
]

SHORTLIST_UNIVERSE = [
    "ABB.NS",
    "ADANIENT.NS",
    "ADANIGREEN.NS",
    "ADANIPORTS.NS",
    "AMBUJACEM.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DLF.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",
    "GAIL.NS",
    "GRASIM.NS",
    "HAL.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDHOTEL.NS",
    "INDIGO.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "IOC.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "MAXHEALTH.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "PIDILITIND.NS",
    "PFC.NS",
    "POWERGRID.NS",
    "RECLTD.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHRIRAMFIN.NS",
    "SIEMENS.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAPOWER.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TORNTPHARM.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "UNITDSPR.NS",
    "VBL.NS",
    "VEDL.NS",
    "WIPRO.NS",
    "ZYDUSLIFE.NS",
]

MANUAL_METADATA: Dict[str, Dict[str, Any]] = {
    "ASIANPAINT.NS": {
        "name": "Asian Paints Limited",
        "sector": "Consumer Goods",
        "industry": "Specialty Chemicals / Paints",
        "pe_fallback": 59.02,
        "beta_fallback": 0.30,
        "market_cap_cr_fallback": 257708,
    },
    "ETERNAL.NS": {
        "name": "Eternal Limited",
        "sector": "Consumer Services",
        "industry": "Online Services / Internet Commerce",
        "pe_fallback": "N/A",
        "beta_fallback": "N/A",
        "market_cap_cr_fallback": 247483,
    },
    "TCS.NS": {
        "name": "Tata Consultancy Services Limited",
        "sector": "Technology",
        "industry": "IT Services / Consulting",
        "pe_fallback": 24.00,
        "beta_fallback": 0.45,
        "market_cap_cr_fallback": 795000,
    },
    "WIPRO.NS": {
        "name": "Wipro Limited",
        "sector": "Technology",
        "industry": "IT Services / Consulting",
        "pe_fallback": 19.00,
        "beta_fallback": 0.60,
        "market_cap_cr_fallback": 205000,
    },
    "ITC.NS": {
        "name": "ITC Limited",
        "sector": "Consumer Goods",
        "industry": "FMCG / Conglomerate",
        "pe_fallback": 26.00,
        "beta_fallback": 0.55,
        "market_cap_cr_fallback": 340000,
    },
    "ADANIENT.NS": {
        "name": "Adani Enterprises Limited",
        "sector": "Diversified",
        "industry": "Trading / Infrastructure / Incubation",
        "pe_fallback": 68.00,
        "beta_fallback": 1.30,
        "market_cap_cr_fallback": 350000,
    },
    "RELIANCE.NS": {
        "name": "Reliance Industries Limited",
        "sector": "Energy / Conglomerate",
        "industry": "Oil, Gas, Retail, Telecom",
        "pe_fallback": 22.00,
        "beta_fallback": 0.80,
        "market_cap_cr_fallback": 1650000,
    },
    "HDFCBANK.NS": {
        "name": "HDFC Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Banking",
        "pe_fallback": 17.00,
        "beta_fallback": 0.65,
        "market_cap_cr_fallback": 1180000,
    },
    "SBIN.NS": {
        "name": "State Bank of India",
        "sector": "Financial Services",
        "industry": "Public Sector Banking",
        "pe_fallback": 10.00,
        "beta_fallback": 1.10,
        "market_cap_cr_fallback": 690000,
    },
    "BAJFINANCE.NS": {
        "name": "Bajaj Finance Limited",
        "sector": "Financial Services",
        "industry": "Non-Banking Financial Company",
        "pe_fallback": 28.00,
        "beta_fallback": 1.10,
        "market_cap_cr_fallback": 430000,
    },
    "INFY.NS": {
        "name": "Infosys Limited",
        "sector": "Technology",
        "industry": "IT Services / Consulting",
        "pe_fallback": 21.00,
        "beta_fallback": 0.50,
        "market_cap_cr_fallback": 595000,
    },
    "TATAMOTORS.NS": {
        "name": "Tata Motors Limited",
        "sector": "Consumer Cyclical",
        "industry": "Automobiles",
        "pe_fallback": 8.00,
        "beta_fallback": 1.50,
        "market_cap_cr_fallback": 250000,
    },
    "CIPLA.NS": {
        "name": "Cipla Limited",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers - Specialty / Generic",
        "pe_fallback": 24.00,
        "beta_fallback": 0.75,
        "market_cap_cr_fallback": 113000,
    },
    "NTPC.NS": {
        "name": "NTPC Limited",
        "sector": "Utilities",
        "industry": "Power Generation",
        "pe_fallback": 17.00,
        "beta_fallback": 0.95,
        "market_cap_cr_fallback": 351000,
    },
    "ICICIBANK.NS": {
        "name": "ICICI Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Banking",
        "pe_fallback": 18.00,
        "beta_fallback": 0.90,
        "market_cap_cr_fallback": 850000,
    },
    "KOTAKBANK.NS": {
        "name": "Kotak Mahindra Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Banking",
        "pe_fallback": 19.00,
        "beta_fallback": 0.70,
        "market_cap_cr_fallback": 370000,
    },
    "LT.NS": {
        "name": "Larsen & Toubro Limited",
        "sector": "Industrials",
        "industry": "Engineering / Construction",
        "pe_fallback": 32.00,
        "beta_fallback": 1.10,
        "market_cap_cr_fallback": 480000,
    },
    "SUNPHARMA.NS": {
        "name": "Sun Pharmaceutical Industries",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
        "pe_fallback": 33.00,
        "beta_fallback": 0.65,
        "market_cap_cr_fallback": 380000,
    },
    "BHARTIARTL.NS": {
        "name": "Bharti Airtel Limited",
        "sector": "Communication Services",
        "industry": "Telecom Services",
        "pe_fallback": 58.00,
        "beta_fallback": 0.85,
        "market_cap_cr_fallback": 910000,
    },
    "AXISBANK.NS": {
        "name": "Axis Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Banking",
        "pe_fallback": 12.00,
        "beta_fallback": 1.10,
        "market_cap_cr_fallback": 360000,
    },
    "TITAN.NS": {
        "name": "Titan Company Limited",
        "sector": "Consumer Cyclical",
        "industry": "Luxury Goods",
        "pe_fallback": 88.00,
        "beta_fallback": 0.90,
        "market_cap_cr_fallback": 277000,
    },
    "HAL.NS": {
        "name": "Hindustan Aeronautics Limited",
        "sector": "Industrials",
        "industry": "Aerospace / Defense",
        "pe_fallback": 30.00,
        "beta_fallback": 1.20,
        "market_cap_cr_fallback": 330000,
    },
    "HINDUNILVR.NS": {
        "name": "Hindustan Unilever Limited",
        "sector": "Consumer Goods",
        "industry": "FMCG",
        "pe_fallback": 53.00,
        "beta_fallback": 0.40,
        "market_cap_cr_fallback": 516000,
    },
    "MARUTI.NS": {
        "name": "Maruti Suzuki India Limited",
        "sector": "Consumer Cyclical",
        "industry": "Automobiles",
        "pe_fallback": 25.00,
        "beta_fallback": 0.75,
        "market_cap_cr_fallback": 380000,
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.upper().strip()
    if not cleaned:
        return ""
    if cleaned == "M" or cleaned == "M_M":
        cleaned = "M&M"
    if not cleaned.endswith((".NS", ".BO")):
        cleaned = f"{cleaned}.NS"
    return cleaned


def safe_round(value: Any, digits: int = 2, default: Any = 0) -> Any:
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        if pd.isna(value):
            return default
        return round(float(value), digits)
    except Exception:
        return default


def safe_value(value: Any, default: Any = "N/A") -> Any:
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        if pd.isna(value):
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return value
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in [None, "N/A", ""]:
            return default
        if hasattr(value, "item"):
            value = value.item()
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clamp_number(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def normalize_history_columns(hist: pd.DataFrame) -> pd.DataFrame:
    try:
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = (
                hist.columns.get_level_values(0)
                if hist.columns.nlevels >= 2
                else [col[0] if isinstance(col, tuple) else col for col in hist.columns]
            )
        hist.columns = [str(col).strip() for col in hist.columns]
    except Exception:
        pass
    return hist


def get_market_cap_cr(raw_market_cap: Any) -> Any:
    try:
        if raw_market_cap in [None, "N/A", 0]:
            return "N/A"
        return safe_round(float(raw_market_cap) / 10000000, 2)
    except Exception:
        return "N/A"


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = str(item).strip().upper()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def extract_ticker_df(batch_df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    if batch_df is None or batch_df.empty:
        return None
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            # Check level 1 first (ticker is usually in level 1, field is in level 0)
            if symbol in batch_df.columns.get_level_values(1):
                ticker_df = batch_df.xs(symbol, axis=1, level=1).copy()
                ticker_df.columns = [str(col).strip() for col in ticker_df.columns]
                return ticker_df
            # If level 0 contains the symbol
            elif symbol in batch_df.columns.get_level_values(0):
                ticker_df = batch_df.xs(symbol, axis=1, level=0).copy()
                ticker_df.columns = [str(col).strip() for col in ticker_df.columns]
                return ticker_df
        else:
            # Not a MultiIndex (single ticker flat columns)
            ticker_df = batch_df.copy()
            ticker_df.columns = [str(col).strip() for col in ticker_df.columns]
            return ticker_df
    except Exception as e:
        logger.error(f"Error extracting DataFrame for {symbol}: {e}")
    return None


def prefetch_all_stocks_data(symbols: List[str]):
    global _batch_history_df, _last_batch_fetch_time
    now = datetime.now()
    if _batch_history_df is not None and _last_batch_fetch_time is not None:
        if (now - _last_batch_fetch_time).total_seconds() < BATCH_CACHE_EXPIRY:
            logger.info("Using cached batch historical data.")
            return

    # Clean the input symbols
    clean_syms = dedupe_preserve_order(symbols)
    if not clean_syms:
        return

    logger.info(f"Downloading batch data for {len(clean_syms)} symbols...")
    try:
        df = yf.download(
            tickers=clean_syms,
            period="12mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="column",
        )
        if df is not None and not df.empty:
            _batch_history_df = df
            _last_batch_fetch_time = now
            logger.info("Batch download completed and cached.")
        else:
            logger.warning("Batch download returned empty DataFrame. Keeping previous cache if any.")
    except Exception as e:
        logger.error(f"Error downloading batch data: {e}. Keeping previous cache if any.")


def try_download_symbol(base_symbol: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    base_symbol = base_symbol.upper().strip()
    candidates: List[str] = (
        [base_symbol]
        if base_symbol.endswith((".NS", ".BO"))
        else [f"{base_symbol}.NS", f"{base_symbol}.BO"]
    )

    for sym in candidates:
        try:
            logger.info(f"Trying symbol: {sym}")
            hist = yf.download(
                tickers=sym,
                period="12mo",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
                group_by="column",
            )
            if hist is not None and not hist.empty:
                hist = normalize_history_columns(hist)
                logger.info(f"Downloaded {len(hist)} rows for {sym}")
                return sym, hist
            logger.warning(f"No data for {sym}")
        except Exception as e:
            logger.error(f"Download error for {sym}: {e}")
    return None, None


def calculate_rsi(close_series: pd.Series, period: int = 14) -> float:
    if len(close_series) < period + 1:
        return 50.0

    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]

    if pd.isna(last_gain) or pd.isna(last_loss):
        return 50.0

    if last_loss == 0 and last_gain > 0:
        return 100.0
    if last_loss == 0:
        return 50.0

    rs = last_gain / last_loss
    return safe_round(100 - (100 / (1 + rs)))


def calculate_macd(close_series: pd.Series) -> Tuple[float, float, float]:
    if len(close_series) < 26:
        return 0.0, 0.0, 0.0

    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    return (
        safe_round(macd.iloc[-1]),
        safe_round(signal.iloc[-1]),
        safe_round(histogram.iloc[-1]),
    )


def calculate_atr(
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
    period: int = 14,
) -> float:
    if len(close_series) < period + 1:
        return 0.0

    prev_close = close_series.shift(1)
    tr = pd.concat(
        [
            high_series - low_series,
            (high_series - prev_close).abs(),
            (low_series - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return safe_round(atr.iloc[-1]) if len(atr.dropna()) else 0.0


def calculate_bollinger(
    close_series: pd.Series,
    period: int = 20,
    num_std: int = 2,
) -> Tuple[float, float, float]:
    if len(close_series) < period:
        return 0.0, 0.0, 0.0

    mid = close_series.rolling(period).mean()
    std = close_series.rolling(period).std()

    lower = mid.iloc[-1] - num_std * std.iloc[-1]
    upper = mid.iloc[-1] + num_std * std.iloc[-1]

    return safe_round(lower), safe_round(mid.iloc[-1]), safe_round(upper)


def calculate_adx(
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
    period: int = 14,
) -> float:
    if len(close_series) < period * 2:
        return 0.0

    try:
        up_move = high_series.diff()
        down_move = -low_series.diff()

        plus_dm = pd.Series(0.0, index=high_series.index)
        minus_dm = pd.Series(0.0, index=high_series.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        prev_close = close_series.shift(1)
        tr = pd.concat(
            [
                high_series - low_series,
                (high_series - prev_close).abs(),
                (low_series - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        smoothed_tr = tr.ewm(alpha=1/period, adjust=False).mean()
        smoothed_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()

        plus_di = 100 * (smoothed_plus_dm / smoothed_tr.replace(0, float("nan")))
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr.replace(0, float("nan")))

        denom = (plus_di + minus_di).replace(0, float("nan"))
        dx = 100 * (plus_di - minus_di).abs() / denom
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        return safe_round(adx.iloc[-1]) if len(adx.dropna()) else 0.0
    except Exception as e:
        logger.warning(f"ADX calc error: {e}")
        return 0.0


def check_multi_timeframe(hist: pd.DataFrame) -> Dict[str, Any]:
    try:
        # Make sure index is DatetimeIndex for resampling
        if not isinstance(hist.index, pd.DatetimeIndex):
            hist = hist.copy()
            hist.index = pd.to_datetime(hist.index)
            
        # Resample to weekly. We group by calendar week ending Friday
        # Use agg to get correct weekly high, low, open, close, volume
        weekly = hist.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if len(weekly) < 20:
            return {
                "weekly_trend": "Neutral",
                "weekly_rsi": 50.0,
                "weekly_close": 0.0,
                "weekly_ma20": 0.0
            }
            
        weekly_close = weekly['Close']
        weekly_ma20 = safe_round(weekly_close.rolling(20).mean().iloc[-1])
        weekly_ma50 = safe_round(weekly_close.rolling(50).mean().iloc[-1]) if len(weekly_close) >= 50 else weekly_ma20
        
        weekly_rsi = calculate_rsi(weekly_close)
        
        last_weekly_close = safe_round(weekly_close.iloc[-1])
        
        if last_weekly_close > weekly_ma20 > weekly_ma50:
            weekly_trend = "Bullish"
        elif last_weekly_close < weekly_ma20 < weekly_ma50:
            weekly_trend = "Bearish"
        else:
            weekly_trend = "Neutral"
            
        return {
            "weekly_trend": weekly_trend,
            "weekly_rsi": weekly_rsi,
            "weekly_close": last_weekly_close,
            "weekly_ma20": weekly_ma20
        }
    except Exception as e:
        logger.warning(f"Error calculating weekly timeframe metrics: {e}")
        return {
            "weekly_trend": "Neutral",
            "weekly_rsi": 50.0,
            "weekly_close": 0.0,
            "weekly_ma20": 0.0
        }


def detect_candlestick_patterns(hist: pd.DataFrame) -> List[str]:
    try:
        if len(hist) < 3:
            return []
            
        # Last 2 rows
        c_prev = hist.iloc[-2]
        c_curr = hist.iloc[-1]
        
        o1, h1, l1, c1 = float(c_prev['Open']), float(c_prev['High']), float(c_prev['Low']), float(c_prev['Close'])
        o2, h2, l2, c2 = float(c_curr['Open']), float(c_curr['High']), float(c_curr['Low']), float(c_curr['Close'])
        
        patterns = []
        
        # Calculate sizes
        body1 = abs(c1 - o1)
        range1 = h1 - l1 if (h1 - l1) > 0 else 0.01
        
        body2 = abs(c2 - o2)
        range2 = h2 - l2 if (h2 - l2) > 0 else 0.01
        
        # Candle 2 upper and lower shadows
        lower_shadow2 = min(o2, c2) - l2
        upper_shadow2 = h2 - max(o2, c2)
        
        # 1. Hammer (Bullish Reversal)
        # Small body, lower shadow at least 2x body, very small upper shadow
        if lower_shadow2 >= 2 * body2 and upper_shadow2 <= 0.2 * range2 and range2 > 0.01:
            patterns.append("Hammer (Bullish)")
            
        # 2. Shooting Star (Bearish Reversal)
        # Small body, upper shadow at least 2x body, very small lower shadow
        if upper_shadow2 >= 2 * body2 and lower_shadow2 <= 0.2 * range2 and range2 > 0.01:
            patterns.append("Shooting Star (Bearish)")
            
        # 3. Bullish Engulfing
        # Previous candle is red, current candle is green
        # Current body completely engulfs previous body
        if c1 < o1 and c2 > o2:
            if c2 >= o1 and o2 <= c1 and body2 > body1:
                patterns.append("Bullish Engulfing")
                
        # 4. Bearish Engulfing
        # Previous candle is green, current candle is red
        # Current body completely engulfs previous body
        if c1 > o1 and c2 < o2:
            if c2 <= o1 and o2 >= c1 and body2 > body1:
                patterns.append("Bearish Engulfing")
                
        # 5. Doji
        if body2 <= 0.1 * range2:
            patterns.append("Doji")
            
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting candlestick patterns: {e}")
        return []





def pick_info_dict(ticker: yf.Ticker) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    try:
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            try:
                info.update(dict(fast_info))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"ticker.fast_info failed: {e}")

    try:
        full_info = ticker.info
        if isinstance(full_info, dict):
            info.update(full_info)
    except Exception as e:
        logger.warning(f"ticker.info failed: {e}")

    return info


def build_metadata(final_symbol: str, info: Dict[str, Any]) -> Dict[str, Any]:
    manual = MANUAL_METADATA.get(final_symbol, {})

    raw_name = safe_value(
        info.get("longName") or info.get("shortName") or info.get("displayName"),
        None,
    )
    name = (
        raw_name
        if raw_name not in [None, "N/A"] and str(raw_name).strip().upper() != final_symbol.upper()
        else manual.get("name", final_symbol)
    )

    raw_sector = safe_value(info.get("sector"), None)
    sector = raw_sector if raw_sector not in [None, "N/A"] else manual.get("sector", "N/A")

    raw_industry = safe_value(info.get("industry"), None)
    industry = (
        raw_industry if raw_industry not in [None, "N/A"] else manual.get("industry", "N/A")
    )

    pe_raw = safe_value(info.get("trailingPE"), None)
    pe_ratio = (
        safe_round(pe_raw) if pe_raw not in [None, "N/A"] else manual.get("pe_fallback", "N/A")
    )

    beta_raw = safe_value(info.get("beta"), None)
    beta = (
        safe_round(beta_raw)
        if beta_raw not in [None, "N/A"]
        else manual.get("beta_fallback", "N/A")
    )

    mc_raw = safe_value(info.get("marketCap"), None)
    market_cap = mc_raw if mc_raw not in [None, "N/A"] else "N/A"
    market_cap_cr = (
        get_market_cap_cr(mc_raw)
        if mc_raw not in [None, "N/A"]
        else manual.get("market_cap_cr_fallback", "N/A")
    )

    def pct_field(key: str) -> Any:
        val = safe_value(info.get(key), None)
        if val in [None, "N/A"]:
            return "N/A"
        try:
            return safe_round(float(val) * 100, 2)
        except Exception:
            return "N/A"

    roe = pct_field("returnOnEquity")
    roa = pct_field("returnOnAssets")
    dividend_yield = pct_field("dividendYield")

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "pe": pe_ratio,
        "beta": beta,
        "market_cap": market_cap,
        "market_cap_cr": market_cap_cr,
        "roe": roe,
        "roce": "N/A",
        "roa": roa,
        "dividend_yield": dividend_yield,
    }


def build_meta() -> Dict[str, Any]:
    return {
        "success": True,
        "version": APP_VERSION,
        "ai_enabled": bool(client),
        "default_stocks_count": len(STOCKS_TO_ANALYZE),
        "shortlist_universe_count": len(dedupe_preserve_order(SHORTLIST_UNIVERSE)),
        "generated_at": utc_now_iso(),
    }


def build_health() -> Dict[str, Any]:
    return {
        "success": True,
        "status": "ok",
        "version": APP_VERSION,
        "ai_enabled": bool(client),
        "generated_at": utc_now_iso(),
    }


def compute_trade_scores(metrics: Dict[str, Any]) -> Dict[str, Any]:
    current = safe_float(metrics.get("price"))
    ma20 = safe_float(metrics.get("ma20"))
    ma50 = safe_float(metrics.get("ma50"))
    ma200_raw = metrics.get("ma200")
    ma200_value = safe_value(ma200_raw, None)
    has_ma200 = ma200_value not in [None, "N/A"]
    ma200 = safe_float(ma200_raw, 0.0)

    rsi = safe_float(metrics.get("rsi"), 50)
    macd = safe_float(metrics.get("macd"))
    macd_signal = safe_float(metrics.get("macd_signal"))
    macd_hist = safe_float(metrics.get("macd_hist"))
    adx = safe_float(metrics.get("adx"))
    vol_ratio = safe_float(metrics.get("vol_ratio"))
    atr = safe_float(metrics.get("atr"))
    high52 = safe_float(metrics.get("high52"), current)
    low52 = safe_float(metrics.get("low52"), current)
    change_pct = safe_float(metrics.get("change_pct"))
    bb_lower = safe_float(metrics.get("bb_lower"))
    bb_upper = safe_float(metrics.get("bb_upper"))

    # INTRADAY SCORE CALCULATION
    intraday_score = 0
    intraday_reasons: List[str] = []

    # 1. Price vs MA20 (Short-term Trend)
    if current > ma20:
        intraday_score += 15
        intraday_reasons.append("Price short-term MA20 ke upar trade kar raha hai, trend positive hai.")
    else:
        intraday_score -= 15
        intraday_reasons.append("Price MA20 ke neeche hai, short-term momentum weak hai.")

    # 2. MACD Momentum
    if macd > macd_signal:
        intraday_score += 15
        if macd_hist > 0:
            intraday_score += 10
            intraday_reasons.append("MACD bullish crossover aur positive histogram momentum ko support kar rahe hain.")
        else:
            intraday_reasons.append("MACD signal line ke upar hai, lekin momentum thoda slow hai.")
    else:
        intraday_score -= 15
        if macd_hist < 0:
            intraday_score -= 10
            intraday_reasons.append("MACD bearish crossover aur negative histogram momentum weakness show kar rahe hain.")

    # 3. RSI Momentum Zone
    if 50 <= rsi <= 68:
        intraday_score += 15
        intraday_reasons.append(f"RSI healthy bullish expansion zone me hai ({rsi}).")
    elif 40 <= rsi < 50:
        intraday_score += 5
        intraday_reasons.append(f"RSI neutral-positive zone me consolidation dikha raha hai ({rsi}).")
    elif rsi > 70:
        intraday_score -= 10
        intraday_reasons.append(f"RSI overbought zone me hai ({rsi}), short-term pullback ka risk hai.")
    elif rsi < 30:
        intraday_score += 10  # Potential reversal / oversold bounce setup
        intraday_reasons.append(f"RSI oversold zone me hai ({rsi}), yahan se bounce setup ban sakta hai.")
    else:
        intraday_score -= 10
        intraday_reasons.append(f"RSI low momentum zone me trade kar raha hai ({rsi}).")

    # 4. Volume Surge confirmation (Institutional interest)
    if vol_ratio >= 1.5:
        intraday_score += 20
        intraday_reasons.append(f"Volume surge exceptional hai ({vol_ratio}x), institutional activity ho sakti hai.")
    elif vol_ratio >= 1.2:
        intraday_score += 10
        intraday_reasons.append(f"Volume participation average se upar hai ({vol_ratio}x).")
    elif vol_ratio < 0.8:
        intraday_score -= 15
        intraday_reasons.append("Volume low hai, move me buyers/sellers ka conversion weak lag raha hai.")

    # 5. ADX (Trend strength)
    if adx >= 25:
        intraday_score += 10
        intraday_reasons.append(f"ADX ({adx}) strong trend strength confirm kar raha hai.")
    elif adx < 15:
        intraday_score -= 10
        intraday_reasons.append(f"ADX weak trend strength ({adx}) show kar raha hai, market sideways ho sakta hai.")

    # 6. Bollinger Bands
    if bb_upper > bb_lower and bb_upper != 0:
        bb_pos = (current - bb_lower) / (bb_upper - bb_lower)
        if bb_pos > 0.90:
            intraday_score += 5
            intraday_reasons.append("Price upper Bollinger Band ke close hai, momentum breakout setup ban sakta hai.")
        elif bb_pos < 0.10:
            intraday_score -= 10
            intraday_reasons.append("Price lower Bollinger Band ke pass hai, weakness badh rahi hai.")

    intraday_score = int(clamp_number(intraday_score, -100, 100))
    intraday_bias = "HOLD"
    if intraday_score >= 35:
        intraday_bias = "BUY"
    elif intraday_score <= -30:
        intraday_bias = "SELL"

    # SWING SCORE CALCULATION
    swing_score = 0
    swing_reasons: List[str] = []

    # 1. Long-term Trend Alignment (moving averages)
    if has_ma200:
        if current > ma20 > ma50 > ma200:
            swing_score += 30
            swing_reasons.append("Perfect bullish alignment (Price > MA20 > MA50 > MA200) active hai. High conviction setup.")
        elif current > ma50 > ma200:
            swing_score += 20
            swing_reasons.append("Stock higher-timeframe uptrend me hai (Price > MA50 > MA200).")
        elif current < ma200:
            swing_score -= 30
            swing_reasons.append("Stock long-term MA200 ke neeche trade kar raha hai, structure completely bearish hai.")
        else:
            swing_score += 5
            swing_reasons.append("Price MA200 ke upar hai, lekin intermediate structure mixed hai.")
    else:
        # Fallback without MA200
        if current > ma20 > ma50:
            swing_score += 20
            swing_reasons.append("Bullish trend structure (Price > MA20 > MA50) bana hua hai.")
        elif current < ma50:
            swing_score -= 20
            swing_reasons.append("Price MA50 ke neeche trade kar raha hai, trend bearish hai.")

    # 2. MACD Trend
    if macd > macd_signal:
        swing_score += 15
        swing_reasons.append("MACD bullish territory me trend continuation show kar raha hai.")
    else:
        swing_score -= 15
        swing_reasons.append("MACD bearish regime me trend weakness show kar raha hai.")

    # 3. RSI for Swing
    if 50 <= rsi <= 65:
        swing_score += 15
        swing_reasons.append("RSI swing entry ke liye optimum bullish zone me hai.")
    elif rsi > 70:
        swing_score -= 10
        swing_reasons.append("RSI overbought level par hai, pullback ka high probability wave ban sakta hai.")
    elif 30 <= rsi < 45:
        swing_score += 10
        swing_reasons.append("RSI accumulation zone me hai, favorable risk-reward entry range hai.")
    else:
        swing_score -= 5

    # 4. ADX for Swing
    if adx >= 20:
        swing_score += 15
        swing_reasons.append(f"ADX ({adx}) sustainable swing trend index show kar raha hai.")
    else:
        swing_score -= 10
        swing_reasons.append("ADX low hai, market sideways consolidate ho sakta hai.")

    # 5. Position in 52-Week Range
    if high52 > low52:
        dist_from_low = (current - low52) / (high52 - low52)
        if dist_from_low > 0.90:
            swing_score += 10
            swing_reasons.append("Stock 52-week high ke kareeb breakout consolidation show kar raha hai.")
        elif dist_from_low < 0.20:
            swing_score -= 10
            swing_reasons.append("Stock 52-week low ke kareeb weak demand zone me trade kar raha hai.")

    swing_score = int(clamp_number(swing_score, -100, 100))
    swing_bias = "HOLD"
    if swing_score >= 35:
        swing_bias = "BUY"
    elif swing_score <= -30:
        swing_bias = "SELL"

    return {
        "intraday_score": intraday_score,
        "intraday_bias": intraday_bias,
        "intraday_reasons": intraday_reasons[:4],
        "swing_score": swing_score,
        "swing_bias": swing_bias,
        "swing_reasons": swing_reasons[:4],
    }


def get_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        raw_symbol = symbol.upper().strip()
        logger.info(f"Incoming symbol for data fetch: {raw_symbol}")

        final_symbol = normalize_symbol(raw_symbol)
        
        hist = None
        if _batch_history_df is not None:
            hist = extract_ticker_df(_batch_history_df, final_symbol)
            if hist is not None and not hist.empty:
                valid_rows = hist.dropna(subset=["Close", "High", "Low"])
                if len(valid_rows) >= 60:
                    logger.info(f"Extracted {final_symbol} from cached batch data.")
                else:
                    hist = None

        if hist is None:
            final_symbol, hist = try_download_symbol(raw_symbol)
            if final_symbol is None or hist is None:
                logger.error(f"All symbol variants failed for {raw_symbol}")
                return None

        hist = normalize_history_columns(hist)

        required_cols = ["Close", "High", "Low", "Volume"]
        for col in required_cols:
            if col not in hist.columns:
                logger.error(f"Missing column {col} for {final_symbol}")
                return None

        hist = hist.dropna(subset=["Close", "High", "Low"])
        if len(hist) < 60:
            logger.error(f"Not enough history rows ({len(hist)}) for {final_symbol}")
            return None

        close_series = hist["Close"].dropna()
        high_series = hist["High"].dropna()
        low_series = hist["Low"].dropna()
        volume_series = hist["Volume"].fillna(0)

        if len(close_series) < 60:
            logger.error(f"Close series too short for {final_symbol}")
            return None

        current = safe_round(close_series.iloc[-1])
        prev = safe_round(close_series.iloc[-2] if len(close_series) >= 2 else current)
        change_pct = safe_round(((current - prev) / prev) * 100, 2, 0) if prev else 0

        ma7 = safe_round(close_series.tail(7).mean())
        ma20 = safe_round(close_series.tail(20).mean())
        ma50 = safe_round(close_series.tail(50).mean())
        ma200 = safe_round(close_series.tail(200).mean()) if len(close_series) >= 200 else "N/A"

        rsi = calculate_rsi(close_series)
        macd, macd_signal, macd_hist = calculate_macd(close_series)
        atr = calculate_atr(high_series, low_series, close_series)
        bb_lower, bb_mid, bb_upper = calculate_bollinger(close_series)
        adx = calculate_adx(high_series, low_series, close_series)

        avg_vol = int(volume_series.tail(20).mean()) if len(volume_series) >= 20 else 0
        vol_today = int(volume_series.iloc[-1]) if len(volume_series) else 0
        vol_ratio = safe_round((vol_today / avg_vol), 2, 0) if avg_vol else 0

        high52 = safe_round(high_series.max())
        low52 = safe_round(low_series.min())

        # Support and Resistance
        recent_highs = high_series.tail(20)
        recent_lows = low_series.tail(20)
        support = safe_round(recent_lows.min())
        resistance = safe_round(recent_highs.max())

        # Metadata Cache check
        metadata = _metadata_cache.get(final_symbol)
        if not metadata:
            ticker = yf.Ticker(final_symbol)
            info = pick_info_dict(ticker)
            metadata = build_metadata(final_symbol, info)
            _metadata_cache[final_symbol] = metadata
            save_metadata_cache()

        mtf_data = check_multi_timeframe(hist)
        patterns = detect_candlestick_patterns(hist)

        base_metrics: Dict[str, Any] = {
            "debug_version": APP_VERSION,
            "generated_at": utc_now_iso(),
            "symbol": final_symbol,
            **metadata,
            "price": current,
            "prev": prev,
            "change_pct": change_pct,
            "ma7": ma7,
            "ma20": ma20,
            "ma50": ma50,
            "ma200": ma200,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "adx": adx,
            "atr": atr,
            "bb_lower": bb_lower,
            "bb_mid": bb_mid,
            "bb_upper": bb_upper,
            "vol_ratio": vol_ratio,
            "high52": high52,
            "low52": low52,
            "support": support,
            "resistance": resistance,
            "weekly_trend": mtf_data.get("weekly_trend", "Neutral"),
            "weekly_rsi": mtf_data.get("weekly_rsi", 50.0),
            "candlestick_patterns": patterns,
            "patterns_str": ", ".join(patterns) if patterns else "None",
        }

        scores = compute_trade_scores(base_metrics)
        base_score = safe_round(
            scores["intraday_score"] * 0.4 + scores["swing_score"] * 0.6,
            0,
            0,
        )

        return {
            **base_metrics,
            **scores,
            "base_score": base_score,
        }
    except Exception as e:
        logger.error(f"Data error for {symbol}: {e}", exc_info=True)
        return None


def rule_signal_from_score(score: float) -> Tuple[str, str]:
    if score >= 35:
        return "BUY", "Strong"
    if score >= 18:
        return "BUY", "Moderate"
    if score <= -30:
        return "SELL", "Strong"
    if score <= -15:
        return "SELL", "Moderate"
    return "HOLD", "Moderate"


def get_recent_news_headlines(symbol: str, limit: int = 5) -> List[str]:
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news
        if not news_items or not isinstance(news_items, list):
            return []
        
        headlines = []
        for item in news_items[:limit]:
            title = item.get("title", "").strip()
            if title:
                headlines.append(title)
        return headlines
    except Exception as e:
        logger.warning(f"Error fetching news for {symbol}: {e}")
        return []


def compute_targets(
    price: float,
    atr: float,
    support: float,
    resistance: float,
    rule_signal: str,
    confidence: int,
    risk_budget: float = 1000.0,
) -> Tuple[float, float, float, int, float, float]:
    effective_atr = atr if atr > 0 else max(1.0, round(price * 0.02, 2))

    if rule_signal == "BUY":
        # Stop loss is placed at support (with 1% buffer) or price - 2*ATR, whichever is closer to price (to keep risk tight), but at least 1.5*ATR
        sl_candidate = support * 0.99
        sl = max(price - 3 * effective_atr, min(price - 1.5 * effective_atr, sl_candidate))
        sl = safe_round(sl)

        # Target is placed at 2.0x risk or recent resistance (if resistance is higher than 1.5x risk)
        risk_dist = price - sl
        if risk_dist <= 0:
            risk_dist = 1.5 * effective_atr
            sl = safe_round(price - risk_dist)

        target_candidate = price + 2 * risk_dist
        if resistance > price + 1.2 * risk_dist:
            target = max(target_candidate, resistance)
        else:
            target = target_candidate
        target = safe_round(target)

        rr_ratio = safe_round((target - price) / risk_dist, 2)

    elif rule_signal == "SELL":
        # Stop loss for short selling
        sl_candidate = resistance * 1.01
        sl = min(price + 3 * effective_atr, max(price + 1.5 * effective_atr, sl_candidate))
        sl = safe_round(sl)

        risk_dist = sl - price
        if risk_dist <= 0:
            risk_dist = 1.5 * effective_atr
            sl = safe_round(price + risk_dist)

        target_candidate = price - 2 * risk_dist
        if support < price - 1.2 * risk_dist:
            target = min(target_candidate, support)
        else:
            target = target_candidate
        target = safe_round(target)

        rr_ratio = safe_round((price - target) / risk_dist, 2)

    else:
        # HOLD
        sl = safe_round(price - 2 * effective_atr)
        target = safe_round(price + 2 * effective_atr)
        rr_ratio = 1.0
        risk_dist = 2 * effective_atr

    # Position Sizing
    if risk_dist > 0:
        shares_to_buy = int(risk_budget / risk_dist)
    else:
        shares_to_buy = 0
    
    if shares_to_buy == 0:
        shares_to_buy = 1

    capital_required = safe_round(shares_to_buy * price, 2)
    max_loss = safe_round(shares_to_buy * risk_dist, 2)

    return target, sl, rr_ratio, shares_to_buy, capital_required, max_loss


def normalize_reason_list(reasons: Any, fallback: List[str]) -> List[str]:
    if not isinstance(reasons, list):
        return fallback
    cleaned = [str(x).strip() for x in reasons if str(x).strip()]
    return cleaned[:4] if cleaned else fallback


def normalize_signal(value: Any, fallback: str = "HOLD") -> str:
    signal = str(value or fallback).upper().strip()
    return signal if signal in ["BUY", "SELL", "HOLD"] else fallback


def normalize_strength(value: Any, fallback: str = "Moderate") -> str:
    strength = str(value or fallback).title().strip()
    return strength if strength in ["Strong", "Moderate", "Weak"] else fallback


def normalize_risk(value: Any, fallback: str = "Medium") -> str:
    risk = str(value or fallback).title().strip()
    return risk if risk in ["Low", "Medium", "High"] else fallback


def ai_analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    base_score = safe_float(data.get("base_score"), 0)
    rule_signal, strength = rule_signal_from_score(base_score)
    confidence = int(clamp_number(55 + abs(base_score) * 0.6, 50, 92))

    # Calculate dynamic S/R targets
    price = safe_float(data.get("price"), 0)
    atr = safe_float(data.get("atr"), 0)
    support = safe_float(data.get("support"), price * 0.98)
    resistance = safe_float(data.get("resistance"), price * 1.02)

    target, stoploss, rr_ratio, shares_to_buy, capital_required, max_loss = compute_targets(
        price,
        atr,
        support,
        resistance,
        rule_signal,
        confidence,
    )

    # Fetch news headlines if they are not pre-fetched
    recent_news = data.get("recent_news")
    if recent_news is None:
        recent_news = get_recent_news_headlines(data.get("symbol"))
    
    recent_news_count = len(recent_news)
    news_text = "\n".join([f"- {h}" for h in recent_news]) if recent_news else "No recent news found."

    # Adjust risk label based on confidence
    risk = "Low" if confidence >= 82 else "Medium" if confidence >= 68 else "High"

    default_reasons = [
        f"Trend price vs MA (Price {price}, MA20 {data.get('ma20')}) aur indicators aligned hain.",
        f"Support (Rs {support}) aur Resistance (Rs {resistance}) ko stoploss aur target calculation me use kiya gaya hai.",
        f"Risk-to-Reward ratio {rr_ratio}:1 hai. Rs 1,000 risk ke liye {shares_to_buy} shares buy karna suggested hai.",
    ]
    if recent_news_count > 0:
        default_reasons.append(f"{recent_news_count} news headlines evaluate kiye gaye hain.")
    
    default_summary = (
        f"{data.get('name', data.get('symbol', 'Stock'))} ke liye "
        f"combined technical {rule_signal} setup detect hua hai jiska Risk-to-Reward {rr_ratio}:1 hai."
    )

    if not client:
        return {
            "signal": rule_signal,
            "strength": strength,
            "confidence": confidence,
            "target": target,
            "stoploss": stoploss,
            "risk": risk,
            "reasons": default_reasons,
            "summary": default_summary,
            "ai_used": False,
            "risk_reward_ratio": rr_ratio,
            "shares_to_buy": shares_to_buy,
            "capital_required": capital_required,
            "max_loss": max_loss,
            "recent_news_count": recent_news_count,
        }

    prompt = f"""
Tu disciplined Indian stock market technical analyst aur veteran trader hai jise 50 saal ka trading experience hai.
Primary trade decision rule-based signals se lo. AI ka kaam technical analysis aur news sentiment evaluate karke explanation aur minor target refinement dena hai.

Symbol: {data.get('symbol')}
Company: {data.get('name', 'N/A')}
Sector: {data.get('sector', 'N/A')}
Industry: {data.get('industry', 'N/A')}
Price: Rs {price}
Change %: {data.get('change_pct', 0)}
MA20: {data.get('ma20')}
MA50: {data.get('ma50')}
MA200: {data.get('ma200')}
RSI: {data.get('rsi')}
MACD Hist: {data.get('macd_hist')}
ADX: {data.get('adx')}
ATR: {atr}
Volume Ratio: {data.get('vol_ratio')}x
Support: Rs {support}
Resistance: Rs {resistance}
Intraday score: {data.get('intraday_score')} (Bias: {data.get('intraday_bias')})
Swing score: {data.get('swing_score')} (Bias: {data.get('swing_bias')})
Combined base score: {base_score}

Recent News Headlines for this Stock:
{news_text}

Rule-based suggested trade parameters:
Signal: {rule_signal}
Suggested Stop Loss: Rs {stoploss}
Suggested Target: Rs {target}
Suggested Risk-to-Reward: {rr_ratio}:1
Confidence: {confidence}% (Risk profile: {risk})

Sirf valid JSON object return karo. Koi extra introductory ya explaining text nahi hona chahiye. JSON structure:
{{
  "signal": "BUY ya SELL ya HOLD",
  "strength": "Strong ya Moderate ya Weak",
  "confidence": 0 se 100 ke beech integer,
  "target": number (support/resistance/atr ke according refine karein),
  "stoploss": number (support/resistance/atr ke according refine karein),
  "risk": "Low ya Medium ya High",
  "reasons": ["Hindi me trade validation reason 1", "Hindi me reason 2 (news/technical alignment)", "Hindi me reason 3 (risk profile/levels)"],
  "summary": "2 line summary Hindi me (news aur technicals ka combination)"
}}
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        parsed = json.loads(text)

        if not isinstance(parsed, dict):
            raise ValueError("AI response dict format me nahi hai.")

        signal = normalize_signal(parsed.get("signal"), rule_signal)
        strength_value = normalize_strength(parsed.get("strength"), strength)
        risk_value = normalize_risk(parsed.get("risk"), risk)

        ai_target = safe_float(parsed.get("target"), target)
        ai_stoploss = safe_float(parsed.get("stoploss"), stoploss)

        ai_confidence_raw = parsed.get("confidence", confidence)
        try:
            ai_confidence = int(ai_confidence_raw)
        except Exception:
            ai_confidence = confidence

        if not (0 <= ai_confidence <= 100):
            ai_confidence = confidence

        # Recalculate Risk-to-Reward and Position Sizing based on AI refined stoploss and target
        risk_dist = abs(price - ai_stoploss)
        if risk_dist > 0:
            rr_ratio = safe_round(abs(ai_target - price) / risk_dist, 2)
            shares_to_buy = int(1000.0 / risk_dist)
        else:
            rr_ratio = 1.0
            shares_to_buy = 1

        if shares_to_buy == 0:
            shares_to_buy = 1

        capital_required = safe_round(shares_to_buy * price, 2)
        max_loss = safe_round(shares_to_buy * risk_dist, 2)

        reasons = normalize_reason_list(parsed.get("reasons"), default_reasons)
        summary = str(parsed.get("summary", default_summary)).strip() or default_summary

        return {
            "signal": signal,
            "strength": strength_value,
            "confidence": ai_confidence,
            "target": safe_round(ai_target),
            "stoploss": safe_round(ai_stoploss),
            "risk": risk_value,
            "reasons": reasons,
            "summary": summary,
            "ai_used": True,
            "risk_reward_ratio": rr_ratio,
            "shares_to_buy": shares_to_buy,
            "capital_required": capital_required,
            "max_loss": max_loss,
            "recent_news_count": recent_news_count,
        }
    except Exception as e:
        logger.error(f"AI analyze error: {e}")
        return {
            "signal": rule_signal,
            "strength": strength,
            "confidence": confidence,
            "target": target,
            "stoploss": stoploss,
            "risk": risk,
            "reasons": default_reasons + [f"AI fallback: {e}"],
            "summary": "Rule-based combined technical analysis ke basis par output diya gaya hai.",
            "ai_used": False,
            "risk_reward_ratio": rr_ratio,
            "shares_to_buy": shares_to_buy,
            "capital_required": capital_required,
            "max_loss": max_loss,
            "recent_news_count": recent_news_count,
        }


def build_shortlist_item(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": data["symbol"],
        "name": data["name"],
        "sector": data.get("sector", "N/A"),
        "industry": data.get("industry", "N/A"),
        "price": data.get("price"),
        "change_pct": data.get("change_pct"),
        "intraday_score": data.get("intraday_score"),
        "intraday_bias": data.get("intraday_bias"),
        "swing_score": data.get("swing_score"),
        "swing_bias": data.get("swing_bias"),
        "base_score": data.get("base_score"),
        "rsi": data.get("rsi"),
        "adx": data.get("adx"),
        "vol_ratio": data.get("vol_ratio"),
        "atr": data.get("atr"),
        "ma20": data.get("ma20"),
        "ma50": data.get("ma50"),
        "ma200": data.get("ma200"),
        "support": data.get("support"),
        "resistance": data.get("resistance"),
        "weekly_trend": data.get("weekly_trend", "Neutral"),
        "weekly_rsi": data.get("weekly_rsi", 50.0),
        "candlestick_patterns": data.get("candlestick_patterns", []),
        "patterns_str": data.get("patterns_str", "None"),
    }


def build_news_pick_item(data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    signal = normalize_signal(analysis.get("signal"), "HOLD")
    confidence = int(safe_float(analysis.get("confidence"), 0))
    base_score = safe_float(data.get("base_score"), 0)
    intraday_score = safe_float(data.get("intraday_score"), 0)
    swing_score = safe_float(data.get("swing_score"), 0)
    volume_boost = safe_float(data.get("vol_ratio"), 0) * 5
    trend_boost = safe_float(data.get("adx"), 0) * 0.6

    directional_bonus = 0
    if signal == "BUY":
        directional_bonus = 14
    elif signal == "SELL":
        directional_bonus = 8

    news_score = safe_round(
        base_score * 0.45
        + intraday_score * 0.20
        + swing_score * 0.20
        + confidence * 0.20
        + volume_boost
        + trend_boost
        + directional_bonus,
        2,
        0,
    )

    return {
        "symbol": data["symbol"],
        "name": data["name"],
        "sector": data.get("sector", "N/A"),
        "industry": data.get("industry", "N/A"),
        "price": data.get("price"),
        "change_pct": data.get("change_pct"),
        "signal": signal,
        "strength": normalize_strength(analysis.get("strength"), "Moderate"),
        "confidence": confidence,
        "risk": normalize_risk(analysis.get("risk"), "Medium"),
        "summary": str(analysis.get("summary", "")).strip(),
        "reasons": normalize_reason_list(analysis.get("reasons"), []),
        "target": analysis.get("target"),
        "stoploss": analysis.get("stoploss"),
        "base_score": base_score,
        "intraday_score": intraday_score,
        "swing_score": swing_score,
        "intraday_bias": data.get("intraday_bias", "HOLD"),
        "swing_bias": data.get("swing_bias", "HOLD"),
        "rsi": data.get("rsi"),
        "adx": data.get("adx"),
        "vol_ratio": data.get("vol_ratio"),
        "atr": data.get("atr"),
        "ma20": data.get("ma20"),
        "ma50": data.get("ma50"),
        "ma200": data.get("ma200"),
        "support": data.get("support"),
        "resistance": data.get("resistance"),
        "risk_reward_ratio": analysis.get("risk_reward_ratio", 1.0),
        "shares_to_buy": analysis.get("shares_to_buy", 1),
        "capital_required": analysis.get("capital_required", 0.0),
        "max_loss": analysis.get("max_loss", 0.0),
        "news_score": news_score,
        "news_sentiment": signal,
        "headline_signal": signal,
        "recent_news_count": analysis.get("recent_news_count", 0),
        "ai_used": bool(analysis.get("ai_used", False)),
        "generated_at": utc_now_iso(),
    }


def build_shortlists(symbols: Optional[List[str]] = None, top_n: int = 5) -> Dict[str, Any]:
    top_n = max(1, min(int(top_n), 20))
    universe = dedupe_preserve_order(symbols or SHORTLIST_UNIVERSE)

    # Prefetch Nifty 50 index (^NSEI) along with Nifty 200 universe for efficiency
    universe_with_index = list(universe) + ["^NSEI"]
    prefetch_all_stocks_data(universe_with_index)

    # Calculate overall market trend (Nifty 50)
    nifty_trend = "Neutral"
    try:
        nifty_df = extract_ticker_df(_batch_history_df, "^NSEI")
        if nifty_df is not None and not nifty_df.empty:
            nifty_close = nifty_df["Close"].dropna()
            if len(nifty_close) >= 20:
                nifty_ma20 = nifty_close.tail(20).mean()
                nifty_last = nifty_close.iloc[-1]
                nifty_trend = "Bullish" if nifty_last > nifty_ma20 else "Bearish"
    except Exception as e:
        logger.warning(f"Error calculating Nifty trend: {e}")

    analyzed: List[Dict[str, Any]] = []
    errors: List[str] = []

    logger.info(f"Shortlist analyzing {len(universe)} symbols, top_n={top_n}")

    for symbol in universe:
        try:
            data = get_stock_data(symbol)
            if not data:
                errors.append(f"{symbol}: data unavailable")
                continue

            analyzed.append(data)
        except Exception as e:
            logger.error(f"Shortlist error for {symbol}: {e}")
            errors.append(f"{symbol}: {str(e)}")

    # Calculate sector relative strength
    sector_changes = {}
    for item in analyzed:
        sec = item.get("sector")
        chg = item.get("change_pct", 0.0)
        if sec and sec != "N/A":
            sector_changes.setdefault(sec, []).append(chg)
            
    sector_averages = {sec: safe_round(sum(vals)/len(vals), 2) for sec, vals in sector_changes.items()}
    sorted_sectors = sorted(sector_averages.items(), key=lambda x: x[1], reverse=True)
    sector_ranks = {sec: idx + 1 for idx, (sec, _) in enumerate(sorted_sectors)}

    intraday_buy = sorted(
        [x for x in analyzed if x["intraday_bias"] == "BUY"],
        key=lambda x: (x["intraday_score"], x["base_score"]),
        reverse=True,
    )[:top_n]
    intraday_hold_fallback = sorted(
        [x for x in analyzed if x["intraday_bias"] == "HOLD"],
        key=lambda x: (x["intraday_score"], x["base_score"]),
        reverse=True,
    )
    intraday_sell_fallback = sorted(
        [x for x in analyzed if x["intraday_bias"] == "SELL"],
        key=lambda x: (x["intraday_score"], x["base_score"]),
        reverse=True,
    )

    intraday_top = list(intraday_buy)
    if len(intraday_top) < top_n:
        intraday_top += intraday_hold_fallback[: top_n - len(intraday_top)]
    if len(intraday_top) < top_n:
        intraday_top += intraday_sell_fallback[: top_n - len(intraday_top)]

    swing_buy = sorted(
        [x for x in analyzed if x["swing_bias"] == "BUY"],
        key=lambda x: (x["swing_score"], x["base_score"]),
        reverse=True,
    )[:top_n]
    swing_hold_fallback = sorted(
        [x for x in analyzed if x["swing_bias"] == "HOLD"],
        key=lambda x: (x["swing_score"], x["base_score"]),
        reverse=True,
    )
    swing_sell_fallback = sorted(
        [x for x in analyzed if x["swing_bias"] == "SELL"],
        key=lambda x: (x["swing_score"], x["base_score"]),
        reverse=True,
    )

    swing_top = list(swing_buy)
    if len(swing_top) < top_n:
        swing_top += swing_hold_fallback[: top_n - len(swing_top)]
    if len(swing_top) < top_n:
        swing_top += swing_sell_fallback[: top_n - len(swing_top)]

    # Multi-Timeframe Picks (Aligned setups: Bullish Weekly Trend + BUY/HOLD Daily Bias)
    mtf_buy = sorted(
        [x for x in analyzed if x.get("weekly_trend") == "Bullish" and x.get("intraday_bias") == "BUY"],
        key=lambda x: (x["intraday_score"], x["base_score"]),
        reverse=True,
    )[:top_n]
    mtf_hold_fallback = sorted(
        [x for x in analyzed if x.get("weekly_trend") == "Bullish" and x.get("intraday_bias") == "HOLD"],
        key=lambda x: (x["intraday_score"], x["base_score"]),
        reverse=True,
    )
    mtf_top = list(mtf_buy)
    if len(mtf_top) < top_n:
        mtf_top += mtf_hold_fallback[: top_n - len(mtf_top)]
    if len(mtf_top) < top_n:
        mtf_top += [x for x in intraday_top if x not in mtf_top][: top_n - len(mtf_top)]

    # Candlestick Picks (Stocks with daily patterns detected)
    candle_buy = sorted(
        [x for x in analyzed if x.get("candlestick_patterns") and x.get("intraday_bias") == "BUY"],
        key=lambda x: (x["base_score"]),
        reverse=True,
    )[:top_n]
    candle_hold_fallback = sorted(
        [x for x in analyzed if x.get("candlestick_patterns") and x.get("intraday_bias") == "HOLD"],
        key=lambda x: (x["base_score"]),
        reverse=True,
    )
    candle_any_fallback = sorted(
        [x for x in analyzed if x.get("candlestick_patterns")],
        key=lambda x: (x["base_score"]),
        reverse=True,
    )
    candle_top = list(candle_buy)
    if len(candle_top) < top_n:
        candle_top += [x for x in candle_hold_fallback if x not in candle_top][: top_n - len(candle_top)]
    if len(candle_top) < top_n:
        candle_top += [x for x in candle_any_fallback if x not in candle_top][: top_n - len(candle_top)]
    if len(candle_top) < top_n:
        candle_top += [x for x in swing_top if x not in candle_top][: top_n - len(candle_top)]

    def compile_final_shortlist_item(item_data: Dict[str, Any]) -> Dict[str, Any]:
        item_data = item_data.copy()
        
        # Inject Sector relative strength ranking
        sec = item_data.get("sector")
        rank = sector_ranks.get(sec, 99)
        avg_chg = sector_averages.get(sec, 0.0)
        
        # Top 3 Sectors get a +10 points score boost (Relative Strength confirmation)
        if rank <= 3:
            item_data["base_score"] = item_data.get("base_score", 0) + 10
            if item_data.get("intraday_score", 0) > 0:
                item_data["intraday_score"] = item_data.get("intraday_score", 0) + 5
            if item_data.get("swing_score", 0) > 0:
                item_data["swing_score"] = item_data.get("swing_score", 0) + 5

        # Downgrade stock BUY setup scoring if overall Nifty 50 market trend is Bearish
        if nifty_trend == "Bearish":
            item_data["base_score"] = item_data.get("base_score", 0) - 15

        analysis = ai_analyze(item_data)
        
        # Adjust signal calculations for bearish index structure
        if nifty_trend == "Bearish" and analysis.get("signal") == "BUY":
            analysis["confidence"] = max(50, int(analysis.get("confidence", 70) - 12))
            reasons = list(analysis.get("reasons", []))
            reasons.insert(0, "MARKET WARNING: Nifty 50 index is in a downtrend. Proceed with caution.")
            analysis["reasons"] = reasons

        shortlist_item = build_shortlist_item(item_data)
        shortlist_item.update({
            "weekly_trend": item_data.get("weekly_trend", "Neutral"),
            "weekly_rsi": item_data.get("weekly_rsi", 50.0),
            "candlestick_patterns": item_data.get("candlestick_patterns", []),
            "patterns_str": item_data.get("patterns_str", "None"),
        })

        sector_text = "N/A"
        if sec and sec != "N/A":
            sector_text = f"{sec} #{rank} ({'+' if avg_chg >= 0 else ''}{avg_chg:.2f}%)"

        return {
            **shortlist_item,
            **analysis,
            "nifty_trend": nifty_trend,
            "sector_rank": rank,
            "sector_avg_change": avg_chg,
            "sector_performance_text": sector_text
        }

    intraday_top_final = [compile_final_shortlist_item(x) for x in intraday_top[:top_n]]
    swing_top_final = [compile_final_shortlist_item(x) for x in swing_top[:top_n]]
    mtf_top_final = [compile_final_shortlist_item(x) for x in mtf_top[:top_n]]
    candle_top_final = [compile_final_shortlist_item(x) for x in candle_top[:top_n]]

    return {
        "success": True,
        "version": APP_VERSION,
        "generated_at": utc_now_iso(),
        "ai_enabled": bool(client),
        "top_n": top_n,
        "nifty_trend": nifty_trend,
        "universe_size": len(universe),
        "analyzed_count": len(analyzed),
        "error_count": len(errors),
        "intraday_top": intraday_top_final,
        "swing_top": swing_top_final,
        "multi_timeframe_top": mtf_top_final,
        "candlestick_top": candle_top_final,
        "errors": errors[:20],
    }


def build_news_picks(symbols: Optional[List[str]] = None, top_n: int = 5) -> Dict[str, Any]:
    top_n = max(1, min(int(top_n), 20))
    universe = dedupe_preserve_order(symbols or SHORTLIST_UNIVERSE)

    # Prefetch all stocks data in batch!
    prefetch_all_stocks_data(universe)

    analyzed_raw: List[Dict[str, Any]] = []
    errors: List[str] = []

    logger.info(f"News picks analyzing {len(universe)} symbols, top_n={top_n}")

    # Step 1: Technical evaluation and initial scoring (Rule-based, instant)
    for symbol in universe:
        try:
            data = get_stock_data(symbol)
            if not data:
                errors.append(f"{symbol}: data unavailable")
                continue
            
            # Simple technical ranking score:
            base_score = safe_float(data.get("base_score"), 0)
            intraday_score = safe_float(data.get("intraday_score"), 0)
            swing_score = safe_float(data.get("swing_score"), 0)
            volume_boost = safe_float(data.get("vol_ratio"), 0) * 5
            trend_boost = safe_float(data.get("adx"), 0) * 0.6
            
            # Temporary sorting score to pick best candidate setups
            temp_score = base_score * 0.45 + intraday_score * 0.20 + swing_score * 0.20 + volume_boost + trend_boost
            
            # Store temp score along with raw stock data
            data["_temp_score"] = temp_score
            analyzed_raw.append(data)
        except Exception as e:
            logger.error(f"News picks raw analysis error for {symbol}: {e}")
            errors.append(f"{symbol}: {str(e)}")

    # Sort raw data by temp score descending
    ranked_candidates = sorted(analyzed_raw, key=lambda x: x.get("_temp_score", 0), reverse=True)
    
    # We only analyze the top candidates using AI (limit to top_n * 2, max 10)
    candidate_pool_limit = max(top_n, min(top_n * 2, 10))
    candidates_to_ai = ranked_candidates[:candidate_pool_limit]
    
    analyzed_final: List[Dict[str, Any]] = []
    
    # Step 2: Run dynamic news headlines fetch and AI analysis ONLY on top candidates!
    for data in candidates_to_ai:
        symbol = data["symbol"]
        try:
            logger.info(f"Running dynamic news fetch & AI analysis for candidate: {symbol}")
            # Dynamic news fetch specifically for the candidate
            recent_news = get_recent_news_headlines(symbol)
            data["recent_news"] = recent_news
            
            # Call AI analysis
            analysis = ai_analyze(data)
            
            analyzed_final.append(build_news_pick_item(data, analysis))
        except Exception as e:
            logger.error(f"News picks AI analyze error for {symbol}: {e}")
            errors.append(f"{symbol}: {str(e)}")

    # Step 3: Sort final AI news picks by final news_score and confidence
    ranked_picks = sorted(
        analyzed_final,
        key=lambda x: (
            x.get("news_score", 0),
            x.get("confidence", 0),
            x.get("base_score", 0),
        ),
        reverse=True,
    )

    return {
        "success": True,
        "version": APP_VERSION,
        "generated_at": utc_now_iso(),
        "ai_enabled": bool(client),
        "top_n": top_n,
        "universe_size": len(universe),
        "analyzed_count": len(analyzed_final),
        "error_count": len(errors),
        "items": ranked_picks[:top_n],
        "errors": errors[:20],
    }


def run(symbol: str) -> None:
    if not symbol.endswith((".NS", ".BO")):
        symbol = symbol.upper().strip()

    print()
    print(f"Fetching {symbol}...")
    data = get_stock_data(symbol)

    if not data:
        print(f"{symbol} ka data nahi mila. Symbol ya exchange check karein.")
        return

    print("AI analysis ho raha hai...")
    result = ai_analyze(data)

    signal = result.get("signal", "HOLD")
    sep = "=" * 64

    print(sep)
    print(f"{data['name']}")
    print(f"{data['symbol']}")
    print(sep)
    print(f"Version: {data.get('debug_version', 'N/A')}")
    print(f"Generated At: {data.get('generated_at', 'N/A')}")
    print(f"Sector: {data.get('sector')} | {data.get('industry')}")
    print(f"Price: Rs {data['price']} ({data['change_pct']}%)")
    print(f"Support: Rs {data.get('support')} | Resistance: Rs {data.get('resistance')}")
    print(f"Signal: {signal} | {result.get('strength', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0)}% | Risk: {result.get('risk', 'N/A')}")
    print(f"Target: Rs {result.get('target', 'N/A')} | Stop Loss: Rs {result.get('stoploss', 'N/A')}")
    print(f"Risk-to-Reward: {result.get('risk_reward_ratio', 1.0)}:1")
    print(f"Suggested Position Size (Risk Rs 1,000): {result.get('shares_to_buy', 1)} shares")
    print(f"Capital Required: Rs {result.get('capital_required', 0.0)} | Max Loss: Rs {result.get('max_loss', 0.0)}")
    print(f"AI Used: {result.get('ai_used', False)}")
    print(f"Intraday: {data.get('intraday_score')} | {data.get('intraday_bias')}")
    print(f"Swing: {data.get('swing_score')} | {data.get('swing_bias')}")
    print(f"Base Score: {data.get('base_score')}")
    print(f"PE: {data.get('pe')} | Beta: {data.get('beta')}")
    print(f"MCap Cr: {data.get('market_cap_cr')} | ROE: {data.get('roe')} | Div: {data.get('dividend_yield')}")
    print(f"RSI: {data.get('rsi')} | ADX: {data.get('adx')} | ATR: {data.get('atr')}")
    print(f"MACD/Sig/Hist: {data.get('macd')} / {data.get('macd_signal')} / {data.get('macd_hist')}")
    print(f"MA7/20/50/200: {data.get('ma7')} / {data.get('ma20')} / {data.get('ma50')} / {data.get('ma200')}")
    print(f"BB L/M/U: {data.get('bb_lower')} / {data.get('bb_mid')} / {data.get('bb_upper')}")
    print(f"52W H/L: {data.get('high52')} / {data.get('low52')}")
    print()
    print("Summary:")
    print(result.get("summary", ""))
    print()
    print("Reasons:")
    for i, reason in enumerate(result.get("reasons", []), 1):
        print(f"{i}. {reason}")
    print(sep)


if __name__ == "__main__":
    if not GROQ_API_KEY:
        print()
        print("WARNING: GROQAPIKEY set nahi hai. Sirf rule-based analysis chalega.")
        print()

    if len(sys.argv) > 1 and sys.argv[1].lower() == "shortlist":
        top_n = 5
        if len(sys.argv) > 2:
            try:
                top_n = int(sys.argv[2])
            except Exception:
                pass
        result = build_shortlists(top_n=top_n)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "news":
        top_n = 5
        if len(sys.argv) > 2:
            try:
                top_n = int(sys.argv[2])
            except Exception:
                pass
        result = build_news_picks(top_n=top_n)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "meta":
        print(json.dumps(build_meta(), indent=2, ensure_ascii=False))

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "health":
        print(json.dumps(build_health(), indent=2, ensure_ascii=False))

    elif len(sys.argv) > 1:
        run(sys.argv[1])

    else:
        print()
        print(f"NSE Stock AI Analyzer - {APP_VERSION}")
        print(f"{len(STOCKS_TO_ANALYZE)} stocks analyze ho rahe hain...")
        print()

        for stock_symbol in STOCKS_TO_ANALYZE:
            run(stock_symbol)

        print()
        print("Sample shortlists (top 3):")
        shortlist_result = build_shortlists(top_n=3)
        print(json.dumps(shortlist_result, indent=2, ensure_ascii=False))

        print()
        print("Sample news picks (top 3):")
        news_result = build_news_picks(top_n=3)
        print(json.dumps(news_result, indent=2, ensure_ascii=False))

        print()
        print("Sab stocks analyze ho gaye!")
