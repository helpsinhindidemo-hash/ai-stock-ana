from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List
import os
import time

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure you have your internal imports correct.
from stock_analyzer import (
    get_stock_data,
    ai_analyze,
    build_shortlists,
    build_meta,
    build_health,
    build_news_picks,
    prefetch_all_stocks_data,
)

APP_VERSION = "api_server_v9_news_picks"
APP_START_TS = time.time()

NIFTY_100_STOCKS: List[Dict[str, str]] = [
    {"name": "ABB India", "symbol": "ABB"},
    {"name": "Adani Enterprises", "symbol": "ADANIENT"},
    {"name": "Adani Green Energy", "symbol": "ADANIGREEN"},
    {"name": "Adani Ports & SEZ", "symbol": "ADANIPORTS"},
    {"name": "Ambuja Cements", "symbol": "AMBUJACEM"},
    {"name": "Apollo Hospitals", "symbol": "APOLLOHOSP"},
    {"name": "Asian Paints", "symbol": "ASIANPAINT"},
    {"name": "Avenue Supermarts", "symbol": "DMART"},
    {"name": "Axis Bank", "symbol": "AXISBANK"},
    {"name": "Bajaj Auto", "symbol": "BAJAJ-AUTO"},
    {"name": "Bajaj Finance", "symbol": "BAJFINANCE"},
    {"name": "Bajaj Finserv", "symbol": "BAJAJFINSV"},
    {"name": "Bank of Baroda", "symbol": "BANKBARODA"},
    {"name": "Bharat Electronics", "symbol": "BEL"},
    {"name": "Bharat Petroleum", "symbol": "BPCL"},
    {"name": "Bharti Airtel", "symbol": "BHARTIARTL"},
    {"name": "Britannia Industries", "symbol": "BRITANNIA"},
    {"name": "Canara Bank", "symbol": "CANBK"},
    {"name": "Cipla", "symbol": "CIPLA"},
    {"name": "Coal India", "symbol": "COALINDIA"},
    {"name": "Cummins India", "symbol": "CUMMINSIND"},
    {"name": "Dabur India", "symbol": "DABUR"},
    {"name": "Divis Laboratories", "symbol": "DIVISLAB"},
    {"name": "DLF", "symbol": "DLF"},
    {"name": "Dr Reddys Laboratories", "symbol": "DRREDDY"},
    {"name": "Eicher Motors", "symbol": "EICHERMOT"},
    {"name": "Eternal", "symbol": "ETERNAL"},
    {"name": "GAIL India", "symbol": "GAIL"},
    {"name": "Godrej Consumer Products", "symbol": "GODREJCP"},
    {"name": "Grasim Industries", "symbol": "GRASIM"},
    {"name": "HAL", "symbol": "HAL"},
    {"name": "HCLTech", "symbol": "HCLTECH"},
    {"name": "HDFC AMC", "symbol": "HDFCAMC"},
    {"name": "HDFC Bank", "symbol": "HDFCBANK"},
    {"name": "HDFC Life", "symbol": "HDFCLIFE"},
    {"name": "Hero MotoCorp", "symbol": "HEROMOTOCO"},
    {"name": "Hindalco Industries", "symbol": "HINDALCO"},
    {"name": "Hindustan Unilever", "symbol": "HINDUNILVR"},
    {"name": "ICICI Bank", "symbol": "ICICIBANK"},
    {"name": "Indian Hotels", "symbol": "INDHOTEL"},
    {"name": "Indian Oil Corporation", "symbol": "IOC"},
    {"name": "IndiGo", "symbol": "INDIGO"},
    {"name": "IndusInd Bank", "symbol": "INDUSINDBK"},
    {"name": "Infosys", "symbol": "INFY"},
    {"name": "ITC", "symbol": "ITC"},
    {"name": "Jio Financial Services", "symbol": "JIOFIN"},
    {"name": "JSW Steel", "symbol": "JSWSTEEL"},
    {"name": "Kotak Mahindra Bank", "symbol": "KOTAKBANK"},
    {"name": "Larsen & Toubro", "symbol": "LT"},
    {"name": "Mahindra & Mahindra", "symbol": "M&M"},
    {"name": "Marico", "symbol": "MARICO"},
    {"name": "Maruti Suzuki", "symbol": "MARUTI"},
    {"name": "Max Healthcare", "symbol": "MAXHEALTH"},
    {"name": "Muthoot Finance", "symbol": "MUTHOOTFIN"},
    {"name": "Nestle India", "symbol": "NESTLEIND"},
    {"name": "NTPC", "symbol": "NTPC"},
    {"name": "Oil & Natural Gas Corporation", "symbol": "ONGC"},
    {"name": "Pidilite Industries", "symbol": "PIDILITIND"},
    {"name": "PNB", "symbol": "PNB"},
    {"name": "Power Finance Corporation", "symbol": "PFC"},
    {"name": "Power Grid", "symbol": "POWERGRID"},
    {"name": "REC", "symbol": "RECLTD"},
    {"name": "Reliance Industries", "symbol": "RELIANCE"},
    {"name": "SBI Life Insurance", "symbol": "SBILIFE"},
    {"name": "Shriram Finance", "symbol": "SHRIRAMFIN"},
    {"name": "Siemens", "symbol": "SIEMENS"},
    {"name": "State Bank of India", "symbol": "SBIN"},
    {"name": "Sun Pharma", "symbol": "SUNPHARMA"},
    {"name": "Tata Consultancy Services", "symbol": "TCS"},
    {"name": "Tata Consumer Products", "symbol": "TATACONSUM"},
    {"name": "Tata Power", "symbol": "TATAPOWER"},
    {"name": "Tata Steel", "symbol": "TATASTEEL"},
    {"name": "Tech Mahindra", "symbol": "TECHM"},
    {"name": "Titan Company", "symbol": "TITAN"},
    {"name": "Torrent Pharmaceuticals", "symbol": "TORNTPHARM"},
    {"name": "Trent", "symbol": "TRENT"},
    {"name": "TVS Motor", "symbol": "TVSMOTOR"},
    {"name": "UltraTech Cement", "symbol": "ULTRACEMCO"},
    {"name": "Union Bank of India", "symbol": "UNIONBANK"},
    {"name": "United Spirits", "symbol": "UNITDSPR"},
    {"name": "Varun Beverages", "symbol": "VBL"},
    {"name": "Vedanta", "symbol": "VEDL"},
    {"name": "Wipro", "symbol": "WIPRO"},
    {"name": "Zydus Lifesciences", "symbol": "ZYDUSLIFE"},
]

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
ALLOW_CREDENTIALS = os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_allowed_origins() -> List[str]:
    if not ALLOWED_ORIGINS.strip():
        return ["*"]
    origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
    return origins or ["*"]


def get_cors_allow_credentials(origins: List[str]) -> bool:
    if origins == ["*"] and ALLOW_CREDENTIALS:
        return False
    return ALLOW_CREDENTIALS


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.upper().strip()
    if not cleaned:
        return ""
    # Special fallback for ampersand splitting in URLs (e.g. M&M -> M)
    if cleaned == "M" or cleaned == "M_M":
        cleaned = "M&M"
    if not cleaned.endswith((".NS", ".BO")):
        cleaned = f"{cleaned}.NS"
    return cleaned


def clean_stock_list(stocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    cleaned: List[Dict[str, str]] = []
    for item in stocks:
        name = item.get("name", "").strip()
        symbol = item.get("symbol", "").strip().upper()
        if not name or not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append({"name": name, "symbol": symbol})
    cleaned.sort(key=lambda x: x["name"].lower())
    return cleaned


def build_universe_symbols(stocks: List[Dict[str, str]]) -> List[str]:
    return [f"{item['symbol']}.NS" for item in stocks]


def make_base_response(success: bool = True) -> Dict[str, Any]:
    return {
        "success": success,
        "debug_version": APP_VERSION,
        "generated_at": utc_now_iso(),
        "uptime_seconds": round(time.time() - APP_START_TS, 2),
    }


CLEANED_NIFTY_100_STOCKS = clean_stock_list(NIFTY_100_STOCKS)
CLEANED_NIFTY_100_SYMBOLS = build_universe_symbols(CLEANED_NIFTY_100_STOCKS)
CORS_ORIGINS = get_allowed_origins()
CORS_ALLOW_CREDENTIALS = get_cors_allow_credentials(CORS_ORIGINS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60, flush=True)
    print("Backend started and ready.", flush=True)
    print(f"APP_VERSION: {APP_VERSION}", flush=True)
    print(f"Raw stocks loaded: {len(NIFTY_100_STOCKS)}", flush=True)
    print(f"Cleaned stocks loaded: {len(CLEANED_NIFTY_100_STOCKS)}", flush=True)
    print(f"ALLOWED_ORIGINS: {CORS_ORIGINS}", flush=True)
    print(f"ALLOW_CREDENTIALS: {CORS_ALLOW_CREDENTIALS}", flush=True)
    print(f"PORT: {os.getenv('PORT', 'not-set')}", flush=True)
    
    # Prefetch prices for Nifty 100 on startup to warm up cache
    try:
        print("Prefetching price cache for the Nifty 100 universe...", flush=True)
        prefetch_all_stocks_data(CLEANED_NIFTY_100_SYMBOLS)
        print("Startup prefetch completed successfully.", flush=True)
    except Exception as e:
        print(f"Startup prefetch failed: {e}", flush=True)
        
    print("=" * 60, flush=True)
    yield


app = FastAPI(
    title="NSE Stock AI Analyzer API",
    description="Stock analysis API for Indian market stocks with metadata, indicators, AI/rule-based signal generation, shortlist output, and news picks.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home() -> Dict[str, Any]:
    return {
        **make_base_response(True),
        "message": "NSE Stock AI Analyzer API is running",
        "title": app.title,
        "version": APP_VERSION,
        "usage": {
            "analyze": "/analyze/TCS",
            "debug": "/debug/TCS",
            "stocks": "/stocks",
            "shortlist": "/shortlist?top_n=5",
            "news_picks": "/news-picks?top_n=5",
            "health": "/health",
            "meta": "/meta",
        },
    }


@app.get("/meta")
def meta() -> Dict[str, Any]:
    analyzer_meta = build_meta()

    return {
        **make_base_response(True),
        "title": app.title,
        "description": app.description,
        "version": APP_VERSION,
        "server_version": APP_VERSION,
        "analyzer_version": analyzer_meta.get("version"),
        "ai_enabled": analyzer_meta.get("ai_enabled", False),
        "stocks_count": len(CLEANED_NIFTY_100_STOCKS),
        "shortlist_universe_count": len(CLEANED_NIFTY_100_SYMBOLS),
        "allowed_origins": CORS_ORIGINS,
        "allow_credentials": CORS_ALLOW_CREDENTIALS,
        "features": {
            "stocks": True,
            "analyze": True,
            "debug": True,
            "shortlist": True,
            "news_picks": True,
            "meta": True,
            "health": True,
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    analyzer_health = build_health()

    return {
        **make_base_response(True),
        "status": analyzer_health.get("status", "ok"),
        "server_version": APP_VERSION,
        "analyzer_version": analyzer_health.get("version"),
        "ai_enabled": analyzer_health.get("ai_enabled", False),
        "stocks_count": len(CLEANED_NIFTY_100_STOCKS),
        "allowed_origins": CORS_ORIGINS,
    }


@app.get("/stocks")
def get_stocks() -> Dict[str, Any]:
    return {
        **make_base_response(True),
        "index": "NIFTY 100",
        "count": len(CLEANED_NIFTY_100_STOCKS),
        "stocks": CLEANED_NIFTY_100_STOCKS,
    }


@app.get("/shortlist")
def get_shortlist(top_n: int = Query(default=5, ge=1, le=20)) -> Dict[str, Any]:
    print("=" * 50, flush=True)
    print("SHORTLIST REQUEST", flush=True)
    print("Top N requested:", top_n, flush=True)

    try:
        result = build_shortlists(
            symbols=CLEANED_NIFTY_100_SYMBOLS,
            top_n=top_n,
        )

        if not isinstance(result, dict):
            raise ValueError("build_shortlists() ne invalid format return kiya.")

        intraday_top = result.get("intraday_top", [])
        swing_top = result.get("swing_top", [])
        errors = result.get("errors", [])

        return {
            **make_base_response(True),
            "requested_top_n": top_n,
            "top_n": result.get("top_n", top_n),
            "universe_size": result.get("universe_size", len(CLEANED_NIFTY_100_SYMBOLS)),
            "analyzed_count": result.get("analyzed_count", 0),
            "error_count": result.get("error_count", len(errors)),
            "ai_enabled": result.get("ai_enabled", False),
            "version": result.get("version"),
            "intraday_top": intraday_top,
            "swing_top": swing_top,
            "errors": errors,
        }
    except Exception as e:
        print("SHORTLIST ERROR:", str(e), flush=True)
        return {
            **make_base_response(False),
            "requested_top_n": top_n,
            "top_n": top_n,
            "universe_size": len(CLEANED_NIFTY_100_SYMBOLS),
            "analyzed_count": 0,
            "error_count": 1,
            "intraday_top": [],
            "swing_top": [],
            "errors": [str(e)],
            "error": "Shortlist build nahi ho payi",
        }


@app.get("/news-picks")
def get_news_picks(top_n: int = Query(default=5, ge=1, le=20)) -> Dict[str, Any]:
    print("=" * 50, flush=True)
    print("NEWS PICKS REQUEST", flush=True)
    print("Top N requested:", top_n, flush=True)

    try:
        # Note: Depending on your 'stock_analyzer.py' implementation, you might need to pass `top_n`
        result = build_news_picks(symbols=CLEANED_NIFTY_100_SYMBOLS, top_n=top_n)

        if not isinstance(result, dict):
            raise ValueError("build_news_picks() ne invalid format return kiya.")

        items = result.get("items", [])
        errors = result.get("errors", [])

        if not isinstance(items, list):
            raise ValueError("News picks items invalid format me aaye.")

        return {
            **make_base_response(True),
            "requested_top_n": top_n,
            "top_n": result.get("top_n", top_n),
            "universe_size": result.get("universe_size", len(CLEANED_NIFTY_100_SYMBOLS)),
            "analyzed_count": result.get("analyzed_count", len(items)),
            "error_count": result.get("error_count", len(errors)),
            "ai_enabled": result.get("ai_enabled", False),
            "version": result.get("version"),
            "items": items,
            "errors": errors,
        }
    except Exception as e:
        print("NEWS PICKS ERROR:", str(e), flush=True)
        return {
            **make_base_response(False),
            "requested_top_n": top_n,
            "top_n": top_n,
            "universe_size": len(CLEANED_NIFTY_100_SYMBOLS),
            "analyzed_count": 0,
            "error_count": 1,
            "items": [],
            "errors": [str(e)],
            "error": f"News picks build nahi ho payi: {str(e)}",
        }


@app.get("/debug/{symbol}")
def debug_stock(symbol: str) -> Dict[str, Any]:
    original_symbol = symbol.upper().strip()
    final_symbol = normalize_symbol(original_symbol)

    print("=" * 50, flush=True)
    print("DEBUG REQUEST", flush=True)
    print("Input symbol:", original_symbol, flush=True)
    print("Normalized symbol:", final_symbol, flush=True)

    if not final_symbol:
        return {
            **make_base_response(False),
            "input_symbol": original_symbol,
            "final_symbol": final_symbol,
            "data_found": False,
            "error": "Symbol empty hai",
            "data": None,
        }

    try:
        data = get_stock_data(final_symbol)
        return {
            **make_base_response(data is not None),
            "input_symbol": original_symbol,
            "final_symbol": final_symbol,
            "data_found": data is not None,
            "data": data,
            "error": None if data is not None else "Data nahi mila",
        }
    except Exception as e:
        print("DEBUG ERROR:", str(e), flush=True)
        return {
            **make_base_response(False),
            "input_symbol": original_symbol,
            "final_symbol": final_symbol,
            "data_found": False,
            "error": str(e),
            "data": None,
        }


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str) -> Dict[str, Any]:
    original_symbol = symbol.upper().strip()
    final_symbol = normalize_symbol(original_symbol)

    print("=" * 50, flush=True)
    print("ANALYZE REQUEST", flush=True)
    print("Incoming symbol:", original_symbol, flush=True)
    print("Final symbol used:", final_symbol, flush=True)

    if not final_symbol:
        return {
            **make_base_response(False),
            "error": "Stock symbol empty hai",
            "input_symbol": original_symbol,
            "final_symbol": final_symbol,
        }

    try:
        data = get_stock_data(final_symbol)
        if not data:
            print("No data returned for:", final_symbol, flush=True)
            return {
                **make_base_response(False),
                "error": f"{original_symbol} ka data nahi mila",
                "input_symbol": original_symbol,
                "final_symbol": final_symbol,
            }

        print("Data found for:", final_symbol, flush=True)

        result = ai_analyze(data)
        if not isinstance(result, dict):
            raise ValueError("ai_analyze() ne invalid response diya.")

        merged = {
            **make_base_response(True),
            **data,
            **result,
            "input_symbol": original_symbol,
            "final_symbol": final_symbol,
        }

        print("Signal:", merged.get("signal"), flush=True)
        print("Confidence:", merged.get("confidence"), flush=True)

        return merged

    except Exception as e:
        print("ANALYZE ERROR:", str(e), flush=True)
        return {
            **make_base_response(False),
            "error": f"Analysis failed: {str(e)}",
            "input_symbol": original_symbol,
            "final_symbol": final_symbol,
        }
