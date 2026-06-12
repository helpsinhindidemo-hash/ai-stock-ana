from stock_analyzer import build_news_picks

def get_news_based_picks(top_n: int = 5):
    """
    Wrapper to fetch news picks dynamically from the real stock_analyzer module,
    replacing the previous hardcoded static/mock data.
    """
    result = build_news_picks(top_n=top_n)
    return result
