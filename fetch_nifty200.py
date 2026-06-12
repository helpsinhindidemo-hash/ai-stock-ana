import urllib.request
import pandas as pd

def fetch_and_save():
    url = 'https://archives.nseindia.com/content/indices/ind_nifty200list.csv'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req)
    df = pd.read_csv(res)
    
    with open('nse_nifty200.py', 'w', encoding='utf-8') as f:
        f.write('from typing import List, Dict\n\n')
        f.write('NIFTY_200_STOCKS: List[Dict[str, str]] = [\n')
        for index, row in df.iterrows():
            symbol = str(row['Symbol']).strip()
            name = str(row['Company Name']).strip().replace('"', "'")
            f.write(f'    {{"name": "{name}", "symbol": "{symbol}"}},\n')
        f.write(']\n')

if __name__ == '__main__':
    fetch_and_save()
    print("Successfully saved nse_nifty200.py")
