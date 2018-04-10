import json
from my_site.settings import STOCKS_DATA_FOLDER


def get_stock_file_name(symbol_code):

    filename = STOCKS_DATA_FOLDER + 'stocks_db.json'
    with open(filename, 'r') as f:
        stocks_db = json.load(f)
    for stock in stocks_db['data']:
        if stock['SYMBOL_CODE'] == symbol_code:
            return STOCKS_DATA_FOLDER + stock['name']
    return ''


def get_chart_data(stock_df):

    stock_df = stock_df[-200:]
    dates = []
    for date in stock_df['TRADE_DATE'].values:
        dates.append(str(date)[:10])

    data = {
        'labels': dates,
        'datasets': [{
            'data': stock_df['CLOSE_PRICE'].values.tolist(),
            'label':"CLOSE_PRICE",
            'borderColor':"rgba(173, 209, 243, 1)",
            'borderWidth': 2    ,
            'fill': 'true',
            'pointRadius': 0,
            'fill':'false'}]}
    return data
