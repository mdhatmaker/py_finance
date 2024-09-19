from flask import Flask, render_template, request
# from wtforms.fields.choices import RadioField
# from wtforms.fields.simple import SubmitField, TextAreaField, BooleanField
from api.indicators.basic_indicators.adx_indicator import run_adx_indicator
from api.indicators.basic_indicators.bollinger_bands_indicator import run_bollinger_bands_indicator
from api.indicators.basic_indicators.macd_indicator import run_macd_indicator
from api.indicators.basic_indicators.rsi_indicator import run_rsi_indicator
from api.indicators.basic_indicators.stochastic_indicator import run_stochastic_indicator
from api.indicators.ta_indicators.candlestick_ta_patterns import run_candlestick_ta_patterns
from api.indicators.dynamic_time_warping import generate_time_warping_plots
from api.indicators.obscure_indicators.AroonOscillator import run_aroon_oscillator
from api.indicators.obscure_indicators.DemarkerIndicator import plot_stock_with_demarker
from api.indicators.obscure_indicators.RelativeVigorIndex import run_relative_vigor
from api.indicators.obscure_indicators.TrendExhaustion import generate_trend_exhaustion_plot
from api.indicators.obscure_indicators.ThreeWayAverageCrossover import run_threeway_average_crossover
from api.indicators.obscure_indicators.obscure_indicators import run_obscure_indicators
from api.indicators.fundamental_indicators.undervalued_stocks import get_undervalued_stocks, generate_csv
from api.indicators.ta_indicators.pca_dtw import run_pca_dtw
from api.indicators.ta_indicators.support_resistance import run_support_resistance
from api.indicators.volatility_indicators.average_true_range import run_average_true_range
from api.indicators.volatility_indicators.bollinger_bands import run_bollinger_bands
from api.indicators.volatility_indicators.donchian_channels import run_donchian_channels
from api.indicators.volatility_indicators.keltner_channels import run_keltner_channels
from api.indicators.volatility_indicators.relative_volatility_index import run_relative_volatility_index
from api.indicators.volatility_indicators.volatility_chaikin import run_volatility_chaikin
from api.indicators.volume_indicators.volume_indicators import run_volume_nvi, run_volume_obv, run_volume_vpt, run_volume_vroc, \
    run_volume_cmf, run_volume_vwap, run_volume_adline, run_volume_mfi, run_volume_klinger_oscillator
from api.indicators.z_score import run_z_score
from api.indicators.zig_zag_indicator import run_zig_zag
from api.portfolio_risk.hrp_compare_strategy import run_hrp_compare_strategy
from api.portfolio_risk.hrp_portfolio import run_hrp_portfolio
from api.portfolio_risk.hrp_rebalancing import run_hrp_rebalancing
from api.portfolio_risk.risk_parity_rebalancing import run_parity_rebalancing
from api.predictors.nhits_forecast import run_nhits_forecast
from api.predictors.timegpt_forecast import run_timegpt_forecast
from api.research.financial_statements.balance_sheet import run_balance_sheet
from api.research.financial_statements.financial_ratios import run_financial_ratios
from api.research.financial_statements.income_statement import run_analyze_income_statement
from api.research.financial_statements.pandl_statement import run_analyze_financial_statements
from api.signals.pairs_trading_signal.pairs_trading import run_pairs_trading_signal
from api.signals.sentiment_trading_signal.sentiment_analysis import run_sentiment_analysis_signal
from api.strategies.deep_learning_simple.backtest import run_deep_learning_strategy_backtest
from api.strategies.dynamic_renko.strategy import run_dynamic_renko
from api.strategies.ma_crossover_strategy import run_ma_crossover_strategy
from api.strategies.momentum_and_reversion.strategy import run_sma_crossover, run_naive_momentum, run_mean_reversion
from api.strategies.pairs_trading.citadel_strategy import run_citadel_pairs
from api.api_financial_data.coinbase_api_charts import plot_charts
from api.api_financial_data.binance_websocket_api import run_websocket
from api.predictors.prophet_forecast import run_prophet_forecast
from api.predictors.xgboost_forecast import run_xgboost_forecast
from api.indicators.filter_indicators.kalman_filters import run_kalman_filters
from api.indicators.value_at_risk import run_VaR_historical
from api.indicators.ta_indicators.super_trend import run_supertrend_indicator
import asyncio
import ccxt
from typing import List
from datetime import datetime, timedelta
from AudioTranscripts import get_soup, download_audio
from api.api_financial_data.financial_data import EODHDAPIsDataFetcher
from config import EODHD_API_TOKEN


app = Flask(__name__)
# The "SECRET_KEY" variable in the Flask application’s configuration must be set in order to enable CSRF protection.
# This is necessary to create forms using FlaskForm.
app.config["SECRET_KEY"] = "my_trading_secret"


# def get_tastylive_urls():
#     wdis_urls = []
#     url = "https://www.tastylive.com/shows/wdis-bat-vs-bat"
#     soup = get_soup(url)
#     for link in soup:
#         print(link)
#         if 'href' in link.attrs.keys():
#             if 'wdis' in link.attrs['href']:
#                 wdis_urls.append("https://www.tastylive.com" + link.attrs['href'])
#     return wdis_urls
#
#
# def download_tastylive_audio():
#     wdis_urls = get_tastylive_urls()
#     download_audio(wdis_urls)
#     return


def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def coinbase_historical():
    # Plot charts using Coinbase API
    coins = ['btc', 'eth', 'dot', 'link', 'sol', 'ada']
    plot_charts(coins)


def binance_streaming():
    # Streaming price updates using Binance API
    one_symbol: List[str] = ["btcusdt@kline_1m"]
    two_symbols: List[str] = ["btcusdt@kline_1m", "ethusdt@kline_1m"]
    id: int = 1
    asyncio.run(run_websocket(two_symbols, id))


def supertrend(symbol = "BTC/USDT", start_date = "2023-08-01", interval = '4h'):
    exchange = ccxt.binance()
    run_supertrend_indicator(symbol, start_date, interval, exchange)


def undervalued_sentiment_screener():
    # Stock Screener and AI Sentiment Analysis
    undervalued = get_undervalued_stocks()
    print(undervalued)
    for ticker in undervalued:
        generate_csv(ticker)

####################################################################################################

# @app.route("/exchanges")
# def exchanges():
#     exchanges = ["binance", "bitstamp", "coinbase"]
#     return render_template("exchanges.html", exchanges=exchanges)


# @app.route("/eodhd")
# def exchanges_eodhd():
#     data_fetcher = EODHDAPIsDataFetcher(EODHD_API_TOKEN)
#     exchanges = data_fetcher.fetch_exchanges()
#     return render_template("exchanges.html", exchanges=exchanges)


# @app.route("/tasty")
# def tasty():
#     download_tastylive_audio()
#     exchanges = ["gemini", "bitmex", "ftx"]
#     return render_template("exchanges.html", exchanges=exchanges)


@app.route("/")
def run_api():
    return render_template("run_api.html", msg="")


def data_requests(form_data):
    # print(form_data)
    msg = ""
    ticker_symbol = form_data['ticker_symbol']
    start_date = form_data['start_date']
    end_date = None if form_data['end_date'].strip() == "" else form_data['end_date'].strip()
    if form_data['select_url'] == "/api_coinbase":
        coinbase_historical()
        msg = f"Retrieved Coinbase historical data for {ticker_symbol}"
    elif form_data['select_url'] == "/api_binance":
        binance_streaming()
        msg = f"Started Binance data streaming for {ticker_symbol}"
    return msg


def indicators1(form_data):
    # print(form_data)
    msg = ""
    ticker_symbol = form_data['ticker_symbol']
    start_date = form_data['start_date']
    end_date = None if form_data['end_date'].strip() == "" else form_data['end_date'].strip()
    if form_data['select_url'] == "/ind_adx":
        run_adx_indicator(ticker_symbol, start_date, end_date)
        msg=f"Generated ADX indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_bbands":
        run_bollinger_bands_indicator(ticker_symbol, start_date, end_date)
        msg=f"Generated Bollinger Bands indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_macd":
        run_macd_indicator(ticker_symbol, start_date, end_date)
        msg=f"Generated MACD indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_rsi":
        run_rsi_indicator(ticker_symbol, start_date, end_date)
        msg=f"Generated RSI indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_stoch":
        run_stochastic_indicator(ticker_symbol, start_date, end_date)
        msg=f"Generated Stochastic indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_kalman":
        run_kalman_filters(ticker_symbol, start_date, end_date)
        msg=f"Generated Kalman Filter indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_var":
        run_VaR_historical(ticker_symbol, start_date, end_date)
        msg=f"Generated Value at Risk indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_supertrend":
        supertrend(ticker_symbol, start_date, '4h')
        msg=f"Generated Supertrend indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_dyntimewarp":
        generate_time_warping_plots(ticker_symbol, start_date, end_date)
        msg=f"Generated Dynamic Time Warping indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_undervalued":
        undervalued_sentiment_screener()
        msg=f"Generated Undervalued Sentiment Screener indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_candleta":
        run_candlestick_ta_patterns(ticker_symbol)
        msg=f"Generated Candlestick TA indicators for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_zscore":
        run_z_score(ticker_symbol, start_date, end_date)
        msg=f"Generated Z-Score indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_zigzag":
        if not end_date:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2 * 365)
            end_date = end_date.strftime('%Y-%m-%d')
            start_date = start_date.strftime('%Y-%m-%d')
        run_zig_zag(ticker_symbol, start_date, end_date)
        msg=f"Generated Zig-Zag indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_pcadtw":
        run_pca_dtw(ticker_symbol, start_date, end_date)
        msg=f"Generated Principal Component Analysis / Dynamic Time Warping indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_supportresist":
        run_support_resistance(ticker_symbol, start_date, end_date)
        msg=f"Generated Z-Score indicator for {ticker_symbol}"
    return msg


def indicators2(form_data):
    # print(form_data)
    msg = ""
    ticker_symbol = form_data['ticker_symbol']
    start_date = form_data['start_date']
    end_date = None if form_data['end_date'].strip() == "" else form_data['end_date'].strip()
    if form_data['select_url'] == "/ind_atr":
        run_average_true_range(ticker_symbol, start_date, end_date)
        msg = f"Generated Average True Range indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_bollinger":
        run_bollinger_bands(ticker_symbol, start_date, end_date)
        msg = f"Generated Bollinger Bands indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_donchian":
        run_donchian_channels(ticker_symbol, start_date, end_date)
        msg = f"Generated Donchian Channels indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_keltner":
        run_keltner_channels(ticker_symbol, start_date, end_date)
        msg = f"Generated Keltner Channels indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_rvi":
        run_relative_volatility_index(ticker_symbol, start_date, end_date)
        msg = f"Generated Relative Volatility Index indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_chaikin":
        run_volatility_chaikin(ticker_symbol, start_date, end_date)
        msg = f"Generated Chaikin Money Flow indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_volnvi":
        run_volume_nvi(ticker_symbol, start_date, end_date)
        msg = f"Generated Negative Volume Index indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_volobv":
        run_volume_obv(ticker_symbol, start_date, end_date)
        msg = f"Generated On-Balance Volume indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_volvpt":
        run_volume_vpt(ticker_symbol, start_date, end_date)
        msg = f"Generated Volatility Price Trend indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_volvrc":
        run_volume_vroc(ticker_symbol, start_date, end_date)
        msg = f"Generated Volume Rate of Change indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_volcmf":
        run_volume_cmf(ticker_symbol, start_date, end_date)
        msg = f"Generated Chaikin Money Flow indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_volvwap":
        run_volume_vwap(ticker_symbol, start_date, end_date)
        msg = f"Generated Volume Weighted Average Price indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_voladline":
        run_volume_adline(ticker_symbol, start_date, end_date)
        msg = f"Generated Advance/Decline Line indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_volmfi":
        run_volume_mfi(ticker_symbol, start_date, end_date)
        msg = f"Generated Money Flow Index indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_volko":
        run_volume_klinger_oscillator(ticker_symbol, start_date, end_date)
        msg = f"Generated Klinger Oscillator indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_aroon":
        run_aroon_oscillator(ticker_symbol, start_date, end_date)
        msg = f"Generated Aroon Oscillator indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_demarker":
        plot_stock_with_demarker(ticker_symbol, start_date, end_date)
        msg = f"Generated  indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_relvigor":
        run_relative_vigor(ticker_symbol, start_date, end_date)
        msg = f"Generated Relative Vigor indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_threeway":
        run_threeway_average_crossover(ticker_symbol, start_date, end_date)
        msg = f"Generated Threeway Average Crossover indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_trendexhaust":
        generate_trend_exhaustion_plot(ticker_symbol, start_date, end_date)
        msg = f"Generated Trend Exhaustion indicator for {ticker_symbol}"
    elif form_data['select_url'] == "/ind_obscureind":
        run_obscure_indicators(ticker_symbol, start_date, end_date)
        msg = f"Generated other obscure indicators for {ticker_symbol}"
    return msg


def signals_and_strategies(form_data):
    # print(form_data)
    msg = ""
    ticker_symbol = form_data['ticker_symbol']
    start_date = form_data['start_date']
    end_date = None if form_data['end_date'].strip() == "" else form_data['end_date'].strip()
    if form_data['select_url'] == "/sig_sentiment":
        run_sentiment_analysis_signal(ticker_symbol)    # Trading signal based on sentiment analysis
        msg = f"Generated AI Sentiment Analysis signal for {ticker_symbol}"
    elif form_data['select_url'] == "/sig_pairs":
        assets = ['AAPL', 'MSFT', 'GOOG', 'NFLX', 'GME', 'AMC', 'ORCL', 'PARA']
        run_pairs_trading_signal(assets, start_date, end_date)
        msg = f"Generated Pairs Trading signal for {ticker_symbol}"
    elif form_data['select_url'] == "/strat_smacross":
        run_sma_crossover(ticker_symbol, start_date, end_date)
        msg = f"Generated SMA Crossover strategy for {ticker_symbol}"
    elif form_data['select_url'] == "/strat_momentum":
        run_naive_momentum(ticker_symbol, start_date, end_date)
        msg = f"Generated Naive Momentum strategy for {ticker_symbol}"
    elif form_data['select_url'] == "/strat_meanrevert":
        run_mean_reversion(ticker_symbol, start_date, end_date)
        msg = f"Generated Mean Reversion strategy for {ticker_symbol}"
    elif form_data['select_url'] == "/strat_dynrenko":
        run_dynamic_renko(ticker_symbol, start_date, '4h')
        msg = f"Generated Dynamic Renko strategy for {ticker_symbol}"
    elif form_data['select_url'] == "/strat_citadelpairs":
        run_citadel_pairs(start_date, end_date)
        msg = f"Generated Citadel Pairs strategy for {ticker_symbol}"
    elif form_data['select_url'] == "/strat_deeplearn":
        run_deep_learning_strategy_backtest(ticker_symbol, start_date)
        msg = f"Generated Deep Learning strategy for {ticker_symbol}"
    elif form_data['select_url'] == "/strat_macross":
        run_ma_crossover_strategy(ticker_symbol, start_date, end_date)
        msg = f"Generated MA Crossover strategy for {ticker_symbol}"
    return msg


def portfolio_risk(form_data):
    # print(form_data)
    msg = ""
    ticker_symbol = form_data['ticker_symbol']
    start_date = form_data['start_date']
    end_date = None if form_data['end_date'].strip() == "" else form_data['end_date'].strip()
    if form_data['select_url'] == "/risk_hrpcompare":
        run_hrp_compare_strategy(start_date, end_date)
        msg = f"Generated HRP Compare"
    elif form_data['select_url'] == "/risk_hrpport":
        run_hrp_portfolio(start_date, end_date)
        msg = f"Generated HRP Portfolio"
    elif form_data['select_url'] == "/risk_hrprebalance":
        run_hrp_rebalancing(start_date, end_date)
        msg = f"Generated HRP Rebalancing"
    elif form_data['select_url'] == "/risk_hrpparity":
        run_parity_rebalancing(start_date, end_date, n_days=30)
        msg = f"Generated HRP Parity Rebalancing"
    return msg


def predictions_and_forecasts(form_data):
    # print(form_data)
    msg = ""
    ticker_symbol = form_data['ticker_symbol']
    start_date = form_data['start_date']
    end_date = None if form_data['end_date'].strip() == "" else form_data['end_date'].strip()
    if form_data['select_url'] == "/pred_prophet":
        run_prophet_forecast(ticker_symbol, num_years=10)
        msg = f"Generated Prophet forecast for {ticker_symbol}"
    elif form_data['select_url'] == "/pred_xgboost":
        run_xgboost_forecast(ticker_symbol, start_date, end_date)
        msg = f"Generated XGBoost forecast for {ticker_symbol}"
    elif form_data['select_url'] == "/pred_timegpt":
        run_timegpt_forecast(ticker_symbol, start_date, end_date)
        msg = f"Generated TimeGPT forecast for {ticker_symbol}"
    elif form_data['select_url'] == "/pred_nhits":
        run_nhits_forecast()
        msg = f"Generated NHITS forecast"
    return msg


def research(form_data):
    # print(form_data)
    msg = ""
    ticker_symbol = form_data['ticker_symbol']
    start_date = form_data['start_date']
    end_date = None if form_data['end_date'].strip() == "" else form_data['end_date'].strip()
    if form_data['select_url'] == "/research_incstmt":
        run_analyze_income_statement(ticker_symbol)
        msg = f"Generated Income Statement analysis"
    elif form_data['select_url'] == "/research_finstmts":
        run_analyze_financial_statements(ticker_symbol, nyears=3)
        msg = f"Generated Financial Statements analysis"
    elif form_data['select_url'] == "/research_ratios":
        run_financial_ratios(ticker_symbol, nyears=4)
        msg = f"Generated Financial Ratios analysis"
    elif form_data['select_url'] == "/research_balsheet":
        run_balance_sheet(ticker_symbol)
        msg = f"Generated Balance Sheet analysis"
    return msg


@app.route('/run/<string:id>', methods=['POST', 'GET'])
def run_(id):
    msg_text = ""
    if request.method == 'GET':
        return f"This URL cannot be accessed directly."
    if request.method == 'POST':
        form_data = request.json
        # form_data = request.form
        print(form_data)
        if id == 'data':
            msg_text = data_requests(form_data)
        elif id == 'indicators1':
            msg_text = indicators1(form_data)
        elif id == 'indicators2':
            msg_text = indicators2(form_data)
        elif id == 'sigandstrat':
            msg_text = signals_and_strategies(form_data)
        elif id == 'risk':
            msg_text = portfolio_risk(form_data)
        elif id == 'predict':
            msg_text = predictions_and_forecasts(form_data)
        elif id == 'research':
            msg_text = research(form_data)
        return render_template("run_api.html", msg=msg_text, msgTime=currentTime())


####################################################################################################

if __name__ == '__main__':
    app.run()

