"""


def prophet_forecast():
    ticker = 'AAPL'
    run_prophet_forecast(ticker, num_years=10)


def xgboost_forecast():
    # XGBoost stock price forecast
    symbol = 'AMD'  # 'AAPL'
    startDate = '2022-04-01'
    endDate = None
    datasets = get_datasets(symbol, startDate, endDate)


def timegpt_forecast():
    ticker = "^GSPC"    # S&P 500 index
    start_date = '2016-01-01'
    end_date = '2024-06-14'
    run_timegpt_forecast(ticker, start_date, end_date)


def nhits_forecast():
    run_nhits_forecast()


def kalman_filter():
    # Kalman filter
    symbol = 'GOOG'  # 'AAPL'
    startDate = '2020-05-22'
    endDate = '2024-05-22'
    close_prices = download_close_prices(symbol, startDate, endDate)
    kalman_avg, spread, zscore = calc_kalman_filter(close_prices)
    # Define a range of thresholds to test
    long_entry_thresholds = np.arange(-5, -2, 1)
    short_entry_thresholds = np.arange(2, 5, 1)
    long_exit_thresholds = np.arange(-0.5, 0.5, 0.25)
    short_exit_thresholds = np.arange(-0.5, 0.5, 0.25)
    stop_loss_thresholds = np.arange(0.02, 0.05, 0.01)  # Stop-loss as a percentage of entry price
    best_sharpe_ratio, best_thresholds, results_df = get_best_thresholds(close_prices, kalman_avg, spread, zscore,
                                                                         long_entry_thresholds, short_entry_thresholds,
                                                                         long_exit_thresholds, short_exit_thresholds,
                                                                         stop_loss_thresholds)
    plot_cumulative_returns(symbol, close_prices, kalman_avg, spread, zscore, best_thresholds)


def zig_zag():
    ticker = 'TSLA'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365)
    run_zig_zag(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))


def dynamic_time_warping():
    ticker = "ASML.AS"
    start_date = '2000-01-01'
    end_date = '2023-07-21'
    generate_time_warping_plots(ticker, start_date, end_date)


def value_at_risk():
    # statistical - Value at Risk
    symbol = 'AMD'  #'AAPL'
    startDate = '2022-04-01'
    endDate = None
    returns, VaR_historical = get_VaR_historical(symbol, startDate, endDate)
    plot_VaR_historical(returns, VaR_historical)


def supertrend():
    # Supertrend Indicator
    symbol = "BTC/USDT"
    start_date = "2023-08-01"
    interval = '4h'
    exchange = ccxt.binance()
    run_strategy(symbol, start_date, interval, exchange)


def aroon_oscillator():
    ticker = "AFX.DE"
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    run_aroon_oscillator(ticker, start_date, end_date)


def demarker():
    plot_stock_with_demarker('JNJ', '2022-01-01', '2024-01-01')


def relative_vigor():
    run_relative_vigor('AAPL', '2020-01-01', None)


def threeway_avg_crossover():
    run_threeway_average_crossover('UNA.AS', '2020-01-01', '2024-03-26')


def trend_exhaustion():
    generate_trend_exhaustion_plot("AAPL", '2020-01-01', '2024-01-01')


def obscure_indicators():
    run_obscure_indicators("ASML.AS", '2020-01-01', '2024-01-01')


def average_true_range():
    run_average_true_range("AAPL", "2020-01-01", "2023-01-01")


def bollinger_bands():
    run_bollinger_bands("AAPL", "2020-01-01", "2023-01-01")


def donchian_channels():
    run_donchian_channels("AAPL", "2020-01-01", "2024-01-01")


def keltner_channels():
    run_keltner_channels("AAPL", "2020-01-01", "2024-01-01")


def relative_volatility_index():
    run_relative_volatility_index("AAPL", "2020-01-01", "2024-01-01")


def volatility_chaikin():
    run_volatility_chaikin("AAPL", "2020-01-01", "2024-01-01")


def z_score():
    run_z_score('ASML.AS', '2020-01-01', '2024-01-01')


def pairs_trading_signal():
    assets = ['AAPL', 'MSFT', 'GOOG', 'NFLX', 'GME', 'AMC', 'ORCL', 'PARA']
    start_date = '2019-01-01'
    end_date = None
    run_pairs_trading_signal(assets, start_date, end_date)


def candlestick_ta_patterns():
    run_candlestick_ta_patterns(symbol='SAP.DE')


def sentiment_signal():
    # Trading signal based on sentiment analysis
    symbol = 'GOOG'
    data = get_historical_sentiment(symbol)
    analyze_strategy(symbol, data)


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


@app.route("/research_incstmt")
def run_research_income_statement():
    run_analyze_income_statement("AAPL")
    return render_template("run_api.html", msg="Generated income statement analysis", msgTime=currentTime())


@app.route("/research_finstmts")
def run_research_financial_statements():
    run_analyze_financial_statements('AAPL', nyears=3)
    return render_template("run_api.html", msg="Generated financial statement analysis", msgTime=currentTime())


@app.route("/research_finratios")
def run_research_financial_ratios():
    run_financial_ratios('AAPL', 4)
    return render_template("run_api.html", msg="Generated financial ratios analysis", msgTime=currentTime())


@app.route("/research_balsheet")
def run_research_balance_sheet():
    run_balance_sheet("AAPL")
    return render_template("run_api.html", msg="Generated balance sheet analysis", msgTime=currentTime())


@app.route("/risk_hrpcompare")
def run_risk_hrp_compare():
    run_hrp_compare_strategy('2018-01-01', '2024-01-01')
    return render_template("run_api.html", msg="Generated HRP Compare portfolio risk", msgTime=currentTime())


@app.route("/risk_hrpport")
def run_risk_hrp_portfolio():
    run_hrp_portfolio('2020-01-01', '2024-01-01')
    return render_template("run_api.html", msg="Generated HRP portfolio risk", msgTime=currentTime())


@app.route("/risk_hrprebalance")
def run_risk_hrp_rebalance():
    run_hrp_rebalancing('2020-01-01', '2024-01-01')
    return render_template("run_api.html", msg="Generated HRP Rebalancing portfolio risk", msgTime=currentTime())


@app.route("/risk_hrpparity")
def run_risk_hrp_parity():
    run_parity_rebalancing("2010-01-01", "2023-01-01", n_days=30)
    return render_template("run_api.html", msg="Generated HRP Parity Rebalancing portfolio risk", msgTime=currentTime())


@app.route("/pred_prophet")
def run_pred_prophet():
    prophet_forecast()
    return render_template("run_api.html", msg="Generated Prophet forecast", msgTime=currentTime())


@app.route("/pred_xgboost")
def run_pred_xgboost():
    xgboost_forecast()
    return render_template("run_api.html", msg="Generated XGBoost forecast", msgTime=currentTime())


@app.route("/pred_timegpt")
def run_pred_timegpt():
    timegpt_forecast()
    return render_template("run_api.html", msg="Generated TimeGPT forecast", msgTime=currentTime())


@app.route("/pred_nhits")
def run_pred_nhits():
    nhits_forecast()
    return render_template("run_api.html", msg="Generated NHITS forecast", msgTime=currentTime())


@app.route("/api_coinbase")
def run_api_coinbase():
    coinbase_historical()
    return render_template("run_api.html", msg="Retrieved Coinbase historical data", msgTime=currentTime())


@app.route("/api_binance")
def run_api_binance():
    binance_streaming()
    return render_template("run_api.html", msg="Started Binance data streaming", msgTime=currentTime())


@app.route("/ind_adx")
def run_ind_adx():
    run_adx_indicator('AAPL', '2020-01-01', None)
    return render_template("run_api.html", msg="Started ADX indicator", msgTime=currentTime())


@app.route("/ind_bbands")
def run_ind_bbands():
    run_bollinger_bands_indicator('AAPL', '2020-01-01', None)
    return render_template("run_api.html", msg="Started Bollinger Bands indicator", msgTime=currentTime())


@app.route("/ind_macd")
def run_ind_macd():
    run_macd_indicator('AAPL', '2020-01-01', None)
    return render_template("run_api.html", msg="Started MACD indicator", msgTime=currentTime())


@app.route("/ind_rsi")
def run_ind_rsi():
    run_rsi_indicator('AAPL', '2020-01-01', None)
    return render_template("run_api.html", msg="Started RSI indicator", msgTime=currentTime())


@app.route("/ind_stoch")
def run_ind_stoch():
    run_stochastic_indicator('AAPL', '2020-01-01', None)
    return render_template("run_api.html", msg="Started Stochastic indicator", msgTime=currentTime())


@app.route("/ind_kalman")
def run_ind_kalman_filter():
    kalman_filter()
    return render_template("run_api.html", msg="Generated Kalman Filter indicator", msgTime=currentTime())


@app.route("/ind_var")
def run_ind_value_at_risk():
    value_at_risk()
    return render_template("run_api.html", msg="Generated Value at Risk indicator", msgTime=currentTime())


@app.route("/ind_supertrend")
def run_ind_supertrend():
    supertrend()
    return render_template("run_api.html", msg="Generated Supertrend indicator", msgTime=currentTime())


@app.route("/ind_aroon")
def run_ind_aroon():
    aroon_oscillator()
    return render_template("run_api.html", msg="Generated Aroon Oscillator indicator", msgTime=currentTime())


@app.route("/ind_demarker")
def run_ind_demarker():
    demarker()
    return render_template("run_api.html", msg="Generated Aroon Oscillator indicator", msgTime=currentTime())


@app.route("/ind_relvigor")
def run_ind_relative_vigor():
    relative_vigor()
    return render_template("run_api.html", msg="Generated Relative Vigor indicator", msgTime=currentTime())


@app.route("/ind_threeway")
def run_ind_threeway_avg_crossover():
    threeway_avg_crossover()
    return render_template("run_api.html", msg="Generated Three-way Average Crossover indicator", msgTime=currentTime())


@app.route("/ind_trendexhaust")
def run_ind_trend_exhaustion():
    trend_exhaustion()
    return render_template("run_api.html", msg="Generated Trend Exhaustion indicator", msgTime=currentTime())


@app.route("/ind_obscureind")
def run_ind_obscure():
    obscure_indicators()
    return render_template("run_api.html", msg="Generated other 'obscure' indicator", msgTime=currentTime())


@app.route("/ind_atr")
def run_ind_atr():
    average_true_range()
    return render_template("run_api.html", msg="Generated Average True Range indicator", msgTime=currentTime())


@app.route("/ind_bollinger")
def run_ind_bollinger():
    bollinger_bands()
    return render_template("run_api.html", msg="Generated Bollinger Bands indicator", msgTime=currentTime())


@app.route("/ind_donchian")
def run_ind_donchian():
    donchian_channels()
    return render_template("run_api.html", msg="Generated Donchian Channels indicator", msgTime=currentTime())


@app.route("/ind_keltner")
def run_ind_keltner():
    keltner_channels()
    return render_template("run_api.html", msg="Generated Keltner Channels indicator", msgTime=currentTime())


@app.route("/ind_rvi")
def run_ind_rvi():
    relative_volatility_index()
    return render_template("run_api.html", msg="Generated Relative Volatility Index indicator", msgTime=currentTime())


@app.route("/ind_chaikin")
def run_ind_chaikin():
    volatility_chaikin()
    return render_template("run_api.html", msg="Generated Volatility Chaikin indicator", msgTime=currentTime())


@app.route("/ind_zscore")
def run_ind_zscore():
    z_score()
    return render_template("run_api.html", msg="Generated Z-Score indicator", msgTime=currentTime())


@app.route("/ind_zigzag")
def run_ind_zigzag():
    zig_zag()
    return render_template("run_api.html", msg="Generated Zig-Zag indicator", msgTime=currentTime())


@app.route("/ind_volnvi")
def run_ind_volnvi():
    run_volume_nvi('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated Negative Volume Index volume indicator", msgTime=currentTime())


@app.route("/ind_volobv")
def run_ind_volobv():
    run_volume_obv('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated On-Balance Volume volume indicator", msgTime=currentTime())


@app.route("/ind_volvpt")
def run_ind_volvpt():
    run_volume_vpt('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated Volume Price Trend volume indicator", msgTime=currentTime())


@app.route("/ind_volvroc")
def run_ind_volvroc():
    run_volume_vroc('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated Volume Rate of Change volume indicator", msgTime=currentTime())


@app.route("/ind_volcmf")
def run_ind_volcmf():
    run_volume_cmf('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated Chaikin Money Flow volume indicator", msgTime=currentTime())


@app.route("/ind_volvwap")
def run_ind_volvwap():
    run_volume_vwap('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated Volume-Weighted Average Price volume indicator", msgTime=currentTime())


@app.route("/ind_voladline")
def run_ind_voladline():
    run_volume_adline('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated Accumulation/Distribution Line volume indicator", msgTime=currentTime())


@app.route("/ind_volmfi")
def run_ind_volmfi():
    run_volume_mfi('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated Money Flow Index volume indicator", msgTime=currentTime())


@app.route("/ind_volko")
def run_ind_volko():
    run_volume_klinger_oscillator('TSLA', '2020-01-01', '2024-06-14')
    return render_template("run_api.html", msg="Generated Klinger Oscillator volume indicator", msgTime=currentTime())


@app.route("/ind_dtw")
def run_ind_dtw():
    dynamic_time_warping()
    return render_template("run_api.html", msg="Generated Dynamic Time Warping indicator", msgTime=currentTime())


@app.route("/ind_undervalued")
def run_ind_undervalued_sentiment():
    undervalued_sentiment_screener()
    return render_template("run_api.html", msg="Generated Undervalued Sentiment Screener indicator", msgTime=currentTime())


@app.route("/ind_candlestick")
def run_ind_candlestick_patterns():
    candlestick_ta_patterns()
    return render_template("run_api.html", msg="Generated Candlestick TA indicators", msgTime=currentTime())


@app.route("/ind_supportresist")
def run_ind_support_resistance():
    run_support_resistance('AAPL', '2023-01-01', None)
    return render_template("run_api.html", msg="Generated Support/Resistance indicators", msgTime=currentTime())


@app.route("/ind_pcadtw")
def run_ind_pca_dtw():
    run_pca_dtw('AAPL', '2023-01-01', '2024-06-17')
    return render_template("run_api.html", msg="Generated Principle Component Analysis / Dynamic Time Warping", msgTime=currentTime())


@app.route("/sig_sentiment")
def run_sig_sentiment():
    sentiment_signal()
    return render_template("run_api.html", msg="Generated Sentiment signal", msgTime=currentTime())


@app.route("/sig_pairs")
def run_sig_pairs_trading():
    pairs_trading_signal()
    return render_template("run_api.html", msg="Generated Pairs Trading signal", msgTime=currentTime())


@app.route("/strat_smacross")
def run_strat_sma_crossover():
    # sma_crossover_strategy()
    run_sma_crossover('AAPL', '2021-01-01', '2023-01-01')
    return render_template("run_api.html", msg="Generated SMA Crossover strategy", msgTime=currentTime())


@app.route("/strat_momentum")
def run_strat_naive_momentum():
    run_naive_momentum('AAPL', '2021-01-01', '2023-01-01')
    return render_template("run_api.html", msg="Generated Naive Momentum strategy", msgTime=currentTime())


@app.route("/strat_meanrevert")
def run_strat_mean_reversion():
    run_mean_reversion('AAPL', '2021-01-01', '2023-01-01')
    return render_template("run_api.html", msg="Generated Mean Reversion strategy", msgTime=currentTime())


@app.route("/strat_dynrenko")
def run_strat_dynamic_renko():
    # exchange = ccxt.binance()
    run_dynamic_renko("ETH/USDT", "2022-12-1", '4h')
    return render_template("run_api.html", msg="Generated Dynamic Renko strategy", msgTime=currentTime())


@app.route("/strat_citadelpairs")
def run_strat_citadel_pairs():
    run_citadel_pairs(start_date='2021-01-01', end_date='2023-10-31')
    return render_template("run_api.html", msg="Generated Citadel Pairs-Trading strategy", msgTime=currentTime())


@app.route("/strat_deeplearn")
def run_strat_deep_learning_simple():
    run_deep_learning_strategy_backtest('ETH/USDT', '2022-01-01', interval='1d', risk_free_rate=0.01)
    return render_template("run_api.html", msg="Generated Citadel Pairs-Trading strategy", msgTime=currentTime())


@app.route("/strat_macross")
def run_strat_ma_crossover():
    run_ma_crossover_strategy('AAPL', '2020-01-01', None, short_window=20, long_window=50)
    return render_template("run_api.html", msg="Generated MA Crossover strategy", msgTime=currentTime())



"""



# @app.route('/data/', methods=['POST', 'GET'])
# def data():
#     if request.method == 'GET':
#         return f"The URL /data is accessed directly. Try going to '/form' to submit form"
#     if request.method == 'POST':
#         form_data = request.form
#         return render_template('data.html', form_data=form_data)

# class MyForm(FlaskForm):
#     my_field = StringField("Email", validators=[DataRequired(), Email()])
#     my_sf = StringField("My Text")
#     my_submit = SubmitField("Submit My Form")
#     my_taf = TextAreaField()
#     my_bf = BooleanField()
#     my_rf = RadioField()
#
#
# @app.route('/', methods=["GET", "POST"])
# def my_route():
#     my_form = MyForm()
#     if my_form.validate_on_submit():
#         my_data = my_form.my_field.data

# # get form data and send to route_two with redirect
# @app.route("/1")
# def route_one():
#     my_form = MyForm()
#     if my_form.validate_on_submit():
#         my_data = my_form.my_field.data
#         return redirect(url_for("route_two", data=my_data))
#     return render_template("one.html", template_form=my_form)
#
#
# # render template with some data
# @app.route("/2/<data>")
# def route_two(data):
#     return render_template("two.html", template_data=data)


# elif form_data['select_url'] == "/ind_aroon":
#     run_aroon_oscillator(ticker_symbol, start_date, end_date)
#     msg=f"Generated Aroon Oscillator indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_demarker":
#     plot_stock_with_demarker(ticker_symbol, start_date, end_date)
#     msg=f"Generated Demarker indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_relvigor":
#     run_relative_vigor(ticker_symbol, start_date, end_date)
#     msg=f"Generated Relative Vigor indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_threeway":
#     run_threeway_average_crossover(ticker_symbol, start_date, end_date)
#     msg=f"Generated Three-way Average Crossover indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_trendexhaust":
#     generate_trend_exhaustion_plot(ticker_symbol, start_date, end_date)
#     msg=f"Generated Trend Exhaustion indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_obscureind":
#     run_obscure_indicators(ticker_symbol, start_date, end_date)
#     msg=f"Generated other 'obscure' indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_atr":
#     run_average_true_range(ticker_symbol, start_date, end_date)
#     msg=f"Generated Average True Range indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_bollinger":
#     run_bollinger_bands(ticker_symbol, start_date, end_date)
#     msg=f"Generated Bollinger Bands indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_donchian":
#     run_donchian_channels(ticker_symbol, start_date, end_date)
#     msg=f"Generated Donchian Channels indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_keltner":
#     run_keltner_channels(ticker_symbol, start_date, end_date)
#     msg=f"Generated Keltner Channels indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_rvi":
#     run_relative_volatility_index(ticker_symbol, start_date, end_date)
#     msg=f"Generated Relative Volatility Index indicator for {ticker_symbol}"
# elif form_data['select_url'] == "/ind_chaikin":
#     run_volatility_chaikin(ticker_symbol, start_date, end_date)
#     msg=f"Generated Volatility Chaikin indicator for {ticker_symbol}"


"""
<!--<form action="/run/ind_kalman" method = "POST">
    <grid>
        <tr>
            <th>Ticker Symbol</th>
            <th>Start Date</th>
            <th>End Date</th>
        </tr>
        <tr>
            <td><input type = "text" name = "ticker_symbol" value="AAPL" /></td>
            <td><input type = "text" name = "start_date" /></td>
            <td><input type = "text" name = "end_date" /></td>
        </tr>
    </grid>
    <input type="hidden" id="open_url" name="open_url" value="/ind_kalman" />
    <p><input type="submit" value="Kalman Filter" /></p>
</form>-->
"""

"""
## views.py:

@app.route ('/export_pdf', methods = ['GET', 'POST'])
def export_pdf():
    form = ExportPDF()
    if form.validate_on_submit():
      try:
        export_pdfs.main_program(form.account_url.data,
          form.api_token.data)
        flash ('PDFs exported')
        return redirect(url_for('export_pdf'))
      except TransportException as e:
        s = e.content
        result = re.search('<error>(.*)</error>', s)
        flash('There was an authentication error: ' + result.group(1))
      except FailedRequest as e:
        flash('There was an error: ' + e.error)
    return render_template('export_pdf.html', title = 'Export PDFs', form = form)

## export_pdf.html:

{% extends "base.html" %}

{% block content %}
{% include 'flash.html' %}
<div class="well well-sm">
  <h3>Export PDFs</h3>
  <form class="navbar-form navbar-left" action="" method ="post" name="receipt">
    {{form.hidden_tag()}}
    <br>
    <div class="control-group{% if form.errors.account_url %} error{% endif %}">
      <label class"control-label" for="account_url">Enter Account URL:</label>
      <div class="controls">
        {{ form.account_url(size = 50, class = "span4")}}
        {% for error in form.errors.account_url %}
          <span class="help-inline">[{{error}}]</span><br>
        {% endfor %}
      </div>
    </div>
    <br>
    <div class="control-group{% if form.errors.api_token %} error{% endif %}">
      <label class"control-label" for="api_token">Enter API Token:</label>
      <div class="controls">
        {{ form.api_token(size = 50, class = "span4")}}
        {% for error in form.errors.api_token %}
          <span class="help-inline">[{{error}}]</span><br>
        {% endfor %}
      </div>
    </div>
    <br>
    <button type="submit" class="btn btn-primary btn-lg">Submit</button>
  <br>
  <br>
  <div class="progress progress-striped active">
  <div class="progress-bar"  role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%">
    <span class="sr-only"></span>
  </div>
</form>
</div>
</div>
{% endblock %}

## and export_pdfs.py:

def main_program(url, token):
    api_caller = api.TokenClient(url, token)
    path = os.path.expanduser('~/Desktop/'+url+'_pdfs/')
    pdfs = list_all(api_caller.pdf.list, 'pdf')
    total = 0
    count = 1
    for pdf in pdfs:
        total = total + 1
    for pdf in pdfs:
        header, body = api_caller.getPDF(pdf_id=int(pdf.pdf_id))
        with open('%s.pdf' % (pdf.number), 'wb') as f:
          f.write(body)
        count = count + 1
        if count % 50 == 0:
          time.sleep(1)

## In that last function I have total the number of PDFs I will export, and have an ongoing count while it is processing. How can I send the current progress to my .html file to fit within the 'style=' tag of the progress bar? Preferably in a way that I can reuse the same tool for progress bars on other pages. Let me know if I haven't provided enough info.

## As some others suggested in the comments, the simplest solution is to run your exporting function in another thread, and let your client pull progress information with another request. There are multiple approaches to handle this particular task. Depending on your needs, you might opt for a more or less sophisticated one.

## Here's a very (very) minimal example on how to do it with threads:

import random
import threading
import time

from flask import Flask


class ExportingThread(threading.Thread):
    def __init__(self):
        self.progress = 0
        super().__init__()

    def run(self):
        # Your exporting stuff goes here ...
        for _ in range(10):
            time.sleep(1)
            self.progress += 10


exporting_threads = {}
app = Flask(__name__)
app.debug = True


@app.route('/')
def index():
    global exporting_threads

    thread_id = random.randint(0, 10000)
    exporting_threads[thread_id] = ExportingThread()
    exporting_threads[thread_id].start()

    return 'task id: #%s' % thread_id


@app.route('/progress/<int:thread_id>')
def progress(thread_id):
    global exporting_threads

    return str(exporting_threads[thread_id].progress)


if __name__ == '__main__':
    app.run()

## In the index route (/) we spawn a thread for each exporting task, and we return an ID to that task so that the client can retrieve it later with the progress route (/progress/[exporting_thread]). The exporting thread updates its progress value every time it thinks it is appropriate.

## On the client side, you would get something like this (this example uses jQuery):

function check_progress(task_id, progress_bar) {
    function worker() {
        $.get('progress/' + task_id, function(data) {
            if (progress < 100) {
                progress_bar.set_progress(progress)
                setTimeout(worker, 1000)
            }
        })
    }
}

## As said, this example is very minimalistic and you should probably go for a slightly more sophisticated approach. Usually, we would store the progress of a particular thread in a database or a cache of some sort, so that we don't rely on a shared structure, hence avoiding most of the memory and concurrency issues my example has.

## Redis (https://redis.io) is an in-memory database store that is generally well-suited for this kind of tasks. It integrates ver nicely with Python (https://pypi.python.org/pypi/redis).


## Python

from flask import Flask, render_template
from threading import Thread
from time import sleep
import json

app = Flask(__name__)
status = None

def task():
  global status
  for i in range(1,11):
    status = i
    sleep(1)

@app.route('/')
def index():
  t1 = Thread(target=task)
  t1.start()
  return render_template('index.html')
  
@app.route('/status', methods=['GET'])
def getStatus():
  statusList = {'status':status}
  return json.dumps(statusList)

if __name__ == '__main__':
  app.run(debug=True)
  
## HTML CSS JS

<!doctype html>
<html>

<head>
  <meta charset="UTF-8">

  <style>
  
  body {
    background-color: #D64F2A;
  }
  
  .progress {
    display: flex;
    position: absolute;
    height: 100%;
    width: 100%;
  }
  
  .status {
    color: white;
    margin: auto;
  }

  .status h2 {
    padding: 50px;
    font-size: 80px;
    font-weight: bold;
  }
  
  </style>

  <title>Status Update</title>

</head>

<body>
  <div class="progress">
    <div class="status">
      <h2 id="innerStatus">Loading...</h2>
    </div>
  </div>
</body>

<script>
var timeout;

async function getStatus() {

  let get;
  
  try {
    const res = await fetch("/status");
    get = await res.json();
  } catch (e) {
    console.error("Error: ", e);
  }
  
  document.getElementById("innerStatus").innerHTML = get.status * 10 + "&percnt;";
  
  if (get.status == 10){
    document.getElementById("innerStatus").innerHTML += " Done.";
    clearTimeout(timeout);
    return false;
  }
   
  timeout = setTimeout(getStatus, 1000);
}

getStatus();
</script>

</html>
"""


# I run this simple but educational Flask SSE implementation on localhost. To handle 3rd party (user uploaded) library in GAE:
#
# Create a directory named lib in your root path.
# copy gevent library directory to lib directory.
# Add these lines to your main.py:
#
# import sys
# sys.path.insert(0,'lib')
# Thats all. If you use lib directory from a child folder, use relative reference: sys.path.insert(0, ../../blablabla/lib')
# From http://flask.pocoo.org/snippets/116/
#
# # author: oskar.blom@gmail.com
# #
# # Make sure your gevent version is >= 1.0
# import gevent
# from gevent.wsgi import WSGIServer
# from gevent.queue import Queue
#
# from flask import Flask, Response
#
# import time
#
#
# # SSE "protocol" is described here: http://mzl.la/UPFyxY
# class ServerSentEvent(object):
#
#     def __init__(self, data):
#         self.data = data
#         self.event = None
#         self.id = None
#         self.desc_map = {
#             self.data : "data",
#             self.event : "event",
#             self.id : "id"
#         }
#
#     def encode(self):
#         if not self.data:
#             return ""
#         lines = ["%s: %s" % (v, k)
#                  for k, v in self.desc_map.iteritems() if k]
#
#         return "%s\n\n" % "\n".join(lines)
#
# app = Flask(__name__)
# subscriptions = []
#
# # Client code consumes like this.
# @app.route("/")
# def index():
#     debug_template = """
#      <html>
#        <head>
#        </head>
#        <body>
#          <h1>Server sent events</h1>
#          <div id="event"></div>
#          <script type="text/javascript">
#
#          var eventOutputContainer = document.getElementById("event");
#          var evtSrc = new EventSource("/subscribe");
#
#          evtSrc.onmessage = function(e) {
#              console.log(e.data);
#              eventOutputContainer.innerHTML = e.data;
#          };
#
#          </script>
#        </body>
#      </html>
#     """
#     return(debug_template)
#
# @app.route("/debug")
# def debug():
#     return "Currently %d subscriptions" % len(subscriptions)
#
# @app.route("/publish")
# def publish():
#     #Dummy data - pick up from request for real data
#     def notify():
#         msg = str(time.time())
#         for sub in subscriptions[:]:
#             sub.put(msg)
#
#     gevent.spawn(notify)
#
#     return "OK"
#
# @app.route("/subscribe")
# def subscribe():
#     def gen():
#         q = Queue()
#         subscriptions.append(q)
#         try:
#             while True:
#                 result = q.get()
#                 ev = ServerSentEvent(str(result))
#                 yield ev.encode()
#         except GeneratorExit: # Or maybe use flask signals
#             subscriptions.remove(q)
#
#     return Response(gen(), mimetype="text/event-stream")
#
# if __name__ == "__main__":
#     app.debug = True
#     server = WSGIServer(("", 5000), app)
#     server.serve_forever()
#     # Then visit http://localhost:5000 to subscribe
#     # and send messages by visiting http://localhost:5000/publish



