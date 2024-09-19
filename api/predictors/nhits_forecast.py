
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
import quandl
import pandas as pd

# https://medium.com/modern-ai/the-most-powerful-time-series-algorithm-or-how-to-forecast-stocks-in-2024-f054189398a4
# https://fred.stlouisfed.org/series/CPALTT01USM661S
# https://www.oecd-ilibrary.org/economics/data/main-economic-indicators_mei-data-en


def generate_predictions():
    dataset_code = "MULTPL/SP500_PE_RATIO_MONTH"
    data = quandl.get(dataset_code)

    # loading data into a Darts TimeSeries object
    # covariates are the columns with the variables we believe will influence our target
    # target in this case is the y_change column with the monthly change in SP500 pricess
    covariates = TimeSeries.from_dataframe(df, time_col='date_time', value_cols=['Rates', 'CPI', 'Emp', 'pe_ratio'])
    target = TimeSeries.from_dataframe(df, time_col='date_time', value_cols=['y_change'])

    # the time window is 24 so 2 years in this case and will try to predict 12 months ahead
    past = 24
    future = 12

    # always scale the data
    cov_scaler = Scaler()
    target_scaler = Scaler()
    scaled_target = target_scaler.fit_transform(target)
    scaled_covariates = cov_scaler.fit_transform(covariates)

    # splitting data into train (80% of data) and test sets (20% of data)
    y_train, y_test = target[:655], target[655:]
    X_train, X_test = covariates[:655], covariates[655-past:]

    # initialize and train the NHiTSModel
    model = NHiTSModel(input_chunk_length=past, output_chunk_length=future, n_epochs=50)

    # fit the model with target and past covariates
    model.fit(y_train, past_covariates=X_train)

    # forecasting future n months (164 months in this case) using the trained model
    prediction = model.predict(n=164, past_covariates=X_test)

    return y_test, prediction


def plot_predictions(y_test, prediction):
    # plot predictions vs real values
    y_test.plot(label='Real Values')
    prediction.plot(label='N-HiTS')


def evaluate_predictions(y_test, prediction):
    # create a dataframe with results
    results = pd.DataFrame(y_test[1:].pd_series().values, columns=['Real'])
    results['Preds'] = prediction.pd_series().values

    # calculate the summation of cases where prediction and test have same sign
    sum(results['Preds'] * results['Real'] > 0)


def run_nhits_forecast():
    y_test, prediction = generate_predictions()
    plot_predictions(y_test, prediction)
    evaluate_predictions(y_test, prediction)


