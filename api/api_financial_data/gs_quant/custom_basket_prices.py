
from gs_quant.data import Dataset, Fields
from datetime import date

# https://developer.gs.com/docs/gsquant/


basket_ds = Dataset('GSCB_FLAGSHIP')
start_date = date(2007,1,1)

vip_px = basket_ds.get_data_series(Fields.CLOSE_PRICE, start=start_date, ticker='GSTHHVIP')
vip_px.tail()




"""
import gs_quant.timeseries as ts
from gs_quant.timeseries import Window

x = ts.generate_series(1000)           # Generate random timeseries with 1000 observations
vol = ts.volatility(x, Window(22, 0))  # Compute realized volatility using a window of 22 and a ramp up value of 0
vol.tail()                             # Show last few values
"""