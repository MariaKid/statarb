import numpy as np
import pandas as pd
from pandas import DataFrame


def get_performance_stats(prices: DataFrame, risk_free_rate: float):
	"""
	Returns dataframe with strategy performance statistics
	"""
	# Pre-allocate empty dataframe for performance statistics
	stats = pd.DataFrame(index=[0])
	ret = prices.pct_change()

	last_index = prices.shape[0] - 1  # -1 since referencing starts from 0
	p_last = prices[last_index]
	stats['Tot Ret'] = (p_last - prices[0]) / prices[0]

	bus_days = 252
	stats['Avg Ret'] = ret.mean() * bus_days
	stats['rfr'] = risk_free_rate  # .mean() * 365
	stats['Std Dev'] = ret.std() * np.sqrt(bus_days)

	stats['SR'] = (stats['Avg Ret'] - stats['rfr']) / stats['Std Dev']

	stats['Skew'] = ret.skew()
	stats['Kurtosis'] = ret.kurtosis()
	stats['HWM'] = prices.max()

	hwm_time = prices.idxmax()
	stats['HWM date'] = hwm_time.date()

	# get all drawdowns (difference between cumulative max price and current price)
	dd = prices.cummax() - prices
	# end date of maximum drawdown
	end_mdd = dd.idxmax()
	start_mdd = prices[:end_mdd].idxmax()

	# Maximum Drawdown: positive proportional loss from peak
	stats['MDD'] = 1 - prices[end_mdd]/prices[start_mdd]
	stats['Peak date'] = start_mdd.date()
	stats['Trough date'] = end_mdd.date()

	"""
	bool_p = prices[end_mdd:] > prices[start_mdd]
	# Wrong, bool type has no max attribute
	if bool_p.idxmax().date() > bool_p.idxmin().date():

		stats['Rec date'] = bool_p.idxmax().date()
		stats['MDD dur'] = (stats['Rec date'] - stats['Peak date'])[0].days
	else:
		stats['Rec date'] = stats['MDD dur'] = 'Yet to recover'
	"""

	return stats.T
