import numpy as np

from datetime import date, timedelta
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

from src.Kalman import kalman
from src.DataRepository import DataRepository
from src.Window import Window
from src.DataRepository import Universes
from src.Features import Features
from src.Tickers import SnpTickers, EtfTickers

lookback_win_len = 150
adflag = int((lookback_win_len/2) - 3)
win = Window(window_start=date(2009, 10, 6), trading_win_len=timedelta(days=150), repository=DataRepository())

#clusterer = Clusterer()
#clusters = clusterer.dbscan(eps=0.1, min_samples=2, window=win)

#c = Cointegrator(repository=DataRepository(), adf_confidence_level=AdfPrecisions.ONE_PCT, entry_z=0.1, exit_z=0.2)
#cointegrated_pairs = c.parallel_generate_pairs(clustering_results=clusters,  current_window=win,
                                                   #hurst_threshold=0.1, zero_crossings_ratio=8, adf_lag=adflag)

cc = [[SnpTickers.HAS, EtfTickers.FXN], [SnpTickers.PCAR, EtfTickers.WOOD], [SnpTickers.LB, EtfTickers.KOL],
    [SnpTickers.PFG, EtfTickers.IGF], [SnpTickers.TMUS, EtfTickers.FXR],  [SnpTickers.PSA, EtfTickers.PBD],
    [SnpTickers.SPG, EtfTickers.FAN], [SnpTickers.AIZ, EtfTickers.SMOG], [SnpTickers.BLK, EtfTickers.BJK],
    [SnpTickers.PSA, EtfTickers.SMOG], [SnpTickers.MAS, EtfTickers.VDE], [SnpTickers.CVS, EtfTickers.USD]]

print(win.window_start)
print(win.window_end)
win.roll_forward_one_day()
end_date = win.get_nth_working_day_ahead(date(2010, 5, 11), 60)
print(end_date)


for pair in cc:
    start_date = date(2009, 10, 9)
    end_date = win.get_nth_working_day_ahead(date(2010, 10, 3), 60)
    print(end_date)
    kalman(start_date=start_date, end_date=end_date, window=win,
            snp_ticker=pair[0], etf_ticker=pair[1])

win.roll_forward_one_day()

    #for pair in cointegrated_pairs[1]:
        #print("Pair ", pair.pair, "Beta: ", round(float(pair.beta), 3), "Intercept: ",
             # round(float(pair.intercept), 3), "Z score: ", round(float(pair.z_score), 5), "t stat: ",
              #round(float(pair.adf_tstat), 5))

