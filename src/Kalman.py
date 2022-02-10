import numpy as np
import pandas as pd

from datetime import date, timedelta
from matplotlib import pyplot as plt
from pykalman import KalmanFilter

from src.DataRepository import DataRepository
from src.Window import Window
from src.DataRepository import Universes
from src.Features import Features
from src.Tickers import SnpTickers, EtfTickers

win = Window(window_start=date(2008, 10, 3), trading_win_len=timedelta(days=360), repository=DataRepository())


def kalman(start_date: date, end_date: date, window: Window, snp_ticker: SnpTickers,
           etf_ticker: EtfTickers):

    snp = window.repository.all_data.get(Universes.SNP)
    etf = window.repository.all_data.get(Universes.ETFs)

    snp = snp.loc[start_date:end_date, pd.IndexSlice[snp_ticker, Features.CLOSE]]
    etf = etf.loc[start_date:end_date:, pd.IndexSlice[etf_ticker, Features.CLOSE]]

    plot_index = snp.index[1:]

    snp = np.array(np.log(snp))
    etf = np.array(np.log(etf))

    means_trace = []
    covs_trace = []
    spreads = []
    spreads_std = []
    step = 0

    x = etf[step]
    y = snp[step]

    observation_matrix_stepwise = np.array([[x, 1]])  # Ft
    observation_stepwise = y  # yt
    observation_cov = 0.001  # vt
    state_cov_multiplier = 0.0001  # wt

    kf = KalmanFilter(n_dim_obs=1,
                      n_dim_state=2,
                      initial_state_mean=np.ones(2),  # θt-1
                      initial_state_covariance=np.ones((2, 2)),  # covθt
                      transition_matrices=np.eye(2),  # Gt
                      transition_covariance=np.eye(2)*state_cov_multiplier,  # wt
                      observation_covariance=observation_cov,  # vt
                      observation_matrices=observation_matrix_stepwise)  # Ft

    P = np.ones((2, 2)) + np.eye(2)*state_cov_multiplier
    spread_std = np.sqrt(observation_matrix_stepwise.dot(P).dot(observation_matrix_stepwise.transpose())[0][0] + observation_cov)
    spread = y - observation_matrix_stepwise.dot(np.ones(2))

    spreads.append(spread)
    spreads_std.append(spread_std)

    # updates θt based on newest observation
    state_means_stepwise, state_covs_stepwise = kf.filter(observation_stepwise)

    means_trace.append(state_means_stepwise[0])
    covs_trace.append(state_covs_stepwise[0])

    for step in range(1, len(snp)):
        x = etf[step]
        y = snp[step]
        observation_matrix_stepwise = np.array([[x, 1]])
        observation_stepwise = y

        state_means_stepwise, state_covs_stepwise = kf.filter_update(
            means_trace[-1], covs_trace[-1],
            observation=observation_stepwise,
            observation_matrix=observation_matrix_stepwise)

        P = covs_trace[-1] + np.eye(2)*state_cov_multiplier
        spread = y - observation_matrix_stepwise.dot(means_trace[-1])[0]
        spread_std = np.sqrt(observation_matrix_stepwise.dot(P).dot(observation_matrix_stepwise.transpose())[0][0] + observation_cov)
        means_trace.append(state_means_stepwise.data)
        covs_trace.append(state_covs_stepwise)
        spreads.append(spread)
        spreads_std.append(spread_std)

    spreads_std = spreads_std[1:]
    spreads = spreads[1:]

    slopes = []
    intercepts = []
    for i in means_trace[1:]:
        slopes.append(i[0])
        intercepts.append(i[1])

    pd.DataFrame(
        dict(
            spreads=spreads,
            slope=slopes,
            intercept=intercepts,
            spreads_std=spreads_std
        ), index=plot_index
    ).plot(subplots=True)
    plt.show()

    return means_trace, covs_trace, spreads, spreads_std
