import numpy as np
import queue as q
import multiprocessing as mp
import statistics as stat
import warnings
import pandas as pd

from pykalman import KalmanFilter

from src.Window import Window
from src.DataRepository import Universes
from src.Features import Features

warnings.filterwarnings("ignore")


def load_q(current_cointegrated_pairs):
    manager = mp.Manager()
    queue = manager.Queue()
    for pair in current_cointegrated_pairs:
        queue.put(pair)
    return queue


def parallel_kalman(current_cointegrated_pairs, window: Window, entry_z: float):
    num_processes = mp.cpu_count()
    processes = []
    pair_queue = load_q(current_cointegrated_pairs)
    results_queue = mp.Manager().Queue()

    for i in range(num_processes):
        new_process = mp.Process(target=kalman, args=(window, pair_queue, results_queue, entry_z))
        processes.append(new_process)
        new_process.start()
    for p in processes:
        p.join()

    selected_cointegrated_pairs = []
    for k in range(results_queue.qsize()):
        selected_cointegrated_pairs.append(results_queue.get())

    return selected_cointegrated_pairs


def kalman(window: Window, test_queue, results_queue, entry_z):

    while not test_queue.empty():
        try:
            pair = test_queue.get(timeout=1)
        except q.Empty:
            break

        try:
            snp = window.get_data(universe=Universes.SNP, tickers=[pair[0][0]], features=[Features.CLOSE])
            etf = window.get_data(universe=Universes.ETFs, tickers=[pair[0][1]], features=[Features.CLOSE])

            snp = np.array(snp)
            etf = np.array(etf)

            snp = np.concatenate(snp, axis=0)
            etf = np.concatenate(etf, axis=0)

            means_trace = []
            covs_trace = []

            step = 0

            x = etf[step]
            y = snp[step]

            observation_matrix_stepwise = np.array([[x, 1]])
            observation_stepwise = y

            kf = KalmanFilter(n_dim_obs=1,
                              n_dim_state=2,
                              initial_state_mean=np.ones(2),
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              transition_covariance=0.001*np.eye(2),
                              observation_covariance=2,
                              observation_matrices=observation_matrix_stepwise)

            spread = y - observation_matrix_stepwise.dot(np.ones(2))
            spreads = [spread]

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

                spread = y - observation_matrix_stepwise.dot(means_trace[-1])[0]

                means_trace.append(state_means_stepwise.data)
                covs_trace.append(state_covs_stepwise)
                spreads.append(spread)

            meanSpread = stat.mean(spreads[2:])
            stdSpread = stat.stdev(spreads[2:])

            z_score = (spread-meanSpread)/stdSpread

            if z_score > entry_z or z_score < -entry_z:
                results_queue.put([pair[0], z_score, means_trace[-1]])
        except:
            print("Error")
            continue


def kalman_single(window: Window, pair, iteration):

    start_date = window.get_nth_working_day_ahead(target=window.window_start, n=-iteration*10)

    snp = window.repository.all_data.get(Universes.SNP)
    etf = window.repository.all_data.get(Universes.ETFs)

    snp = snp.loc[start_date:window.window_end, pd.IndexSlice[pair[0], Features.CLOSE]]
    etf = etf.loc[start_date:window.window_end:, pd.IndexSlice[pair[1], Features.CLOSE]]

    snp = np.array(snp)
    etf = np.array(etf)

    means_trace = []
    covs_trace = []

    step = 0

    x = etf[step]
    y = snp[step]

    observation_matrix_stepwise = np.array([[x, 1]])
    observation_stepwise = y

    kf = KalmanFilter(n_dim_obs=1,
                      n_dim_state=2,
                      initial_state_mean=np.ones(2),
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      transition_covariance=np.eye(2)*0.001,
                      observation_covariance=2,
                      observation_matrices=observation_matrix_stepwise)

    spread = y - observation_matrix_stepwise.dot(np.ones(2))
    spreads = [spread]

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

        spread = y - observation_matrix_stepwise.dot(means_trace[-1])[0]

        means_trace.append(state_means_stepwise.data)
        covs_trace.append(state_covs_stepwise)
        spreads.append(spread)

    meanSpread = stat.mean(spreads[2:])
    stdSpread = stat.stdev(spreads[2:])

    z_score = (spread - meanSpread) / stdSpread

    return {"pair": pair, "z_score": z_score}
