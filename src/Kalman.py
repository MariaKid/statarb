import numpy as np
import pandas as pd
import queue as q
import multiprocessing as mp

from datetime import date
from pykalman import KalmanFilter

from src.Window import Window
from src.DataRepository import Universes
from src.Features import Features


def load_q(current_cointegrated_pairs):
    manager = mp.Manager()
    queue = manager.Queue()

    for pair in current_cointegrated_pairs:
        queue.put(pair)

    return queue


def parallel_kalman(current_cointegrated_pairs, start_date: date, end_date: date, window: Window):

    num_processes = mp.cpu_count()
    processes = []
    pair_queue = load_q(current_cointegrated_pairs)
    results_queue = mp.Manager().Queue()

    for i in range(num_processes):
        new_process = mp.Process(target=kalman, args=(start_date, end_date, window, pair_queue, results_queue))
        processes.append(new_process)
        new_process.start()

    for p in processes:
        p.join()

    selected_cointegrated_pairs = []
    for k in range(results_queue.qsize()):
        selected_cointegrated_pairs.append(results_queue.get())

    return selected_cointegrated_pairs


def kalman(start_date: date, end_date: date, window: Window, test_queue, results_queue):
    while not test_queue.empty():
        try:
            pair = test_queue.get(timeout=1)
        except q.Empty:
            break

        try:
            snp = window.repository.all_data.get(Universes.SNP)
            etf = window.repository.all_data.get(Universes.ETFs)

            snp = snp.loc[start_date:end_date, pd.IndexSlice[pair[0][0], Features.CLOSE]]
            etf = etf.loc[start_date:end_date:, pd.IndexSlice[pair[0][1], Features.CLOSE]]

            snp = np.array(snp)
            etf = np.array(etf)

            means_trace = []
            covs_trace = []
            # spreads = []
            # spreads_std = []
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
            spread_std = np.sqrt(observation_matrix_stepwise.dot(P).dot(observation_matrix_stepwise.transpose())[0][0] +
                                 observation_cov)
            spread = y - observation_matrix_stepwise.dot(np.ones(2))

            # spreads.append(spread)
            # spreads_std.append(spread_std)

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
                spread_std = np.sqrt(observation_matrix_stepwise.dot(P).dot(observation_matrix_stepwise.transpose())[0][0] +
                                     observation_cov)
                means_trace.append(state_means_stepwise.data)
                covs_trace.append(state_covs_stepwise)
                # spreads.append(spread)
                # spreads_std.append(spread_std)

            if spread > spread_std or spread < -spread_std:
                results_queue.put([pair[0], spread, spread_std, means_trace[-1]])
        except:
            continue


def kalman_single(start_date: date, end_date: date, window: Window, pair):
    snp = window.repository.all_data.get(Universes.SNP)
    etf = window.repository.all_data.get(Universes.ETFs)

    snp = snp.loc[start_date:end_date, pd.IndexSlice[pair[0], Features.CLOSE]]
    etf = etf.loc[start_date:end_date:, pd.IndexSlice[pair[1], Features.CLOSE]]

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
                      transition_covariance=np.eye(2) * state_cov_multiplier,  # wt
                      observation_covariance=observation_cov,  # vt
                      observation_matrices=observation_matrix_stepwise)  # Ft

    P = np.ones((2, 2)) + np.eye(2) * state_cov_multiplier
    spread_std = np.sqrt(observation_matrix_stepwise.dot(P).dot(observation_matrix_stepwise.transpose())[0][0] +
                         observation_cov)
    spread = y - observation_matrix_stepwise.dot(np.ones(2))

    # spreads.append(spread)
    # spreads_std.append(spread_std)

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

        P = covs_trace[-1] + np.eye(2) * state_cov_multiplier
        spread = y - observation_matrix_stepwise.dot(means_trace[-1])[0]
        spread_std = np.sqrt(observation_matrix_stepwise.dot(P).dot(observation_matrix_stepwise.transpose())[0][0] +
                             observation_cov)
        means_trace.append(state_means_stepwise.data)
        covs_trace.append(state_covs_stepwise)

        # spreads.append(spread)
        # spreads_std.append(spread_std)

    return {"pair": pair, "spread": spread, "spread_std": spread_std}
