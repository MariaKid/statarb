import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from typing import Dict, List
from src.DataRepository import Universes
from src.Features import Features


class Clusterer:
    def __init__(self):
        self.cluster_history = []

    def dbscan(self, min_samples: int, eps: float = None, window=None) -> Dict[int, List]:
        """
        Determines the SNP/ETF clusters according to volume and intraday volatility using the dbscan algorithm

        Parameters:
            min_samples: minimum number of points to define cluster
            eps: maximum distance for two points to be considered as close
            window: Window object containing data for specified time window

        Returns:
            clusters: dictionary with cluster numbers as keys and ticker lists as values
        """
        self.window = window

        # Filter dataframe according to volume and intraday volatility
        clustering_features = [Features.INTRADAY_VOL, Features.VOLUME]
        snp_to_cluster = self.window.get_data(Universes.SNP, None, clustering_features)
        etf_to_cluster = self.window.get_data(Universes.ETFs, None, clustering_features)

        # Concatenate SNP and ETF dataframes, use common dates in case of different lengths
        try:
            data_to_cluster = pd.concat([snp_to_cluster, etf_to_cluster], axis=1)
        except ValueError as _:
            print(f"Got different lengths for ETF and SNP for clustering data, taking intersection...")
            common_dates = sorted(set(snp_to_cluster.index).intersection(set(etf_to_cluster.index)))

            data_to_cluster = pd.concat([
                snp_to_cluster.loc[snp_to_cluster.index.intersection(common_dates)].drop_duplicates(keep='first'),
                etf_to_cluster.loc[etf_to_cluster.index.intersection(common_dates)].drop_duplicates(keep='first')
            ], axis=1)

        # Calculate mean of individual SNP/ETF volatility and volume over time window, rank results in ascending order
        features_mean = data_to_cluster.mean(axis=0)
        ranked_mean_intraday_vols = features_mean.loc[:, Features.INTRADAY_VOL].rank(ascending=True)
        ranked_mean_volumes = features_mean.loc[:, Features.VOLUME].rank(ascending=True)

        # Normalise results
        def rank_normaliser(val, series):
            return (val - 0.5*(max(series) - min(series)))/(0.5*(max(series) - min(series)))

        normed_volume_ranks = ranked_mean_volumes.apply(lambda val: rank_normaliser(val, ranked_mean_volumes))
        normed_intraday_vol_ranks = ranked_mean_intraday_vols.apply(
            lambda val: rank_normaliser(val, ranked_mean_intraday_vols))

        # Reconcatenate results and perform dbscan algorithm to create the clusters
        df = pd.concat([normed_volume_ranks, normed_intraday_vol_ranks], axis=1)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df)

        labels = dbscan.labels_
        self.tickers = normed_volume_ranks.index
        self.unique_labels = set(labels)
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise = list(labels).count(-1)
        self.noise = np.where(labels == -1)[0]

        # Cluster dictionary
        clusters = {}
        for cluster_num in self.unique_labels:
            tickers_in_cluster = df.index[np.where(labels == cluster_num)[0]]
            clusters[cluster_num] = tickers_in_cluster.values

        return clusters
