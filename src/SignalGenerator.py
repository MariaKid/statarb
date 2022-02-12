from typing import List
import pandas as pd
from src.Portfolio import Portfolio, Position
from src.Features import PositionType, Features
from datetime import timedelta
from src.DataRepository import Universes
from src.Filters import Filters
from src.Kalman import kalman_single


class Decision:
    def __init__(self, position: Position, old_action: PositionType, new_action: PositionType):
        self.position: Position = position
        self.new_action: PositionType = new_action
        self.old_action: PositionType = old_action


class SignalGenerator:
    def __init__(self, port: Portfolio, time_stop_loss: int, entry_z: float, exit_z: float, emergency_delta_z: float):
        self.port: Portfolio = port
        self.emergency_delta_z: float = emergency_delta_z
        self.entry_z: float = entry_z
        self.exit_z: float = exit_z
        self.time_stop_loss: int = time_stop_loss

        self.filter = Filters()
        self.natural_close_count: int = 0
        self.emergency_close_count: int = 0
        self.time_stop_loss_count: int = 0
        self.volume_shock_filter: int = 0

    def make_decision(self, kalman_parallel_pairs: List) -> List[Decision]:
        """
        Creates Decision object for each position to be used to execute trades. Decisions are either open, close or hold

        Parameters:
            kalman_parallel_pairs: list of all cointegrated pairs in latest rolling window with spread>spread_std
        """
        today = self.port.current_window.window_end

        decisions = []
        positions = self.port.cur_positions

        for position in positions:
            position_pair = [position.asset1, position.asset2]
            position.it += 1
            kalman_result = kalman_single(window=self.port.current_window, pair=position_pair, iteration=position.it)
            z_score = kalman_result["z_score"]

            if today > (position.init_date + timedelta(days=self.time_stop_loss)) and (abs(z_score) < self.exit_z):
                decisions.append(Decision(position=position, old_action=position.position_type,
                                          new_action=PositionType.NOT_INVESTED))
                self.time_stop_loss_count += 1
            else:
                if position.position_type is PositionType.SHORT:
                    natural_close_required = z_score < self.exit_z
                    emergency_close_required = z_score > (self.entry_z + self.emergency_delta_z)

                    if natural_close_required or emergency_close_required:
                        decisions.append(Decision(position=position, old_action=PositionType.SHORT,
                                                  new_action=PositionType.NOT_INVESTED))
                        if natural_close_required:
                            self.natural_close_count += 1
                            print("Close due to exit threshold", position_pair, "z_score: ", round(z_score, 5))
                        else:
                            self.emergency_close_count += 1
                            print("Close due to emergency threshold", position_pair, "final spread: ",
                                  round(z_score, 5))
                    else:
                        decisions.append(Decision(position=position, old_action=PositionType.SHORT,
                                                  new_action=PositionType.SHORT))

                elif position.position_type is PositionType.LONG:
                    natural_close_required = z_score > -self.exit_z
                    emergency_close_required = z_score < -(self.exit_z + self.emergency_delta_z)

                    if natural_close_required or emergency_close_required:
                        decisions.append(Decision(position=position, old_action=PositionType.LONG,
                                                  new_action=PositionType.NOT_INVESTED))
                        if natural_close_required:
                            self.natural_close_count += 1
                            print("Close due to exit threshold ", position_pair, "z_score: ", round(z_score, 5))
                        else:
                            self.emergency_close_count += 1
                            print("Close due to exit threshold", position_pair, "z_score: ", round(z_score, 5))
                    else:
                        decisions.append(Decision(position=position, old_action=PositionType.LONG,
                                                  new_action=PositionType.LONG))

        current_posn_pairs = [[position.asset1, position.asset2] for position in positions]

        for pair in kalman_parallel_pairs:
            pair_assets = pair[0]
            z_score = pair[1]
            beta = pair[2][0]

            if beta < 0:
                continue

            if pair_assets not in current_posn_pairs:
                if z_score > self.entry_z:
                    p1 = pair_assets[0]
                    p2 = pair_assets[1]
                    shock = self.filter.run_volume_shock_filter_single_pair(pair_assets, self.port.current_window)
                    if not shock:
                        df1 = self.port.current_window.get_data(universe=Universes.SNP, tickers=[p1],
                                                                features=[Features.CLOSE])
                        df2 = self.port.current_window.get_data(universe=Universes.ETFs, tickers=[p2],
                                                                features=[Features.CLOSE])

                        cur_price = pd.concat([df1, df2], axis=1)

                        price_snp = cur_price.iloc[-1, 0]
                        price_etf = cur_price.iloc[-1, 1]

                        total = price_snp + beta*price_etf
                        ratio = int(5000/total)

                        quantity_snp = ratio
                        quantity_etf = ratio*beta

                        decisions.append(
                            Decision(
                                position=Position(
                                    ticker1=p1,
                                    ticker2=p2,
                                    value1=-quantity_snp*price_snp,
                                    value2=quantity_etf*price_etf,
                                    investment_type=PositionType.SHORT,
                                    z_score=z_score,
                                    beta=beta,
                                    init_date=today),
                                old_action=PositionType.NOT_INVESTED,
                                new_action=PositionType.SHORT,
                            )
                        )
                    else:
                        self.volume_shock_filter += 1

                elif z_score < -self.entry_z:
                    p1 = pair_assets[0]
                    p2 = pair_assets[1]
                    shock = self.filter.run_volume_shock_filter_single_pair(pair_assets, self.port.current_window)
                    if not shock:
                        df1 = self.port.current_window.get_data(universe=Universes.SNP, tickers=[p1],
                                                                features=[Features.CLOSE])
                        df2 = self.port.current_window.get_data(universe=Universes.ETFs, tickers=[p2],
                                                                features=[Features.CLOSE])

                        cur_price = pd.concat([df1, df2], axis=1)

                        price_snp = cur_price.iloc[-1, 0]
                        price_etf = cur_price.iloc[-1, 1]

                        total = price_snp + beta * price_etf
                        ratio = int(5000/total)

                        quantity_snp = ratio
                        quantity_etf = ratio*beta

                        decisions.append(
                            Decision(
                                position=Position(
                                    ticker1=p1,
                                    ticker2=p2,
                                    value1=quantity_snp*price_snp,
                                    value2=-quantity_etf*price_etf,
                                    investment_type=PositionType.LONG,
                                    z_score=z_score,
                                    beta=beta,
                                    init_date=today
                                ),
                                old_action=PositionType.NOT_INVESTED,
                                new_action=PositionType.LONG,
                            )
                        )
                    else:
                        self.volume_shock_filter += 1

        print("natural close count: ", self.natural_close_count)
        print("emergency (z-score) close count: ", self.emergency_close_count)
        print("time stop-loss close count: ", self.time_stop_loss_count)
        return decisions
