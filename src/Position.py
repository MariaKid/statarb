from pandas import DataFrame, to_datetime
from datetime import date

from src.Features import PositionType
from src.Tickers import Tickers
from src.Window import Window


class Position:
    def __init__(self, ticker1: Tickers,
                 ticker2: Tickers,
                 value1: float,
                 value2: float,
                 investment_type: PositionType,
                 init_z: float,
                 init_date: date,
                 beta: float,
                 intercept: float,
                 init_value: float = 0,
                 commission: float = 0):
        self.asset1: Tickers = ticker1
        self.asset2: Tickers = ticker2
        self.value1: float = value1
        self.value2: float = value2
        self.position_type: PositionType = investment_type
        self.init_z = init_z
        self.beta: float = beta
        self.intercept: float = intercept
        self.init_date = init_date
        self.init_value: float = init_value
        self.commission: float = commission

        self.pnl: float = -commission
        self.quantity1: float = 0
        self.quantity2: float = 0
        self.simple_return: float = 0
        self.current_value: float = init_value
        self.pos_hist = list()
        self.closed: bool = False

    def set_position_value(self, value: float) -> None:
        """
        Sets cash dedicated to pair

        Parameters:
            value: initial total value of pair
        """
        self.init_value = value
        self.current_value = value

    def update_position_pnl(self, value: float, window: Window) -> None:
        """
        Checks if position is open, updates profit and loss, simple return and new current value at the end of window,
        record this position

        Parameters:
            value: current value of pair
            window: time window
        """
        if not self.closed:
            self.pnl += value - self.current_value
            self.simple_return = value / self.current_value - 1
            self.current_value = value
        self.pos_hist.append([window.window_end, self.current_value, self.pnl, self.simple_return])

    def close_trade(self, value: float, window: Window) -> None:
        """
        Closes position with a certain value, updates pair pnl and records closed position

        Parameters:
            value: current value of pair
            window: time window
        """
        self.pnl += value - self.current_value - self.commission
        self.current_value = value
        self.closed = True
        self.pos_hist.append([window.window_end, self.current_value, self.pnl])

    def get_pos_hist(self) -> DataFrame:
        """
        Records pair position history

        Returns:
             df: time series of position value and pnl
        """
        df = DataFrame(self.pos_hist, columns=['date', 'pos_value', 'pnl', 'return'])
        df['date'] = to_datetime(df['date'])
        df = df.set_index('date')
        return df.round(2)
