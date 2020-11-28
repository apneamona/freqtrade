"""
Performance pair list filter
"""
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from pandas import DataFrame, Series

from freqtrade.pairlist.IPairList import IPairList
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone

from freqtrade.edge import Edge

logger = logging.getLogger(__name__)
SORT_VALUES = ['profit', 'win_rate', 'expectancy']


class PerformanceFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._db_url = config['db_url']

        self._max_trade_duration = pairlistconfig.get('max_trade_duration', 0)
        self._max_trades = pairlistconfig.get('max_trades', 0)
        self._min_trades = pairlistconfig.get('min_trades', 0)
        self._min_profit = pairlistconfig.get('min_profit', 0)
        self._min_winrate = pairlistconfig.get('min_winrate', 0.5)
        self._min_expectancy = pairlistconfig.get('min_expectancy', 0.3)

        if self._max_trades != 0 and self._max_trades < self._min_trades:
            raise OperationalException(f"Set max_trades to >= {self._min_trades} "
                                       "or 0 to use all trades")

        self._sort_key = pairlistconfig.get('sort_key')

        if not self._validate_keys(self._sort_key):
            raise OperationalException(
                f'key {self._sort_key} not in {SORT_VALUES}')

        self._enabled = self._sort_key != None

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requries tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """
        return False

    def _validate_keys(self, key):
        return key in SORT_VALUES

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (f"{self.name} - Sorting pairs by {self._sort_key}" +
                (f" and filtering below {self._min_profit * 100}%." if self._min_profit != 0 else "") + 
                (f" and filtering below {self._min_expectancy}." if self._min_expectancy != 0 else "") +
                (f" and filtering below {self._min_winrate * 100}%." if self._min_winrate != 0 else ".")
                )

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:

        columns = ["pair", "profit", "open_rate", "close_rate", "duration", "sell_reason"]

        trades = pd.DataFrame([(t.pair,
                                t.calc_profit(),
                                t.open_rate, t.close_rate,
                                (round((t.close_date.timestamp() - t.open_date.timestamp()) / 60, 2)
                                if t.close_date else None),
                                t.sell_reason

                                )
                            for t in Trade.get_trades([Trade.open_date < datetime.utcnow(),
                                Trade.is_open == False,
                                Trade.close_profit != None,
                    ]).all()], columns=columns)

        # Clean all active trades
        trades = trades.dropna()

        # Removing trades with a duration more than X minutes
        if self._max_trade_duration != 0:
            trades = trades[trades['duration'] < self._max_trade_duration]
            self.log_on_refresh(logger.info, f"Excluding trades with "
                                                f"duration above maximum value {self._max_trade_duration}.")

        # Limiting maximum number of trades
        if self._max_trades != 0:
            trades = trades.groupby('pair').tail(self._max_trades)
            self.log_on_refresh(logger.info, f"Limiting maximum "
                                                f"number of trades {self._max_trades}.")

        # Construct new dataframe and aggregate values per pair
        df = trades.groupby('pair').agg(
            profit = ('profit', 'sum'),
            n_trades = ('profit', 'count'), # number of all trades
            profit_sum = ('profit', lambda x: x[x > 0].sum()), # cumulative profit of all winning trades
            loss_sum = ('profit', lambda x: abs(x[x < 0].sum())), # cumulative loss of all losing trades
            n_win_trades = ('profit', lambda x: x[x > 0].count()) # number of winning trades
            ).reset_index()

        # This is same as in Edge module
        # Calculating number of losing trades, average win and average loss
        df['n_loss_trades'] = df['n_trades'] - df['n_win_trades']
        df['average_win'] = df['profit_sum'] / df['n_win_trades'] if df['n_win_trades'] > 0 else 0.0
        df['average_loss'] = df['loss_sum'] / df['n_loss_trades'] if df['n_loss_trades'] > 0 else 0.0

        # Win rate = number of profitable trades / number of trades
        df['winrate'] = df['n_win_trades'] / df['n_trades'] if df['n_trades'] > 0 else 0.0

        # risk_reward_ratio = average win / average loss
        df['risk_reward_ratio'] = df['average_win'] / df['average_loss'] if df['average_loss'] > 0 else df['average_win']

        # required_risk_reward = (1 / winrate) - 1
        # df['required_risk_reward'] = (1 / df['winrate']) - 1

        # expectancy = (risk_reward_ratio * winrate) - (lossrate)
        df['expectancy'] = (df['risk_reward_ratio'] * df['winrate']) - (1 - df['winrate'])

        # df['E'] = ((df['winrate'] * df['average_win']) - ((1-df['winrate']) * df['average_loss'])) / df['average_loss'] # Expectancy
        # df['E1'] = ((1 + df['risk_reward_ratio']) * df['winrate']) - 1
        # df['E2'] = (df['risk_reward_ratio'] * df['winrate']) - (1 - df['winrate'])

        # Excluding pairs having less than min_trades
        if self._min_trades != 0:
            df = df[df['n_trades'] >= self._min_trades]
            self.log_on_refresh(logger.info, f"Limiting pairs with "
                                                f"less then minimum number of trades {self._min_trades}.")

        # Pairlist dataframe
        list_df = pd.DataFrame({'pair':pairlist})

        # Joining pairlist and aggregated performance by pair
        # filling missing values and sorting by sort_key
        sorted_df = list_df.join(df.set_index('pair'), on='pair')\
            .fillna(0)\
            .sort_values(by=[self._sort_key], ascending=False)\
            .reset_index(drop=True)

        # Removing pairs with big loss
        if self._min_profit != 0:
            sorted_df = sorted_df[sorted_df['profit'] >= self._min_profit]
            self.log_on_refresh(logger.info, f"Removed pairs from whitelist with "
                                                f"profit below min_profit value {self._min_profit * 100}%.")

        # Removing pairs bellow minimum winrate
        if self._min_winrate != 0:
            sorted_df = sorted_df[sorted_df['winrate'] >= self._min_winrate]
            self.log_on_refresh(logger.info, f"Removed pairs from whitelist with "
                                                f"winrate below min_winrate value {self._min_winrate}.")

        # Removing pairs with negative expectany
        if self._min_expectancy != 0:
            sorted_df = sorted_df[sorted_df['expectancy'] >= self._min_expectancy]
            self.log_on_refresh(logger.info, f"Removed pairs from whitelist with "
                                                f"expectancy below min_expectancy value {self._min_expectancy}.")

        print(sorted_df)

        pairlist = sorted_df['pair'].tolist()

        return pairlist