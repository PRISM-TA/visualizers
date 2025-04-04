from app.models.TradeLog import TradeLog
from app.strategies.BaseStrategy import BaseStrategy, BaseStrategyParam
from app.datafeed.DataFeeder import DataFeeder

from datetime import date

class BAHParam(BaseStrategyParam):
    def __init__(self, initial_capital: float):
        super().__init__(initial_capital)
        pass

class BAHStrategy(BaseStrategy):
    strategy_name: str = "BuyAndHoldStrategy"
    trades: list[TradeLog] = []

    def __init__(self, datafeeder: DataFeeder, param: BAHParam):
        super().__init__(datafeeder, param)

    def run(self, ticker: str, model: str, feature_set: str):
        datafeed = self.datafeeder.pullData(ticker=ticker, classifier_model=model, feature_set=feature_set)
        shares = 0

        for idx, daily_data in enumerate([datafeed[0], datafeed[-1]]):
            if idx == 0:
                self.shares = self.param.initial_capital / daily_data.close
                self._handle_buy(ticker, daily_data.report_date, daily_data.close, self.shares)
                continue
            if idx == 1:
                self._handle_sell(ticker, daily_data.report_date, daily_data.close, self.shares, "End of trading period")

    def dump_trade_logs(self) -> list[TradeLog]:
        return self.trades

    