from sqlalchemy import Column, Integer, String, Date, Float
from dataclasses import dataclass
from datetime import date

@dataclass
class TradeBotDataFeed:
    report_date: date
    ticker: str
    model: str
    feature_set: str
    uptrend_prob: float
    side_prob: float
    downtrend_prob: float
    predicted_label: int
    open: float
    close: float
    rsi_1: float
    rsi_2: float
    rsi_3: float
    rsi_4: float
    rsi_5: float
    rsi_6: float
    rsi_7: float
    rsi_8: float
    rsi_9: float
    rsi_10: float
    rsi_11: float
    rsi_12: float
    rsi_13: float
    rsi_14: float
    rsi_15: float
    rsi_16: float
    rsi_17: float
    rsi_18: float
    rsi_19: float
    rsi_20: float

    def __repr__(self):
        return f"<TradeBotDataFeed(date={self.report_date}, ticker={self.ticker}, " \
               f"model={self.model}, feature_set={self.feature_set}, open={self.open}, close={self.close})>"