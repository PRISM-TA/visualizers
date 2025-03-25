from sqlalchemy import Column, Date, Float, BigInteger, String, schema
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class EquityIndicators(Base):
    __tablename__ = 'equity_indicators'
    __table_args__ = {'schema': 'fyp'}

    # Composite primary key
    ticker = Column(String, primary_key=True)
    report_date = Column(Date, primary_key=True)
    
    # Technical indicators
    rsi_1 = Column(Float(4))
    rsi_2 = Column(Float(4))
    rsi_3 = Column(Float(4))
    rsi_4 = Column(Float(4))
    rsi_5 = Column(Float(4))
    rsi_6 = Column(Float(4))
    rsi_7 = Column(Float(4))
    rsi_8 = Column(Float(4))
    rsi_9 = Column(Float(4))
    rsi_10 = Column(Float(4))
    rsi_11 = Column(Float(4))
    rsi_12 = Column(Float(4))
    rsi_13 = Column(Float(4))
    rsi_14 = Column(Float(4))
    rsi_15 = Column(Float(4))
    rsi_16 = Column(Float(4))
    rsi_17 = Column(Float(4))
    rsi_18 = Column(Float(4))
    rsi_19 = Column(Float(4))
    rsi_20 = Column(Float(4))
    sma_10 = Column(Float(4))
    sma_20 = Column(Float(4))
    sma_50 = Column(Float(4))
    sma_200 = Column(Float(4))
    ema_10 = Column(Float(4))
    ema_20 = Column(Float(4))
    ema_50 = Column(Float(4))
    ema_200 = Column(Float(4))
    macd_12_26_9_line = Column(Float(4))
    macd_12_26_9_signal = Column(Float(4))
    macd_12_26_9_histogram = Column(Float(4))
    rv_10 = Column(Float(4))
    rv_20 = Column(Float(4))
    rv_30 = Column(Float(4))
    rv_60 = Column(Float(4))
    hls_10 = Column(Float(4))
    hls_20 = Column(Float(4))
    obv = Column(BigInteger)
    pct_5 = Column(Float(4)) # Add Percentage Change 
    pct_20 = Column(Float(4)) # Add Percentage Change
    pct_50 = Column(Float(4)) # Add Percentage Change
    pct_200 = Column(Float(4)) # Add Percentage Change

    def __repr__(self):
        return f"<EquityIndicators(date={self.report_date}, ticker={self.ticker})>"