from sqlalchemy import Column, Date, Float, Integer, String, schema
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    __table_args__ = {'schema': 'fyp'}

    report_date = Column(Date, primary_key=True)
    ticker = Column(String, primary_key=True)
    open = Column(Float(8))
    close = Column(Float(8))
    low = Column(Float(8))
    high = Column(Float(8))
    volume = Column(Integer)
    type = Column(String)

    def __repr__(self):
        return f"<MarketData(date={self.report_date}, ticker={self.ticker})>"