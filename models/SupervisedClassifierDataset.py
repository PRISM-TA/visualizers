from sqlalchemy import Column, Date, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SupClassifierDataset(Base):
    __tablename__ = 'supervised_classifier_dataset'
    __table_args__ = {'schema': 'fyp'}

    start_date = Column(Date, primary_key=True)
    end_date = Column(Date, primary_key=True)
    ticker = Column(String, primary_key=True)
    label = Column(Float(4))

    def __repr__(self):
        return f"<SupClassifierDataset(start_date={self.start_date}, end_date={self.end_date}, ticker={self.ticker})>"