from sqlalchemy import Column, Integer, String, JSON
from database import Base


class Predictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    plate_id = Column(String)
    date = Column(String)
    time = Column(String)

    upperRowVibrios = Column(Integer)
    upperRowStaphylos = Column(Integer)
    lowerRowVibrios = Column(Integer)
    lowerRowStaphylos = Column(Integer)

    upperRowPred = Column(JSON)


