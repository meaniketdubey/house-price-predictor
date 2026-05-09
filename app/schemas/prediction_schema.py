from pydantic import BaseModel


class HousePredictionRequest(BaseModel):

    size: float
    bedrooms: int

