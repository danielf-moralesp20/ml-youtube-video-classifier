from pydantic import BaseModel
from models.platform import Platform


class PredictionRequest(BaseModel):
    url: str
    platform: Platform
