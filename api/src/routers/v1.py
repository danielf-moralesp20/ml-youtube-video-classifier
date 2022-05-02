from fastapi import APIRouter
from src.models.prediction_request import PredictionRequest


router = APIRouter(prefix="/api/v1")


@router.post('/estimate')
def make_estimation(request: PredictionRequest):
    return {"message": "Hello World"}
