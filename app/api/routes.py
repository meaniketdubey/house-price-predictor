from fastapi import APIRouter

from app.schemas.prediction_schema import (

    HousePredictionRequest

)

from app.services.prediction_service import (

    Predictservice
)


router = APIRouter()


prediction_service = Predictservice()



@router.get("/health")
def health_check():

    return {
        "status" : "ok"
    }

@router.post("/predict")
def predict_house_price(

    request:HousePredictionRequest):

    prediction = prediction_service.predict(
        request.size,
        request.bedrooms    
    )

    return{
        "predicted_price": prediction
    }