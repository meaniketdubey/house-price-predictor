from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="House Price Prediction API"
)

app.include_router(router)