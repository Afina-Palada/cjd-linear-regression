from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .predict import predict_price

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/click")
def get_price(days_to_departure: int, service_class: str, count_tickets: int, service_price: float, trip: float):
    return predict_price({
        'Дней до отправления': [days_to_departure],
        'Тип/Кл.обсл.': [service_class],
        'Кол-во прод. мест': [count_tickets],
        'Сумма серв. усл.': [service_price],
        'Расстояние': [trip]
    }) * count_tickets
