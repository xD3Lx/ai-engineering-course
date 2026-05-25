from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Hello World API")


class HelloResponse(BaseModel):
    message: str


class GreetRequest(BaseModel):
    name: str


@app.get("/", response_model=HelloResponse)
def read_root() -> HelloResponse:
    return HelloResponse(message="Hello, World!")
