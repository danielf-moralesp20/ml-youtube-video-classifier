import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import v1


app = FastAPI()
app.include_router(v1.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "null"
    ],
    allow_methods=["*"],
)


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
