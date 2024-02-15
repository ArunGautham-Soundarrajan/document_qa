from contextlib import asynccontextmanager

from database.database import engine
from database.models import Base
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from routers.v1 import files, search
from vector_search.text_embedders import TextEmbedding

Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield {"text_embedder": TextEmbedding()}


app = FastAPI(lifespan=lifespan)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router)
app.include_router(search.router)


@app.get("/")
async def health(request: Request):
    return Response("Server is running.")
