from fastapi import FastAPI

from routers import documents_router, query_router

app = FastAPI()

app.include_router(documents_router)
app.include_router(query_router)
