from fastapi import FastAPI
from app.routes import product
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Include product routes
app.include_router(product.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://xv3.hmr.ph/",
        "https://xv3.staging.hmr.ph/"
    ],  # Only allow these origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
