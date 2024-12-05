from fastapi import FastAPI
from app.api.routes.recommendations import router 


app = FastAPI(
    title="Movie Recommendation API",
    description="API for recommending movies using collaborative filtering and SVD models.",
    version="1.0.0",
)

# Include the routes
app.include_router(router)

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the Movie Recommendation API!"}
