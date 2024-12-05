from fastapi import APIRouter, HTTPException, Query
from app.api.models.recommendation_models import ResponseModel, TopRatedMoviesResponse , SimilarMovieResponse
from app.api.controllers.MLController import MLController
from app.api.utils.logging import setup_logger

logger = setup_logger()
router = APIRouter()
ml_controller=MLController()

@router.get("/home_recommendations/{user_id}", response_model=ResponseModel, tags=["Home Recommendations"])
async def home_recommendations(
    user_id: int, 
    top_n: int = Query(50, description="Number of recommendations to fetch for new users"),
    diversity_factor: float = Query(0.5, description="Diversity factor for new user recommendations (0-1)")
):
    """
    Get personalized recommendations for a user.

    - For new users: Provides onboarding recommendations.
    - For old users: Provides personalized recommendations based on historical data.
    """
    try:
        logger.info(f"Received recommendation request for user_id: {user_id}")

        if ml_controller.is_new_user(user_id=user_id):
            logger.info(f"User {user_id} is new. Generating onboarding recommendations.")
            recommendations = ml_controller.recommend_for_new_user(top_n=top_n, diversity_factor=diversity_factor)
            return {
                "user_id": user_id,
                "type": "new_user",
                "recommendations": recommendations,
                "message": "This is a new user. Provide onboarding recommendations."
            }
        else:
            logger.info(f"User {user_id} is old. Generating recommendations based on history.")
            recommendations = ml_controller.recommend_for_old_user(user_id=user_id)
            return {
                "user_id": user_id,
                "type": "old_user",
                "recommendations": recommendations,
                "message": "This is an old user. Recommendations calculated based on history."
            }
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top_rated", response_model=TopRatedMoviesResponse, tags=["Top Rated"])
async def get_top_rated_movies():
    """ Get the top-rated movies globally. """
    try:
        logger.info("Fetching top-rated movies.")
        top_movies = ml_controller.top_rated_movies()
        return {"top_rated_movies": top_movies}
    except Exception as e:
        logger.error(f"Error fetching top-rated movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.get("/similar_movie_rec/{user_id}/{movie_id}", response_model=SimilarMovieResponse, tags=["Similar Movies"])
async def get_similar_movie_rec(
    user_id: int,
    movie_id: int,
    top_n: int = Query(50, description="Number of similar movies to recommend"),
    diversity_factor: float = Query(0.5, description="Diversity factor for new user recommendations (0-1)")

):
    """
    Get movie recommendations similar to a given movie for a user.

    - For new users: Provides generalized recommendations similar to the movie.
    - For old users: Provides personalized recommendations similar to the movie.
    """
    try:
        logger.info(f"Received similar movie recommendation request for user_id: {user_id}, movie_id: {movie_id}")

        if ml_controller.is_new_user(user_id=user_id):
            logger.info(f"User {user_id} is new. Generating similar recommendations for movie {movie_id}.")
            similar_recommendations = ml_controller.recommend_for_new_user(watched_movie_id=movie_id,top_n=top_n,diversity_factor=diversity_factor)
            return {
                "user_id": user_id,
                "type": "new_user",
                "movie_id": movie_id,
                "similar_recommendations": similar_recommendations,
                "message": f"This is a new user. Providing generalized recommendations similar to movie {movie_id}."
            }
        else:
            logger.info(f"User {user_id} is old. Generating personalized recommendations for movie {movie_id}.")
            similar_recommendations = ml_controller.recommend_similar_movie_for_old_user(user_id=user_id,new_movie_id=movie_id,top_n=top_n)
            return {
                "user_id": user_id,
                "type": "old_user",
                "movie_id": movie_id,
                "similar_recommendations": similar_recommendations,
                "message": f"This is an old user. Personalized recommendations similar to movie {movie_id}."
            }
    except Exception as e:
        logger.error(f"Error generating similar movie recommendations for user_id: {user_id}, movie_id: {movie_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))