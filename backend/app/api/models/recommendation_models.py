from pydantic import BaseModel
from typing import List, Union,Optional


class ResponseModel(BaseModel):
    user_id: int
    type: str
    message: str
    recommendations: Union[List[int], None]

class TopRatedMoviesResponse(BaseModel):
    top_rated_movies: Union[List[int]]


class SimilarMovieResponse(BaseModel):
    user_id: int
    type: str
    message: str
    movie_id: int
    similar_recommendations: Union[List[int], None]
