import numpy as np

def new_user_recommendation(
    ratings, 
    item_item_recommendations, 
    watched_movie_id=None, 
    top_n=10, 
    diversity_factor=0.3
):
    """
    Recommends movies for a new user with a focus on diversity.
    - If no movie is watched, recommend a mix of popular and random movies.
    - If a movie is watched, recommend a mix of similar and diverse movies.
    
    :param ratings: DataFrame with columns ['userId', 'movieId', 'rating']
    :param item_item_recommendations: Precomputed dictionary of movieId -> recommended movie IDs
    :param watched_movie_id: ID of the movie watched by the new user
    :param top_n: Number of movies to recommend
    :param diversity_factor: Proportion of recommendations to include as random or diverse
    
    :return: List of recommended movie IDs
    """
    
    # If no movie has been watched, recommend popular and random movies
    if watched_movie_id is None:
        # Get the top n popular movies based on the mean rating
        popular_movies = (
            ratings.groupby('movieId')['rating']
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        
        # Add diversity by mixing in some random movies
        all_movies = ratings['movieId'].unique()
        num_diverse = int(top_n * diversity_factor)
        random_movies = list(np.random.choice(all_movies, size=num_diverse, replace=False))
        
        return popular_movies[:top_n - num_diverse] + random_movies
    
    else:
        # Get similar movies based on item-item recommendations
        similar_movies = item_item_recommendations.get(watched_movie_id, [])
        
        # If no similar movies, return an empty list or a fallback strategy (e.g., random recommendations)
        if not similar_movies:
            print("No similar movies found for the watched movie!")
            similar_movies = []
        
        # Add diversity by mixing in random movies
        all_movies = ratings['movieId'].unique()
        num_diverse = int(top_n * diversity_factor)
        random_movies = list(np.random.choice(all_movies, size=num_diverse, replace=False))
        
        return similar_movies[:top_n - num_diverse] + random_movies
    
    
