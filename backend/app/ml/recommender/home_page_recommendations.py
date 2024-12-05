import numpy as np
import random

def home_page_recommendations(
    user_id, 
    ratings, 
    user_movie_matrix, 
    item_item_recommendations, 
    user_user_recommendations, 
    svd_prediction_matrix,
    movie_id_to_index,
    user_id_to_index,
    index_to_movie_id,
    weights=(0.4, 0.3, 0.3), 
    top_n=50, 
    global_top_movies=5  # Number of global top rated movies to include
):
    """
    Generates home page recommendations for a returning user based on their history, 
    with randomization, diversity, and sorting by predicted and average ratings.
    
    :param user_id: ID of the user for whom to generate recommendations
    :param ratings: DataFrame with columns ['userId', 'movieId', 'rating']
    :param user_movie_matrix: User-item matrix
    :param item_item_recommendations: Precomputed item-item recommendations
    :param user_user_recommendations: Precomputed user-user recommendations
    :param svd_prediction_matrix: SVD-based prediction matrix
    :param weights: Weights for blending the recommendation methods (item-item, user-user, SVD)
    :param top_n: Number of movies to recommend
    :param diversity_factor: Factor to control diversity in the recommendations (between 0 and 1)
    :param global_top_movies: Number of global top rated movies to include
    :return: List of recommended movie IDs
    """
    
    # Get the user's ratings history
    user_ratings = ratings[ratings['userId'] == user_id]
    rated_movies = user_ratings['movieId'].tolist()
    
    # ----- 1. Item-Item Collaborative Filtering -----
    top_rated_movies = user_ratings.sort_values(by='rating', ascending=False)['movieId'].head(5).tolist()
    item_item_movies = []
    for movie_id in top_rated_movies:
        if movie_id in item_item_recommendations:
            item_item_movies.extend(item_item_recommendations[movie_id])
    
    item_item_movies = [m for m in item_item_movies if m not in rated_movies]
    
    # ----- 2. User-User Collaborative Filtering -----
    user_user_movies = []
    if user_id in user_user_recommendations:
        similar_user_top_movies = user_user_recommendations[user_id]
        user_user_movies.extend([m for m in similar_user_top_movies if m not in rated_movies])
    
    # ----- 3. Predicted Ratings (SVD) -----
    user_index = user_id_to_index[user_id]
    if user_index in user_movie_matrix.index:
        svd_predictions = svd_prediction_matrix[user_index]
        svd_movie_ids = [index_to_movie_id[idx] for idx in user_movie_matrix.columns]
        svd_recommendations = sorted(
            zip(svd_movie_ids, svd_predictions), 
            key=lambda x: x[1], 
            reverse=True
        )
        svd_movies = [movie_id for movie_id, pred in svd_recommendations if movie_id not in rated_movies]
    else:
        svd_movies = []
    
    # ----- Global Top Rated Movies -----
    popular_movies = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(global_top_movies).index.tolist()
    popular_movies = [m for m in popular_movies if m not in rated_movies]
    
    # ----- Blending Recommendations -----
    item_item_count = int(weights[0] * top_n)
    user_user_count = int(weights[1] * top_n)
    svd_count = top_n - item_item_count - user_user_count
    
    # Combine recommendations with weights
    combined_recommendations = (
        item_item_movies[:item_item_count] + 
        user_user_movies[:user_user_count] + 
        svd_movies[:svd_count]
    )
    
    # Add Global Top Rated Movies
    combined_recommendations = list(set(combined_recommendations) | set(popular_movies))
    
    # Introduce diversity by randomly shuffling the combined recommendations
    random.shuffle(combined_recommendations)
    
    # Get the top N recommended movies, considering the total size
    final_recommendations = combined_recommendations[:top_n]
    
    # ----- Get the predicted ratings for each movie in the final recommendations
    
    predicted_ratings = []
    for movie_id in final_recommendations:
        # Check if we have a predicted rating for the movie
        if movie_id in svd_movies:
            movie_index = movie_id_to_index[movie_id]
            predicted_rating = svd_prediction_matrix[user_index][movie_index]
        else:
            predicted_rating = 0  # Default to 0 if no prediction is available
        
        predicted_ratings.append((movie_id, predicted_rating))
    
    # Sort the recommendations by the predicted ratings
    final_recommendations = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)
    
    # Return only the movie IDs sorted by their predicted ratings
    return [movie_id for movie_id, _ in final_recommendations[:top_n]]
