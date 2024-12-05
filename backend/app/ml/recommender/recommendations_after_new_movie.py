import random
import numpy as np

def recommendations_after_new_movie(
    user_id, 
    new_movie_id, 
    ratings, 
    item_item_recommendations, 
    svd_prediction_matrix, 
    user_movie_matrix, 
    user_id_to_index, 
    index_to_movie_id, 
    weights=(0.7, 0.2), 
    top_n=10,
    rating_threshold=1
):
    """
    Generate recommendations after a user watches a new movie.
    
    :param user_id: ID of the user
    :param new_movie_id: ID of the newly watched movie
    :param ratings: DataFrame with columns ['userId', 'movieId', 'rating']
    :param item_item_recommendations: Precomputed item-item recommendations
    :param svd_prediction_matrix: SVD-based prediction matrix
    :param user_movie_matrix: User-item matrix
    :param user_id_to_index: Mapping from user ID to index in the SVD matrix
    :param index_to_movie_id: Mapping from matrix indices to movie IDs
    :param weights: Weights for blending the recommendation methods (item-item, SVD)
    :param top_n: Number of movies to recommend
    :param rating_threshold: Minimum predicted rating threshold for SVD-based recommendations
    :return: List of recommended movie IDs
    """
    # Get the user's ratings history
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    
    # ----- 1. Item-Item Collaborative Filtering -----
    similar_movies = []
    if new_movie_id in item_item_recommendations:
        similar_movies = item_item_recommendations[new_movie_id]
    
    # Remove movies the user has already rated
    similar_movies = [m for m in similar_movies if m not in user_rated_movies]
    
    # ----- 2. Predicted Ratings (SVD) -----
    svd_recommendations = []
    user_index = user_id_to_index[user_id]
    if user_index in user_movie_matrix.index:
        svd_predictions = svd_prediction_matrix[user_index]
        svd_movie_ids = [index_to_movie_id[idx] for idx in user_movie_matrix.columns]
        svd_recommendations = [
            movie_id for movie_id, pred in zip(svd_movie_ids, svd_predictions) 
            if pred > rating_threshold and movie_id not in user_rated_movies
        ]
    
    # ----- 3. Global Top Rated Movies -----
    global_top_movies = (
        ratings.groupby('movieId')['rating']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    
    # Remove movies the user has already rated
    global_top_movies = [m for m in global_top_movies if m not in user_rated_movies]
    
    # ----- Blending Recommendations -----
    item_item_count = int(weights[0] * top_n)
    svd_count = int(weights[1] * top_n)
    global_count = top_n - item_item_count - svd_count
    
    # Select the recommendations
    recommendations = (
        similar_movies[:item_item_count] +
        svd_recommendations[:svd_count] +
        global_top_movies[:global_count]
    )
    
    print(f"{len(similar_movies[:item_item_count])} / {item_item_count}")
    print(f"{len(svd_recommendations[:svd_count])} / {svd_count}")
    print(f"{len(global_top_movies[:global_count])} / {global_count}")
    
    # Deduplicate and introduce diversity
    recommendations = list(dict.fromkeys(recommendations))
    
    # Ensure the first 5 movies are strictly similar movies
    top_similar = similar_movies[:5]
    remaining_recommendations = [movie for movie in recommendations if movie not in top_similar]
    random.shuffle(remaining_recommendations)
    
    # Combine and limit to top_n
    final_recommendations = top_similar + remaining_recommendations[:top_n - len(top_similar)]
    
    # If not enough recommendations, fill with additional top-rated movies
    additional_movies = [m for m in global_top_movies if m not in final_recommendations]
    while len(final_recommendations) < top_n and additional_movies:
        final_recommendations.append(additional_movies.pop(0))
    
    return final_recommendations[:top_n]
