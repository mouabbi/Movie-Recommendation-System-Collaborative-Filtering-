import pandas as pd

def generate_user_item_matrix(ratings, movies):
    """
    Generate the user-item matrix from ratings and movies data.

    :param ratings: DataFrame containing user ratings with columns ['userId', 'movieId', 'rating']
    :param movies: DataFrame containing movie metadata with column ['movieId']
    :return: A DataFrame representing the user-item matrix
    """
    # Movie ID to index mapping
    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(movies['movieId'])}
    ratings['movieIndex'] = ratings['movieId'].map(movie_id_to_index)
    
    # User ID to index mapping
    user_id_to_index = {user_id: index for index, user_id in enumerate(ratings['userId'].unique())}
    ratings['userIndex'] = ratings['userId'].map(user_id_to_index)

    # Check for unmapped IDs
    if ratings['movieIndex'].isnull().any():
        raise ValueError("Some movie IDs in the ratings data do not exist in the movies data.")
    if ratings['userIndex'].isnull().any():
        raise ValueError("Some user IDs could not be mapped to indices.")

    # Create the user-item matrix
    user_movie_matrix = ratings.pivot(index='userIndex', columns='movieIndex', values='rating').fillna(0)

    return user_movie_matrix


def generate_id_mappings(ratings, movies):
    """
    Generate ID mappings for users and movies.

    :param ratings: DataFrame containing user ratings with columns ['userId', 'movieId']
    :param movies: DataFrame containing movie metadata with column ['movieId']
    :return: A tuple containing:
             - movie_id_to_index: Mapping from movieId to matrix column index
             - index_to_movie_id: Mapping from matrix column index to movieId
             - user_id_to_index: Mapping from userId to matrix row index
             - index_to_user_id: Mapping from matrix row index to userId
    """
    # Movie ID to index mapping
    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(movies['movieId'])}
    index_to_movie_id = {index: movie_id for movie_id, index in movie_id_to_index.items()}

    # User ID to index mapping
    user_id_to_index = {user_id: index for index, user_id in enumerate(ratings['userId'].unique())}
    index_to_user_id = {index: user_id for user_id, index in user_id_to_index.items()}

    return movie_id_to_index, index_to_movie_id, user_id_to_index, index_to_user_id
