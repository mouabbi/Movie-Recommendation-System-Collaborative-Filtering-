from sklearn.neighbors import NearestNeighbors
import pandas as pd

def item_item_collaborative_filtering(user_movie_matrix, index_to_movie_id, k=5):
    """
    Recommend movies based on item-item similarity (k-nearest neighbors) using preprocessed data.
    
    :param user_movie_matrix: The preprocessed user-item matrix (DataFrame)
    :param index_to_movie_id: Mapping from matrix column index to movieId
    :param k: The number of nearest neighbors to consider
    
    :return: A dictionary of movieId -> recommended movie IDs
    """
    # Validate input data
    if not isinstance(user_movie_matrix, pd.DataFrame):
        raise ValueError("user_movie_matrix must be a pandas DataFrame.")
    
    # Compute the item-item similarity using k-NN
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_movie_matrix.T)  # Transpose to treat movies as items

    recommendations = {}
    
    # Iterate through each movie in the matrix
    for movie_id in user_movie_matrix.columns:
        # Find the k nearest neighbors (most similar movies)
        distances, indices = knn.kneighbors(user_movie_matrix.T.loc[movie_id].values.reshape(1, -1), n_neighbors=k)
        
        # Get the indices of the similar movies
        similar_movie_indices = user_movie_matrix.columns[indices[0]]
        
        # Convert movie indices back to movie IDs
        similar_movie_ids = [index_to_movie_id[idx] for idx in similar_movie_indices]
        
        # Store recommendations for the current movie
        recommendations[index_to_movie_id[movie_id]] = similar_movie_ids
    
    return recommendations
