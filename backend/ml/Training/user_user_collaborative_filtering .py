import numpy as np
from sklearn.neighbors import NearestNeighbors

def user_user_collaborative_filtering(user_movie_matrix, index_to_movie_id, index_to_user_id, k=5):
    """
    Recommend movies based on user-user similarity (k-nearest neighbors) using preprocessed data.

    :param user_movie_matrix: The preprocessed user-item matrix
    :param index_to_movie_id: Mapping from matrix column index to movieId
    :param index_to_user_id: Mapping from matrix row index to userId
    :param k: The number of nearest neighbors to consider
    
    :return: A dictionary of userId -> recommended movie IDs
    """
    if not isinstance(user_movie_matrix, (np.ndarray, pd.DataFrame)):
        raise ValueError("user_movie_matrix must be a pandas DataFrame or a numpy array.")

    # Compute the user-user similarity using k-NN
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_movie_matrix)

    recommendations = {}

    # Iterate over each user in the matrix
    for user_idx in user_movie_matrix.index:
        distances, indices = knn.kneighbors(user_movie_matrix.loc[user_idx].values.reshape(1, -1), n_neighbors=k)

        # Get movies rated highly by similar users
        similar_users = user_movie_matrix.iloc[indices[0]]
        recommended_movie_indices = similar_users.mean(axis=0).sort_values(ascending=False).index[:10]

        # Convert movie indices back to movie IDs
        recommended_movie_ids = [index_to_movie_id[idx] for idx in recommended_movie_indices]
        recommendations[index_to_user_id[user_idx]] = recommended_movie_ids

    return recommendations
