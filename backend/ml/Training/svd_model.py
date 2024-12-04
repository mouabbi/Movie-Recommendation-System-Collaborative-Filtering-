import numpy as np
from sklearn.decomposition import TruncatedSVD

def svd_recommendation(user_movie_matrix, n_components=50):
    """
    Apply Singular Value Decomposition (SVD) for matrix factorization 
    to generate recommendations.

    :param user_movie_matrix: DataFrame or numpy array representing the user-item matrix
    :param n_components: Number of latent features to learn (default is 50)

    :return: A numpy array representing the prediction matrix (user-item)
    """
    if not isinstance(user_movie_matrix, (np.ndarray, pd.DataFrame)):
        raise ValueError("user_movie_matrix must be a pandas DataFrame or a numpy array.")

    # Convert DataFrame to numpy array if necessary
    if isinstance(user_movie_matrix, pd.DataFrame):
        user_movie_matrix = user_movie_matrix.values

    # Apply SVD
    svd = TruncatedSVD(n_components=n_components)
    latent_matrix = svd.fit_transform(user_movie_matrix)

    # Calculate prediction matrix
    prediction_matrix = np.dot(latent_matrix, svd.components_)

    return prediction_matrix
