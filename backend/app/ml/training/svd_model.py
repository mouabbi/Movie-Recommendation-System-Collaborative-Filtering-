import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def svd_recommendation(user_movie_matrix,  n_components=100):
    """
    Apply SVD for matrix factorization using preprocessed data.
    :param user_movie_matrix: The preprocessed user-item matrix
    :param n_components: Number of latent features to learn
    
    :return: A matrix of predictions (user-item)
    """
    # Apply SVD
    svd = TruncatedSVD(n_components=n_components,random_state=10)
    latent_matrix = svd.fit_transform(user_movie_matrix)
    
    # Get predictions
    prediction_matrix = np.dot(latent_matrix, svd.components_)
    
    return prediction_matrix
