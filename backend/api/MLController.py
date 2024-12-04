import os
import pickle
import numpy as np
from ml.recommender import home_page_recommendations
from ml.Training import item_item_collaborative_filtering,user_user_collaborative_filtering ,svd_recommendation
from ml.precomputed import save_precomputed_data, load_precomputed_data


class MLController:
    def __init__(self, ratings, user_id=None):
        self.ratings = ratings
        self.user_id = user_id
        self.item_item_matrix = None
        self.user_user_matrix = None
        self.user_item_matrix = None
        self.svd_predictions = None
        self.movie_id_to_index = None
        self.index_to_movie_id = None

        # Load precomputed data or initialize it
        self.load_or_initialize_precomputed_data()

    def load_or_initialize_precomputed_data(self):
        """ Load precomputed matrices and maps if they exist. """
        precomputed_data_path = 'ml/precomputed/'
        
        # Check and load precomputed data
        if os.path.exists(precomputed_data_path):
            self.load_precomputed_data()
        else:
            self.update_precomputed_data()

    def load_precomputed_data(self):
        """ Load precomputed matrices and maps from disk. """
        try:
            with open(os.path.join('ml/precomputed', 'item_item_matrix.pkl'), 'rb') as f:
                self.item_item_matrix = pickle.load(f)
            with open(os.path.join('ml/precomputed', 'user_user_matrix.pkl'), 'rb') as f:
                self.user_user_matrix = pickle.load(f)
            with open(os.path.join('ml/precomputed', 'user_item_matrix.pkl'), 'rb') as f:
                self.user_item_matrix = pickle.load(f)
            with open(os.path.join('ml/precomputed/maps', 'movie_id_to_index.pkl'), 'rb') as f:
                self.movie_id_to_index = pickle.load(f)
            with open(os.path.join('ml/precomputed/maps', 'index_to_movie_id.pkl'), 'rb') as f:
                self.index_to_movie_id = pickle.load(f)
            with open(os.path.join('ml/precomputed', 'svd_predictions.pkl'), 'rb') as f:
                self.svd_predictions = pickle.load(f)
        except Exception as e:
            print(f"Error loading precomputed data: {e}")
            self.update_precomputed_data()

    def update_precomputed_data(self):
        """ Recalculate and store matrices and maps if not found. """
        print("Updating precomputed data...")
        
        # Calculate user-item matrix and store it
        self.user_item_matrix = calculate_user_item_matrix(self.ratings)
        save_precomputed_data('user_item_matrix.pkl', self.user_item_matrix)
        
        # Calculate item-item matrix and store it
        self.item_item_matrix = calculate_item_item_matrix(self.user_item_matrix)
        save_precomputed_data('item_item_matrix.pkl', self.item_item_matrix)
        
        # Calculate user-user matrix and store it
        self.user_user_matrix = calculate_user_user_matrix(self.user_item_matrix)
        save_precomputed_data('user_user_matrix.pkl', self.user_user_matrix)
        
        # Calculate SVD predictions and store it
        self.svd_predictions = self.calculate_svd_predictions()
        save_precomputed_data('svd_predictions.pkl', self.svd_predictions)
        
        # Calculate and store movie ID to index and index to movie ID maps
        self.movie_id_to_index, self.index_to_movie_id = self.generate_movie_maps()
        save_precomputed_data('movie_id_to_index.pkl', self.movie_id_to_index)
        save_precomputed_data('index_to_movie_id.pkl', self.index_to_movie_id)

    def generate_movie_maps(self):
        """ Generate and return movie maps (ID to index and index to ID). """
        movie_ids = self.ratings['movieId'].unique()
        movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}
        return movie_id_to_index, index_to_movie_id

    def calculate_svd_predictions(self):
        """ Calculate and return the SVD-based predictions. """
        # Assuming SVD prediction logic is implemented in ml.training
        return calculate_svd_predictions(self.user_item_matrix)

    def get_home_page_recommendations(self, top_n=50):
        """ Get personalized home page recommendations for the user. """
        if self.user_id:
            return home_page_recommendations_with_diversity(
                user_id=self.user_id,
                ratings=self.ratings,
                user_movie_matrix=self.user_item_matrix,
                item_item_recommendations=self.item_item_matrix,
                user_user_recommendations=self.user_user_matrix,
                svd_prediction_matrix=self.svd_predictions,
                top_n=top_n
            )
        else:
            print("User ID is required for personalized recommendations.")
            return []
