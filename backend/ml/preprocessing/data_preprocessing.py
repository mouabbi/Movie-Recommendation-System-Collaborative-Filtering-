import os
import pickle
from ml.preprocessing.data_preprocessing import generate_id_mappings, generate_user_item_matrix
from ml.recommender import home_page_recommendations
from ml.Training import item_item_collaborative_filtering, user_user_collaborative_filtering, svd_recommendation


class MLController:
    def __init__(self, ratings, movies, user_id=None):
        self.ratings = ratings
        self.movies = movies
        self.user_id = user_id
        self.item_item_matrix = None
        self.user_user_matrix = None
        self.user_item_matrix = None
        self.svd_predictions = None
        self.movie_id_to_index = None
        self.index_to_movie_id = None
        self.user_id_to_index = None
        self.index_to_user_id = None

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
            with open(os.path.join('ml/precomputed/maps', 'user_id_to_index.pkl'), 'rb') as f:
                self.user_id_to_index = pickle.load(f)
            with open(os.path.join('ml/precomputed/maps', 'index_to_user_id.pkl'), 'rb') as f:
                self.index_to_user_id = pickle.load(f)
            with open(os.path.join('ml/precomputed', 'svd_predictions.pkl'), 'rb') as f:
                self.svd_predictions = pickle.load(f)
        except Exception as e:
            print(f"Error loading precomputed data: {e}")
            self.update_precomputed_data()

    def update_precomputed_data(self):
        """ Recalculate and store matrices and maps if not found. """
        print("Updating precomputed data...")
        
        # Generate user-item matrix
        self.user_item_matrix = generate_user_item_matrix(self.ratings, self.movies)
        self.save_precomputed_data('user_item_matrix.pkl', self.user_item_matrix)
        
        # Calculate item-item matrix and store it
        self.item_item_matrix = self.calculate_item_item_matrix(self.user_item_matrix)
        self.save_precomputed_data('item_item_matrix.pkl', self.item_item_matrix)
        
        # Calculate user-user matrix and store it
        self.user_user_matrix = self.calculate_user_user_matrix(self.user_item_matrix)
        self.save_precomputed_data('user_user_matrix.pkl', self.user_user_matrix)
        
        # Calculate SVD predictions and store it
        self.svd_predictions = self.calculate_svd_predictions()
        self.save_precomputed_data('svd_predictions.pkl', self.svd_predictions)
        
        # Generate and save maps
        self.generate_and_save_maps()

    def generate_and_save_maps(self):
        """ Generate and save the four required maps. """
        movie_id_to_index, index_to_movie_id, user_id_to_index, index_to_user_id = generate_id_mappings(self.ratings)
        
        # Save the generated maps
        self.save_precomputed_data('movie_id_to_index.pkl', movie_id_to_index)
        self.save_precomputed_data('index_to_movie_id.pkl', index_to_movie_id)
        self.save_precomputed_data('user_id_to_index.pkl', user_id_to_index)
        self.save_precomputed_data('index_to_user_id.pkl', index_to_user_id)

    def calculate_item_item_matrix(self, user_item_matrix):
        """ Calculate the item-item similarity matrix using collaborative filtering. """
        return item_item_collaborative_filtering(user_item_matrix)

    def calculate_user_user_matrix(self, user_item_matrix):
        """ Calculate the user-user similarity matrix using collaborative filtering. """
        return user_user_collaborative_filtering(user_item_matrix)

    def calculate_svd_predictions(self):
        """ Calculate and return the SVD-based predictions. """
        return svd_recommendation(self.user_item_matrix)

    def get_home_page_recommendations(self, top_n=50):
        """ Get personalized home page recommendations for the user. """
        if self.user_id:
            return home_page_recommendations(
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

    def save_precomputed_data(self, filename, data):
        """ Save precomputed data to the specified file. """
        file_path = os.path.join('ml/precomputed', filename)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {file_path}")
