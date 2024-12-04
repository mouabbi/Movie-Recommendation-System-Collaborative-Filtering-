import os
import pickle
import pandas as pd
from ml.preprocessing.data_preprocessing import generate_id_mappings
from ml.recommender import new_user_recommendation,home_page_recommendations,recommendations_after_new_movie
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
        # Generate and save maps
        self.generate_and_save_maps()
        
        # Generate user-item matrix
        self.user_item_matrix = self.generate_user_item_matrix(self.ratings, self.movies)
        self.save_precomputed_data('user_item_matrix.pkl', self.user_item_matrix)
        
        # Generate item-item matrix and store it
        self.item_item_matrix = item_item_collaborative_filtering(self.user_item_matrix, self.index_to_movie_id)
        self.save_precomputed_data('item_item_matrix.pkl', self.item_item_matrix)
        
        # Generate user-user matrix and store it
        self.user_user_matrix = user_user_collaborative_filtering(self.user_item_matrix, self.index_to_movie_id, self.index_to_user_id)
        self.save_precomputed_data('user_user_matrix.pkl', self.user_user_matrix)
        
        # Generate SVD predictions and store them
        self.svd_predictions = svd_recommendation(self.user_item_matrix, n_components=50)
        self.save_precomputed_data('svd_predictions.pkl', self.svd_predictions)

    def generate_and_save_maps(self):
        """ Generate and save the four required maps. """
        movie_id_to_index, index_to_movie_id, user_id_to_index, index_to_user_id = generate_id_mappings(self.ratings)
        
        # Save the generated maps
        self.save_precomputed_data('movie_id_to_index.pkl', movie_id_to_index)
        self.save_precomputed_data('index_to_movie_id.pkl', index_to_movie_id)
        self.save_precomputed_data('user_id_to_index.pkl', user_id_to_index)
        self.save_precomputed_data('index_to_user_id.pkl', index_to_user_id)

    def save_precomputed_data(self, filename, data):
        """ Save precomputed data to the specified file. """
        file_path = os.path.join('ml/precomputed', filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {file_path}")
    
    def recommend_for_new_user(self, watched_movie_id=None, top_n=10, diversity_factor=0.3):
        """
        Recommend movies for a new user by using the new_user_recommendation function.
        """
        return new_user_recommendation(
            ratings=self.ratings, 
            item_item_recommendations=self.item_item_matrix, 
            watched_movie_id=watched_movie_id, 
            top_n=top_n, 
            diversity_factor=diversity_factor
        )
        
    def recommend_for_old_user(self, user_id, weights=(0.4, 0.3, 0.3), top_n=50, global_top_movies=5):
        """
        Recommends movies for an old user by calling the home_page_recommendations function.
        :param user_id: ID of the user to generate recommendations for
        :param weights: Weights for blending the recommendation methods (item-item, user-user, SVD)
        :param top_n: Number of movies to recommend
        :param global_top_movies: Number of global top rated movies to include
        :return: List of recommended movie IDs
        """
        # Call home_page_recommendations for old user
        recommendations = home_page_recommendations(
            user_id=user_id,
            ratings=self.ratings,
            user_movie_matrix=self.user_item_matrix,
            item_item_recommendations=self.item_item_matrix,
            user_user_recommendations=self.user_user_matrix,
            svd_prediction_matrix=self.svd_predictions,
            user_id_to_index=self.user_id_to_index,
            index_to_movie_id=self.index_to_movie_id,
            movie_id_to_index=self.movie_id_to_index,
            weights=weights,
            top_n=top_n,
            global_top_movies=global_top_movies
        )

        return recommendations
    
    
    def recommend_similar_movie_for_old_user(self, user_id, new_movie_id, top_n=10, weights=(0.7, 0.2), rating_threshold=1):
        """
        Recommends similar movies for an old user after watching a new movie.
        :param user_id: ID of the user
        :param new_movie_id: ID of the newly watched movie
        :param top_n: Number of movies to recommend
        :param weights: Weights for blending the recommendation methods (item-item, SVD)
        :param rating_threshold: Minimum predicted rating threshold for SVD-based recommendations
        :return: List of recommended movie IDs
        """
        recommendations = recommendations_after_new_movie(
            user_id=user_id, 
            new_movie_id=new_movie_id, 
            ratings=self.ratings, 
            item_item_recommendations=self.item_item_matrix, 
            svd_prediction_matrix=self.svd_predictions, 
            user_movie_matrix=self.user_item_matrix, 
            user_id_to_index=self.user_id_to_index, 
            index_to_movie_id=self.index_to_movie_id, 
            weights=weights, 
            top_n=top_n, 
            rating_threshold=rating_threshold
        )

        return recommendations
    