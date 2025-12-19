import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class SpotifyRecommender:
    def __init__(self, df):
        self.df = df
        self.feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                             'acousticness', 'instrumentalness', 'liveness', 
                             'valence', 'tempo']
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(df[self.feature_cols])
    
    def content_based_recommendations(self, track_indices, n_recommendations=10):
        """Seçilen şarkılara benzer şarkıları bul"""
        user_profile = self.scaled_features[track_indices].mean(axis=0).reshape(1, -1)
        
        similarities = cosine_similarity(user_profile, self.scaled_features)[0]
        
        mask = np.ones(len(similarities), dtype=bool)
        mask[track_indices] = False
        similarities_filtered = similarities.copy()
        similarities_filtered[~mask] = -1
        
        top_indices = similarities_filtered.argsort()[-n_recommendations:][::-1]
        
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations
    
    def collaborative_filtering(self, track_indices, n_recommendations=10):
        """Basit collaborative filtering simülasyonu"""
        similarity_matrix = cosine_similarity(self.scaled_features)
        
        similar_scores = similarity_matrix[track_indices].mean(axis=0)
        
        mask = np.ones(len(similar_scores), dtype=bool)
        mask[track_indices] = False
        similar_scores_filtered = similar_scores.copy()
        similar_scores_filtered[~mask] = -1
        
        top_indices = similar_scores_filtered.argsort()[-n_recommendations:][::-1]
        
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['cf_score'] = similar_scores[top_indices]
        
        return recommendations
    
    def hybrid_recommendations(self, track_indices, n_recommendations=10, alpha=0.6):
        """Content-based ve collaborative'i birleştir"""
        user_profile = self.scaled_features[track_indices].mean(axis=0).reshape(1, -1)
        content_scores = cosine_similarity(user_profile, self.scaled_features)[0]
        
        similarity_matrix = cosine_similarity(self.scaled_features)
        collab_scores = similarity_matrix[track_indices].mean(axis=0)
        
        hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores
        
        mask = np.ones(len(hybrid_scores), dtype=bool)
        mask[track_indices] = False
        hybrid_scores_filtered = hybrid_scores.copy()
        hybrid_scores_filtered[~mask] = -1
        
        top_indices = hybrid_scores_filtered.argsort()[-n_recommendations:][::-1]
        
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['hybrid_score'] = hybrid_scores[top_indices]
        
        return recommendations
    
    def get_user_profile(self, track_indices):
        """Kullanıcının müzik profilini çıkar"""
        selected_songs = self.df.iloc[track_indices]
        profile = selected_songs[self.feature_cols].mean()
        return profile