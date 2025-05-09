# src/features/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import os

class FeatureEngineering:
    def __init__(self, max_features=5000, n_components=300):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Fit and transform the text data
        """
        # Convert to TF-IDF features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Apply dimensionality reduction with SVD (similar to word embeddings)
        svd_features = self.svd.fit_transform(tfidf_matrix)
        
        self.is_fitted = True
        
        # Explained variance
        explained_variance = self.svd.explained_variance_ratio_.sum()
        print(f"Explained variance with {self.svd.n_components} components: {explained_variance:.2%}")
        
        return svd_features
    
    def transform(self, texts):
        """
        Transform new text data using fitted vectorizer and SVD
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        svd_features = self.svd.transform(tfidf_matrix)
        
        return svd_features
    
    def save_models(self, output_dir='models/saved'):
        """
        Save the fitted vectorizer and SVD model
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted yet. Call fit_transform first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the TF-IDF vectorizer
        with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save the SVD model
        with open(os.path.join(output_dir, 'svd_model.pkl'), 'wb') as f:
            pickle.dump(self.svd, f)
        
        print(f"Feature engineering models saved to {output_dir}")
    
    @classmethod
    def load_models(cls, model_dir='models/saved'):
        """
        Load previously fitted vectorizer and SVD model
        """
        instance = cls()
        
        # Load the TF-IDF vectorizer
        with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            instance.tfidf_vectorizer = pickle.load(f)
        
        # Load the SVD model
        with open(os.path.join(model_dir, 'svd_model.pkl'), 'rb') as f:
            instance.svd = pickle.load(f)
        
        instance.is_fitted = True
        
        return instance

# Example usage
if __name__ == "__main__":
    # Load processed data
    import glob
    
    processed_files = glob.glob('data/processed/processed_news_*.csv')
    if processed_files:
        latest_file = max(processed_files, key=os.path.getctime)
        print(f"Creating features from {latest_file}")
        
        df = pd.read_csv(latest_file)
        
        if 'combined_text' in df.columns:
            # Create features
            feature_eng = FeatureEngineering(max_features=10000, n_components=300)
            features = feature_eng.fit_transform(df['combined_text'].fillna(''))
            
            # Save the feature models
            feature_eng.save_models()
            
            # Save the features
            feature_df = pd.DataFrame(
                features, 
                columns=[f'feature_{i}' for i in range(features.shape[1])]
            )
            
            # Add article IDs or other metadata if needed
            if 'publishedAt' in df.columns:
                feature_df['published_date'] = df['publishedAt']
            if 'source' in df.columns and 'name' in df['source'].iloc[0]:
                feature_df['source'] = df['source'].apply(lambda x: x.get('name', ''))
            elif 'source.name' in df.columns:
                feature_df['source'] = df['source.name']
            
            # Save features
            timestamp = latest_file.split('_')[-1].split('.')[0]
            feature_path = f"data/interim/features_{timestamp}.csv"
            feature_df.to_csv(feature_path, index=False)
            
            print(f"Features saved to {feature_path}")
        else:
            print("No 'combined_text' column found in the processed data")
    else:
        print("No processed data files found")