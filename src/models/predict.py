# src/models/predict.py
import numpy as np
import pandas as pd
import os
import pickle
import glob
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import ElectionLSTM
from data_processing.text_preprocessor import TextPreprocessor
from features.feature_engineering import FeatureEngineering
from features.sentiment_analyzer import SentimentAnalyzer
from data_processing.prepare_lstm_data import prepare_lstm_data

def load_latest_model():
    """
    Load the latest trained model
    """
    model_files = glob.glob('models/saved/lstm_election_model_*.h5')
    if not model_files:
        raise FileNotFoundError("No trained models found")
    
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model from {latest_model}")
    
    return ElectionLSTM.load_model(latest_model)

def prepare_new_data(news_data, sequence_length=7):
    """
    Prepare new news data for prediction
    
    Args:
        news_data: DataFrame with news articles
        sequence_length: Length of sequences for LSTM input
        
    Returns:
        Processed sequences ready for prediction
    """
    # Load the preprocessing pipeline
    preprocessor = TextPreprocessor()
    feature_eng = FeatureEngineering.load_models()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Load the feature scaler
    scaler_files = glob.glob('data/processed/feature_scaler.pkl')
    if not scaler_files:
        raise FileNotFoundError("Feature scaler not found")
    
    with open(scaler_files[0], 'rb') as f:
        scaler = pickle.load(f)
    
    # Process text
    processed_df = preprocessor.process_dataframe(news_data)
    
    # Extract features
    features = feature_eng.transform(processed_df['combined_text'].fillna(''))
    feature_df = pd.DataFrame(
        features, 
        columns=[f'feature_{i}' for i in range(features.shape[1])]
    )
    
    # Add any metadata columns needed
    if 'publishedAt' in news_data.columns:
        feature_df['published_date'] = news_data['publishedAt']
    
    # Analyze sentiment
    sentiment_df = sentiment_analyzer.analyze_dataframe(processed_df)
    
    # Select sentiment columns
    sentiment_cols = [
        col for col in sentiment_df.columns 
        if col.startswith('sentiment_') and not col.endswith('label')
    ]
    
    # Combine features and sentiment
    for col in sentiment_cols:
        feature_df[col] = sentiment_df[col].values[:len(feature_df)]
    
    # Select input columns (must match the columns used in training)
    input_cols = [col for col in feature_df.columns if col.startswith('feature_') or col.startswith('sentiment_')]
    
    # Scale features
    feature_df[input_cols] = scaler.transform(feature_df[input_cols])
    
    # Create sequences
    if len(feature_df) < sequence_length:
        raise ValueError(f"Not enough data to create sequences. Need at least {sequence_length} articles.")
    
    X_sequences = []
    
    for i in range(len(feature_df) - sequence_length + 1):
        X_sequences.append(feature_df[input_cols].values[i:i+sequence_length])
    
    return np.array(X_sequences)

def predict_from_news(news_data):
    """
    Make election predictions from news data
    """
    # Load model
    model = load_latest_model()
    
    # Get sequence length from model input shape
    sequence_length = model.model.input_shape[1]
    
    # Prepare data
    X_sequences = prepare_new_data(news_data, sequence_length)
    
    # Make predictions
    predictions = model.predict_election(X_sequences)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'sequence_id': range(len(predictions)),
        'bjp_win_probability': predictions.flatten(),
        'predicted_winner': ['BJP' if p > 0.5 else 'Opposition' for p in predictions.flatten()]
    })
    
    # Calculate aggregate prediction
    avg_probability = results['bjp_win_probability'].mean()
    overall_prediction = 'BJP' if avg_probability > 0.5 else 'Opposition'
    confidence = max(avg_probability, 1 - avg_probability)
    
    print(f"Overall prediction: {overall_prediction} with {confidence:.2%} confidence")
    
    return results, overall_prediction, confidence

# Example usage
if __name__ == "__main__":
    # Load some news data - this would be replaced with real-time data in production
    latest_news_files = glob.glob('data/raw/election_news_*.csv')
    
    if latest_news_files:
        latest_file = max(latest_news_files, key=os.path.getctime)
        print(f"Using news data from {latest_file}")
        
        news_df = pd.read_csv(latest_file)
        
        # Use most recent news articles
        recent_news = news_df.sort_values('publishedAt', ascending=False).head(30)
        
        results, winner, confidence = predict_from_news(recent_news)
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv(f'results/prediction_results_{timestamp}.csv', index=False)
        
        # Save overall prediction
        with open(f'results/overall_prediction_{timestamp}.txt', 'w') as f:
            f.write(f"Predicted Winner: {winner}\n")
            f.write(f"Confidence: {confidence:.2%}\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        
        print(f"Results saved to results/prediction_results_{timestamp}.csv")
    else:
        print("No news data files found")