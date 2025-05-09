# src/data_processing/prepare_lstm_data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import glob

def prepare_lstm_data(feature_path, sentiment_path, output_dir='data/processed', test_size=0.2):
    """
    Prepare data for LSTM model by combining features and sentiment
    and creating sequences
    """
    # Load feature data
    print(f"Loading features from {feature_path}")
    feature_df = pd.read_csv(feature_path)
    
    # Load sentiment data
    print(f"Loading sentiment from {sentiment_path}")
    sentiment_df = pd.read_csv(sentiment_path)
    
    # Ensure we have a date column for ordering
    date_col = None
    for col_name in ['published_date', 'publishedAt', 'date']:
        if col_name in feature_df.columns:
            date_col = col_name
            break
    
    if date_col is None:
        print("Warning: No date column found. Using original order.")
    else:
        # Convert to datetime and sort
        feature_df[date_col] = pd.to_datetime(feature_df[date_col], errors='coerce')
        feature_df = feature_df.sort_values(by=date_col)
    
    # Select feature columns
    feature_cols = [col for col in feature_df.columns if col.startswith('feature_')]
    
    if not feature_cols:
        raise ValueError("No feature columns found in feature data")
    
    # Select sentiment columns from sentiment data
    sentiment_cols = [
        col for col in sentiment_df.columns 
        if col.startswith('sentiment_') and not col.endswith('label')
    ]
    
    if not sentiment_cols:
        raise ValueError("No sentiment columns found in sentiment data")
    
    # Combine features and sentiment
    # First, ensure we can match rows between the two dataframes
    if 'combined_text' in sentiment_df.columns and 'combined_text' in feature_df.columns:
        # Join on text content
        merged_df = pd.merge(
            feature_df, 
            sentiment_df[['combined_text'] + sentiment_cols],
            on='combined_text', 
            how='inner'
        )
    else:
        # Assume same order if we can't match otherwise
        print("Warning: No common key for joining. Assuming same order.")
        merged_df = feature_df.copy()
        for col in sentiment_cols:
            merged_df[col] = sentiment_df[col].values[:len(merged_df)]
    
    print(f"Combined data shape: {merged_df.shape}")
    
    # Create target variable - party victory prediction
    # For now, we'll use a synthetic target based on sentiment
    # In a real project, you'd use labeled data with actual election outcomes
    
    # Example: Define a synthetic target
    # Positive sentiment towards BJP = 1, otherwise = 0
    # This is just a placeholder - you'll need real labeled data
    merged_df['target_bjp_win'] = (
        (merged_df['sentiment_vader_compound'] > 0.2) & 
        (merged_df['sentiment_textblob_polarity'] > 0.1)
    ).astype(int)
    
    # Select input columns
    input_cols = feature_cols + sentiment_cols
    
    # Scale features
    scaler = MinMaxScaler()
    merged_df[input_cols] = scaler.fit_transform(merged_df[input_cols])
    
    # Save the scaler
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create sequences for LSTM
    sequence_length = 7  # Use 7 days of news as input sequence
    
    X_sequences = []
    y_values = []
    
    for i in range(len(merged_df) - sequence_length):
        X_sequences.append(merged_df[input_cols].values[i:i+sequence_length])
        y_values.append(merged_df['target_bjp_win'].values[i+sequence_length])
    
    X = np.array(X_sequences)
    y = np.array(y_values)
    
    print(f"Created {len(X)} sequences with shape {X.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Save prepared data
    timestamp = os.path.basename(feature_path).split('_')[-1].split('.')[0]
    
    np.save(os.path.join(output_dir, f'X_train_{timestamp}.npy'), X_train)
    np.save(os.path.join(output_dir, f'X_test_{timestamp}.npy'), X_test)
    np.save(os.path.join(output_dir, f'y_train_{timestamp}.npy'), y_train)
    np.save(os.path.join(output_dir, f'y_test_{timestamp}.npy'), y_test)
    
    # Save metadata
    metadata = {
        'sequence_length': sequence_length,
        'input_columns': input_cols,
        'feature_columns': feature_cols,
        'sentiment_columns': sentiment_cols,
        'target_column': 'target_bjp_win',
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape,
        'y_test_shape': y_test.shape
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(os.path.join(output_dir, f'lstm_data_metadata_{timestamp}.csv'), index=False)
    
    print(f"Data preparation complete. Files saved to {output_dir}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Find the latest feature and sentiment files
    feature_files = glob.glob('data/interim/features_*.csv')
    sentiment_files = glob.glob('data/interim/sentiment_*.csv')
    
    if feature_files and sentiment_files:
        latest_feature = max(feature_files, key=os.path.getctime)
        latest_sentiment = max(sentiment_files, key=os.path.getctime)
        
        print(f"Using latest feature file: {latest_feature}")
        print(f"Using latest sentiment file: {latest_sentiment}")
        
        prepare_lstm_data(latest_feature, latest_sentiment)
    else:
        print("Missing required files")