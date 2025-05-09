# src/pipeline.py
import os
import sys
import glob
import pandas as pd
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('election_prediction_pipeline')

# Import all necessary modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection.news_api_collector import collect_election_news
from data_processing.text_preprocessor import TextPreprocessor
from features.feature_engineering import FeatureEngineering
from features.sentiment_analyzer import SentimentAnalyzer
from data_processing.prepare_lstm_data import prepare_lstm_data
from models.lstm_model import ElectionLSTM
from models.predict import predict_from_news
from visualization.visualize_results import plot_prediction_probabilities, plot_model_evaluation

def run_pipeline(api_key, days_back=30, retrain_model=False):
    """
    Run the complete election prediction pipeline
    
    Args:
        api_key: News API key
        days_back: Number of days to look back for news
        retrain_model: Whether to retrain the model or use the latest
        
    Returns:
        Dictionary with results and paths
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting pipeline run at {timestamp}")
    
    results = {}
    
    # Step 1: Collect news data
    logger.info("Collecting news data...")
    news_df = collect_election_news(api_key, days_back=days_back)
    results['raw_data_path'] = f"data/raw/election_news_{timestamp}.csv"
    
    # Step 2: Process text
    logger.info("Processing text data...")
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.process_dataframe(news_df)
    processed_path = f"data/processed/processed_news_{timestamp}.csv"
    processed_df.to_csv(processed_path, index=False)
    results['processed_data_path'] = processed_path
    
    # Step 3: Extract features
    logger.info("Extracting features...")
    feature_eng = FeatureEngineering(max_features=10000, n_components=300)
    features = feature_eng.fit_transform(processed_df['combined_text'].fillna(''))
    feature_eng.save_models()
    
    feature_df = pd.DataFrame(
        features, 
        columns=[f'feature_{i}' for i in range(features.shape[1])]
    )
    feature_df['published_date'] = processed_df['publishedAt']
    feature_path = f"data/interim/features_{timestamp}.csv"
    feature_df.to_csv(feature_path, index=False)
    results['feature_data_path'] = feature_path
    
    # Step 4: Sentiment analysis
    logger.info("Analyzing sentiment...")
    sentiment_analyzer = SentimentAnalyzer(use_transformer=True)
    sentiment_df = sentiment_analyzer.analyze_dataframe(processed_df)
    sentiment_path = f"data/interim/sentiment_{timestamp}.csv"
    sentiment_df.to_csv(sentiment_path, index=False)
    results['sentiment_data_path'] = sentiment_path
    
    # Step 5: Prepare LSTM data
    logger.info("Preparing data for LSTM model...")
    X_train, X_test, y_train, y_test = prepare_lstm_data(
        feature_path, 
        sentiment_path,
        output_dir='data/processed'
    )
    results['training_data_prepared'] = True
    
    # Step 6: Train or load model
    if retrain_model:
        logger.info("Training new LSTM model...")
        input_shape = X_train.shape[1:]
        model = ElectionLSTM(input_shape)
        model.build_model(lstm_units=128, dropout_rate=0.3)
        
        history = model.train(
            X_train, y_train, 
            X_test, y_test,
            epochs=50, 
            batch_size=32
        )
        
        model_path = model.save_model(f'lstm_election_model_{timestamp}')
        results['model_path'] = model_path
        results['model_trained'] = True
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        results['evaluation_metrics'] = metrics
        
        # Plot training history
        history_path = os.path.join('results', f'training_history_{timestamp}.png')
        model.plot_training_history(save_path=history_path)
        results['history_plot_path'] = history_path
    else:
        # Load existing model
        model_files = glob.glob('models/saved/lstm_election_model_*.h5')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            logger.info(f"Loading existing model from {latest_model}")
            model = ElectionLSTM.load_model(latest_model)
            results['model_path'] = latest_model
            results['model_trained'] = False
        else:
            logger.error("No existing models found and retrain_model=False")
            results['error'] = "No existing models found"
            return results
    
    # Step 7: Make predictions
    logger.info("Making predictions...")
    prediction_results, winner, confidence = predict_from_news(news_df)
    
    prediction_path = f'results/prediction_results_{timestamp}.csv'
    prediction_results.to_csv(prediction_path, index=False)
    results['prediction_path'] = prediction_path
    results['predicted_winner'] = winner
    results['prediction_confidence'] = confidence
    
    # Step 8: Visualize results
    logger.info("Generating visualizations...")
    plot_path = plot_prediction_probabilities(prediction_path)
    results['prediction_plot_path'] = plot_path
    
    # Find evaluation metrics if any
    eval_files = glob.glob('models/saved/evaluation_metrics_*.json')
    if eval_files:
        latest_eval = max(eval_files, key=os.path.getctime)
        eval_plot_path = plot_model_evaluation(latest_eval)
        results['evaluation_plot_path'] = eval_plot_path
    
    logger.info("Pipeline complete!")
    return results

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    
    # Get API key from environment or prompt
    api_key = os.environ.get('NEWS_API_KEY')
    if not api_key:
        api_key = input("Enter your News API key: ")
    
    # Run the pipeline
    results = run_pipeline(
        api_key=api_key,
        days_back=30,
        retrain_model=True  # Set to False to use existing model
    )
    
    print("\nPipeline results:")
    for key, value in results.items():
        print(f"{key}: {value}")