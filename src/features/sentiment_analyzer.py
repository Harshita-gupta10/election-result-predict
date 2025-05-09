# src/features/sentiment_analyzer.py
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import os

# Download VADER lexicon
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self, use_transformer=True):
        self.vader = SentimentIntensityAnalyzer()
        self.use_transformer = use_transformer
        
        if use_transformer:
            # Use a more advanced transformer-based sentiment analyzer
            # Note: This will download a pre-trained model on first use
            self.transformer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True, 
                max_length=512
            )
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of the given text
        
        Returns:
            Dictionary with sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'vader_compound': 0,
                'vader_pos': 0,
                'vader_neg': 0,
                'vader_neu': 0,
                'textblob_polarity': 0,
                'textblob_subjectivity': 0,
                'transformer_label': 'NEUTRAL',
                'transformer_score': 0.5
            }
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        result = {
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity
        }
        
        # Transformer-based sentiment (if enabled)
        if self.use_transformer:
            try:
                # Limit text length for transformer
                transformer_result = self.transformer(text[:512])[0]
                result['transformer_label'] = transformer_result['label']
                result['transformer_score'] = transformer_result['score']
            except Exception as e:
                print(f"Transformer error: {str(e)}")
                result['transformer_label'] = 'ERROR'
                result['transformer_score'] = 0.5
        
        return result
    
    def analyze_dataframe(self, df, text_column='combined_text'):
        """
        Analyze sentiment for all texts in the dataframe
        
        Args:
            df: pandas DataFrame
            text_column: column name containing text to analyze
            
        Returns:
            DataFrame with added sentiment columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply sentiment analysis to each text
        sentiment_results = result_df[text_column].apply(self.analyze_text)
        
        # Convert results to DataFrame columns
        sentiment_df = pd.json_normalize(sentiment_results)
        
        # Add sentiment columns to result DataFrame
        for column in sentiment_df.columns:
            result_df[f'sentiment_{column}'] = sentiment_df[column]
        
        return result_df

# Example usage
if __name__ == "__main__":
    # Load processed data
    import glob
    
    processed_files = glob.glob('data/processed/processed_news_*.csv')
    if processed_files:
        latest_file = max(processed_files, key=os.path.getctime)
        print(f"Analyzing sentiment in {latest_file}")
        
        df = pd.read_csv(latest_file)
        
        if 'combined_text' in df.columns:
            # Analyze sentiment
            analyzer = SentimentAnalyzer(use_transformer=True)
            sentiment_df = analyzer.analyze_dataframe(df)
            
            # Save results
            timestamp = latest_file.split('_')[-1].split('.')[0]
            sentiment_path = f"data/interim/sentiment_{timestamp}.csv"
            sentiment_df.to_csv(sentiment_path, index=False)
            
            print(f"Sentiment analysis saved to {sentiment_path}")
        else:
            print("No 'combined_text' column found in the processed data")
    else:
        print("No processed data files found")