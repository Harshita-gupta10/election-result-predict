# src/data_processing/text_preprocessor.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, additional_stopwords=None):
        # Initialize lemmatizer and stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        
        # Add custom Indian election-related stopwords that don't provide sentiment
        indian_stopwords = {
            'india', 'indian', 'election', 'elections', 'vote', 'votes', 
            'voting', 'poll', 'polls', 'bjp', 'congress', 'party', 'parties'
        }
        
        self.stopwords.update(indian_stopwords)
        
        # Add any additional stopwords
        if additional_stopwords:
            self.stopwords.update(additional_stopwords)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text):
        """Complete text preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stopwords and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def process_dataframe(self, df, text_columns=None):
        """
        Process multiple text columns in a dataframe
        
        Args:
            df: pandas DataFrame
            text_columns: list of column names to process
        
        Returns:
            DataFrame with processed text columns
        """
        # Default to processing title and content if available
        if text_columns is None:
            text_columns = []
            for col in ['title', 'description', 'content']:
                if col in df.columns:
                    text_columns.append(col)
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Process each text column
        for column in text_columns:
            if column in processed_df.columns:
                processed_df[f'processed_{column}'] = processed_df[column].apply(self.preprocess_text)
        
        # Create a combined text field from all processed columns
        if len(text_columns) > 0:
            processed_columns = [f'processed_{col}' for col in text_columns if f'processed_{col}' in processed_df.columns]
            processed_df['combined_text'] = processed_df[processed_columns].apply(
                lambda row: ' '.join(row.dropna().astype(str)), axis=1
            )
        
        return processed_df

# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Load the most recent raw data file
    import glob
    import os
    
    raw_files = glob.glob('data/raw/election_news_*.csv')
    if raw_files:
        latest_file = max(raw_files, key=os.path.getctime)
        print(f"Processing {latest_file}")
        
        raw_df = pd.read_csv(latest_file)
        processed_df = preprocessor.process_dataframe(raw_df)
        
        # Save processed data
        timestamp = latest_file.split('_')[-1].split('.')[0]
        processed_path = f"data/processed/processed_news_{timestamp}.csv"
        processed_df.to_csv(processed_path, index=False)
        
        print(f"Processed data saved to {processed_path}")
    else:
        print("No raw data files found")