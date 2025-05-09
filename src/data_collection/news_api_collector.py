# src/data_collection/news_api_collector.py
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time

def collect_election_news(api_key, days_back=30):
    """
    Collect Indian election news from News API
    """
    base_url = "https://newsapi.org/v2/everything"
    
    # Calculate date range (from x days ago until today)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # Search queries related to Indian elections
    search_queries = [
        "Indian elections", "India elections", "BJP India", 
        "Congress India election", "Modi election", 
        "Lok Sabha election", "Indian political parties",
        "Election Commission India"
    ]
    
    all_articles = []
    
    for query in search_queries:
        params = {
            'q': query,
            'from': start_date,
            'to': end_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,  # Maximum allowed by News API
            'apiKey': api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'ok':
                    print(f"Collected {len(data['articles'])} articles for query: {query}")
                    all_articles.extend(data['articles'])
                else:
                    print(f"Error in API response: {data['message']}")
            else:
                print(f"Error fetching data: {response.status_code}")
                
            # Respect API rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
    
    # Remove duplicates based on title
    unique_articles = []
    seen_titles = set()
    
    for article in all_articles:
        if article['title'] not in seen_titles:
            unique_articles.append(article)
            seen_titles.add(article['title'])
    
    # Save to CSV and JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/raw/election_news_{timestamp}.csv"
    json_path = f"data/raw/election_news_{timestamp}.json"
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(unique_articles)
    df.to_csv(csv_path, index=False)
    
    # Save full JSON data
    with open(json_path, 'w') as f:
        json.dump(unique_articles, f)
    
    print(f"Collected {len(unique_articles)} unique articles")
    print(f"Data saved to {csv_path} and {json_path}")
    
    return df

if __name__ == "__main__":
    # Replace with your News API key
    API_KEY = "your_news_api_key_here"
    collect_election_news(API_KEY)