# 🇮🇳 Indian Election Prediction using NLP and LSTM

A data-driven project that leverages **Natural Language Processing (NLP)** and **Long Short-Term Memory (LSTM)** neural networks to predict Indian election outcomes based on sentiment analysis of news articles.

---

## 📁 Project Structure

```
election_prediction/
│
├── config/                    # Configuration files
├── data/                      # Data storage
│   ├── raw/                   # Raw collected data
│   ├── interim/               # Intermediate cleaned data
│   └── processed/             # Final processed data
│
├── logs/                      # Log files
├── models/                    # Model files
│   └── saved/                 # Trained models (HDF5/Checkpoint)
│
├── notebooks/                 # Jupyter notebooks for EDA & prototyping
├── results/                   # Output predictions and evaluation
├── src/                       # Source code
│   ├── data_collection/       # News API scripts
│   ├── data_processing/       # Preprocessing scripts
│   ├── features/              # Feature extraction and sentiment
│   ├── models/                # LSTM training and inference
│   ├── visualization/         # Visualizations and plots
│   └── pipeline.py            # Full pipeline runner
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Python version and Clone the Repository
Use Python 3.9.6
```bash
git clone https://github.com/Harshita-gupta10/election-result-predict.git
cd election-results-predict
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required NLTK Resources
```
python -c "import nltk;
nltk.download('punkt');
nltk.download('stopwords');
nltk.download('wordnet');
nltk.download('vader_lexicon')"
```

### 4. Configure News API Key
- Sign up at [https://newsapi.org](https://newsapi.org) for a free API key.
- Set your API key as an environment variable:

#### Windows
```cmd
set NEWS_API_KEY=your_api_key_here
```

#### macOS/Linux
```bash
export NEWS_API_KEY=your_api_key_here
```

---

## 🚀 Usage Guide

### Step-by-Step Commands

```bash
# 1. Collect News Data
python src/data_collection/news_api_collector.py

# 2. Preprocess Text
python src/data_processing/text_preprocessor.py

# 3. Extract Features
python src/features/feature_engineering.py

# 4. Run Sentiment Analysis
python src/features/sentiment_analyzer.py

# 5. Prepare Data for LSTM
python src/data_processing/prepare_lstm_data.py

# 6. Train the LSTM Model
python src/models/lstm_model.py

# 7. Run Predictions
python src/models/predict.py

# 8. Visualize Results
python src/visualization/visualize_results.py
```

### Or Run the Full Pipeline
```bash
python src/pipeline.py
```

---

## 🧪 Troubleshooting & Tips

| Issue                         | Solution                                                                 |
|------------------------------|--------------------------------------------------------------------------|
| News API Errors               | Ensure your API key is valid and exported correctly.                    |
| Missing Files or Folders     | Run `setup_project.py` to auto-generate the directory structure.        |
| TensorFlow Compatibility     | Check Python/TensorFlow version; reinstall compatible versions.         |
| Out-of-Memory Errors         | Use smaller batch sizes or reduce data volume for training.             |
| Empty News Results           | Adjust News API query terms, regions, or date range.                    |

---

## 🌟 Future Enhancements

### ✅ Planned Features

- **🧠 Topic Modeling**: Use LDA/BERTopic to understand dominant political topics.
- **📍 Named Entity Recognition**: Extract politicians, parties, and constituency names.
- **📱 Social Media Signals**: Integrate Twitter/X, Reddit for multi-channel sentiment.
- **📊 Constituency-Level Modeling**: Use local news, demographics, and historical data.
- **📈 Time-Series Forecasting**: Model news momentum and election cycles.
- **🧩 Ensemble Models**: Blend LSTM with XGBoost or Random Forest.
- **🔍 Explainability**: Use SHAP/LIME for interpreting prediction logic.
- **📉 Auto Updates**: Schedule daily data collection and model retraining.
- **🧠 Transfer Learning**: Fine-tune BERT, RoBERTa for contextual accuracy.
- **🖥️ Interactive Dashboard**: Build a real-time web dashboard (Plotly/Dash/Streamlit).

---

## 📚 References

- News API: [https://newsapi.org](https://newsapi.org)
- NLTK: [https://www.nltk.org](https://www.nltk.org)
- TensorFlow: [https://www.tensorflow.org](https://www.tensorflow.org)
- Election Commission of India: [https://eci.gov.in](https://eci.gov.in)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for more information.
