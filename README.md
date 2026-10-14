# Customer Feedback Sentiment Analyzer

An end-to-end NLP application that classifies customer reviews into positive, negative, or neutral sentiment using a TF-IDF and Logistic Regression baseline model. The model is served via a FastAPI REST API and visualized through an interactive Streamlit dashboard.

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn (TfidfVectorizer + LogisticRegression), pandas, joblib
- **API:** FastAPI, uvicorn, python-multipart
- **Dashboard:** Streamlit, requests

## Project Structure

```
sentiment-analyzer/
├── data/
│   └── amazon_alexa.tsv       # Dataset (placed manually by user; not in git)
├── model/
│   └── sentiment_model.joblib # Trained model pipeline
├── results/
│   ├── metrics.csv            # Classification report metrics
│   └── test_predictions.csv   # Model test predictions with confidence scores
├── preprocess.py              # Text cleaning function
├── train.py                   # Data loading, EDA, cleaning, training, and evaluation
├── api.py                     # FastAPI REST API
├── dashboard.py               # Streamlit web dashboard
├── sample_reviews.csv         # Sample dataset for bulk upload demo
├── requirements.txt           # Pinned dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

## Dataset

This project uses the **Amazon Alexa Reviews** dataset from Kaggle (~3,150 reviews).

1. Download `amazon_alexa.tsv` from Kaggle.
2. Create the `data/` directory if needed and place `amazon_alexa.tsv` inside `data/`.

## Setup and How to Run

1. **Create and activate virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```powershell
   python train.py
   ```
   This prints EDA statistics, cleaning progress, real evaluation metrics, and saves `model/sentiment_model.joblib` and evaluation files in `results/`.

4. **Run the FastAPI service:**
   ```powershell
   uvicorn api:app --reload
   ```
   API docs are available at `http://127.0.0.1:8000/docs`.

5. **Run the Streamlit dashboard:**
   ```powershell
   streamlit run dashboard.py
   ```
   Opens the interactive dashboard in your browser.

## API Endpoints

### 1. `GET /`
Health check endpoint.
- **Example Response:**
  ```json
  {"status": "ok"}
  ```

### 2. `POST /predict`
Analyzes a single review text.
- **Example Request:**
  ```json
  {"text": "Love my Echo speaker, sounds amazing!"}
  ```
- **Example Response:**
  ```json
  {
    "text": "Love my Echo speaker, sounds amazing!",
    "sentiment": "positive",
    "confidence": 0.9860
  }
  ```

### 3. `POST /predict-csv`
Analyzes a CSV file containing reviews (max 5000 rows). Must contain a `review` column; a `date` column is optional.
- **Example Request:** Multipart form upload with file `sample_reviews.csv`.
- **Example Response:**
  ```json
  [
    {
      "review": "Absolutely love this smart speaker, the sound is amazing!",
      "sentiment": "positive",
      "confidence": 0.8948,
      "date": "2023-01-15"
    }
  ]
  ```

## Results

*Note: The dataset consists mostly of 5-star reviews, leading to significant class imbalance addressed using balanced class weights in Logistic Regression.*

Paste the real evaluation metrics printed by running `python train.py`:
- **Accuracy:** `0.8649`
- **Macro F1:** `0.5839`

Detailed evaluation metrics and predictions are saved in:
- `results/metrics.csv`
- `results/test_predictions.csv`

## Live Demo Links

- **API:** `https://customer-feedback-sentiment-analyzer-337g.onrender.com`
- **Dashboard:** `https://customer-feedback-sentiment-analyzer.streamlit.app/`
