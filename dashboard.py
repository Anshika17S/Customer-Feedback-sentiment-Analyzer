import os
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# 1. Title and description
st.title("Customer Feedback Sentiment Analyzer")
st.write("Classify customer reviews as positive, negative, or neutral using a TF-IDF and Logistic Regression model.")

# 2. Single review section
st.header("Single review")
review_text = st.text_area("Enter review text:")

if st.button("Analyze Review"):
    if not review_text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Analyzing review..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": review_text},
                    timeout=90
                )
            if response.status_code == 200:
                result = response.json()
                st.write(f"**Sentiment:** {result['sentiment']}")
                st.write(f"**Confidence:** {result['confidence']}")
            else:
                st.error("API call failed with status code " + str(response.status_code))
        except Exception:
            st.error("Could not connect to the API. Make sure the API server is running.")

# 3. Bulk CSV analysis section
st.header("Bulk CSV analysis")
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])

if uploaded_file is not None:
    try:
        with st.spinner("Analyzing CSV..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(
                f"{API_URL}/predict-csv",
                files=files,
                timeout=90
            )

        if response.status_code == 200:
            data = response.json()
            res_df = pd.DataFrame(data)

            # Total reviews and counts of each sentiment
            st.write(f"**Total reviews:** {len(res_df)}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", int((res_df["sentiment"] == "positive").sum()))
            col2.metric("Neutral", int((res_df["sentiment"] == "neutral").sum()))
            col3.metric("Negative", int((res_df["sentiment"] == "negative").sum()))

            # Sentiment distribution bar chart
            st.subheader("Sentiment distribution")
            st.bar_chart(res_df["sentiment"].value_counts())

            # Sentiment trend line chart (if date column exists)
            if "date" in res_df.columns:
                st.subheader("Sentiment trend")
                trend_df = res_df.copy()
                trend_df["date"] = pd.to_datetime(trend_df["date"])
                trend_df["month"] = trend_df["date"].dt.strftime("%Y-%m")
                monthly_counts = trend_df.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
                st.line_chart(monthly_counts)

            # Results table
            st.subheader("Results")
            st.dataframe(res_df)

            # Download button
            csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_bytes,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
        else:
            st.error("API error: " + response.text)
    except Exception:
        st.error("Could not connect to the API. Make sure the API server is running.")

# 4. Model performance section
st.header("Model performance")
metrics_file = "results/metrics.csv"
predictions_file = "results/test_predictions.csv"

if os.path.exists(metrics_file) and os.path.exists(predictions_file):
    metrics_df = pd.read_csv(metrics_file, index_col=0)
    predictions_df = pd.read_csv(predictions_file)

    # Accuracy and Macro F1 metric cards
    accuracy = float(metrics_df.loc["accuracy", "f1-score"])
    macro_f1 = float(metrics_df.loc["macro avg", "f1-score"])

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Macro F1", f"{macro_f1:.4f}")

    # Metrics table
    st.subheader("Classification metrics")
    st.dataframe(metrics_df)

    # Confusion matrix table
    st.subheader("Confusion matrix")
    cm = pd.crosstab(predictions_df["actual"], predictions_df["predicted"])
    st.dataframe(cm)

    # Test predictions table and download
    st.subheader("Test predictions")
    st.dataframe(predictions_df)

    pred_csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download test predictions as CSV",
        data=pred_csv_bytes,
        file_name="test_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Run train.py first to create the results files.")
