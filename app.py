# app.py - Streamlit Application for Sentiment Analysis

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import matplotlib.pyplot as plt
import time

# Set page config at the very beginning of the script
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load the saved vectorizer and model
@st.cache_resource
def load_models():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('models/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return vectorizer, model

# Text preprocessing function (same as in training)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# Function to predict sentiment
def predict_sentiment(text, vectorizer, model):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform using the vectorizer
    X_new = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][prediction]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, probability, processed_text

# Streamlit App
def main():
    st.title("Sentiment Analysis App")
    st.write("Enter text to analyze its sentiment")
    
    # Initialize NLTK resources
    download_nltk_resources()
    
    # Load models
    try:
        vectorizer, model = load_models()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please make sure you've run the model training script first and saved the models.")
        return
    
    # Text input
    text_input = st.text_area("Enter text for sentiment analysis:", height=150)
    
    # File upload option
    st.write("OR")
    uploaded_file = st.file_uploader("Upload a CSV or TXT file with text data", type=['csv', 'txt'])
    
    # Batch analysis container
    batch_results = None
    
    # Analyze button for single text
    if st.button("Analyze Sentiment") and text_input:
        with st.spinner("Analyzing..."):
            sentiment, confidence, processed_text = predict_sentiment(text_input, vectorizer, model)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Results")
                
                # Create sentiment meter
                if sentiment == "Positive":
                    st.markdown(f"### Sentiment: {sentiment} 😊")
                    sentiment_color = "green"
                else:
                    st.markdown(f"### Sentiment: {sentiment} 😔")
                    sentiment_color = "red"
                
                st.markdown(f"### Confidence: {confidence:.2%}")
                st.progress(float(confidence))
                
                # Show processed text
                st.subheader("Processed Text")
                st.write(processed_text)
            
            with col2:
                # Visualization
                fig, ax = plt.subplots(figsize=(4, 4))
                labels = ['Positive', 'Negative']
                sizes = [confidence if sentiment == "Positive" else 1-confidence, 
                         confidence if sentiment == "Negative" else 1-confidence]
                colors = ['green', 'red']
                explode = (0.1, 0)
                
                ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct='%1.1f%%', shadow=True, startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # Check if the DataFrame has a column named 'text' or prompt the user to select a column
                if 'text' in df.columns:
                    text_column = 'text'
                else:
                    st.info("Please select the column containing the text to analyze:")
                    text_column = st.selectbox("Text column", df.columns.tolist())
                
                if st.button("Analyze Batch Data"):
                    with st.spinner("Analyzing batch data..."):
                        # Create progress bar
                        progress_bar = st.progress(0)
                        total_rows = len(df)
                        
                        # Add sentiment columns
                        df['sentiment'] = ""
                        df['confidence'] = 0.0
                        
                        # Process each row
                        for i, row in df.iterrows():
                            text = row[text_column]
                            if isinstance(text, str):
                                sentiment, confidence, _ = predict_sentiment(text, vectorizer, model)
                                df.at[i, 'sentiment'] = sentiment
                                df.at[i, 'confidence'] = confidence
                            
                            # Update progress bar
                            progress_bar.progress((i + 1) / total_rows)
                        
                        batch_results = df
                        
            elif uploaded_file.name.endswith('.txt'):
                text_content = uploaded_file.read().decode('utf-8')
                lines = text_content.split('\n')
                
                if st.button("Analyze Batch Data"):
                    with st.spinner("Analyzing batch data..."):
                        # Create progress bar
                        progress_bar = st.progress(0)
                        total_lines = len(lines)
                        
                        # Create a DataFrame to store results
                        results = []
                        
                        # Process each line
                        for i, line in enumerate(lines):
                            if line.strip():
                                sentiment, confidence, _ = predict_sentiment(line, vectorizer, model)
                                results.append({
                                    'text': line,
                                    'sentiment': sentiment,
                                    'confidence': confidence
                                })
                            
                            # Update progress bar
                            progress_bar.progress((i + 1) / total_lines)
                        
                        batch_results = pd.DataFrame(results)
            
            # Display batch results
            if batch_results is not None:
                st.subheader("Batch Analysis Results")
                st.dataframe(batch_results)
                
                # Download button for results
                csv = batch_results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                
                # Show summary statistics
                st.subheader("Summary")
                positive_count = (batch_results['sentiment'] == 'Positive').sum()
                negative_count = (batch_results['sentiment'] == 'Negative').sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positive Sentiments", positive_count)
                    st.metric("Negative Sentiments", negative_count)
                    st.metric("Average Confidence", f"{batch_results['confidence'].mean():.2%}")
                
                with col2:
                    # Create a pie chart of sentiment distribution
                    fig, ax = plt.subplots(figsize=(4, 4))
                    labels = ['Positive', 'Negative']
                    sizes = [positive_count, negative_count]
                    colors = ['green', 'red']
                    explode = (0.1, 0)
                    
                    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                           autopct='%1.1f%%', shadow=True, startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This sentiment analysis model uses a Random Forest classifier trained on TF-IDF features.
        The text preprocessing steps include:
        - Converting to lowercase
        - Removing HTML tags and special characters
        - Removing stopwords
        - Lemmatization
        
        The model classifies text as either Positive or Negative.
        """)

if __name__ == "__main__":
    main()