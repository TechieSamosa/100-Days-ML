import streamlit as st
from Predictor import load_model, predict_sentiment

# Set page title and favicon
st.set_page_config(page_title="Sentiment Analysis App", page_icon=":smiley:")

def main():
    st.title("Sentiment Analysis App")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This is a simple Sentiment Analysis web application built with Streamlit. "
        "It uses a pre-trained model to predict the sentiment of the input text."
    )

    # Load the model
    model_path = "best_model.pkl"
    model = load_model(model_path)

    # Input text for sentiment analysis
    text = st.text_area("Enter text to analyze:", max_chars=300)

    if st.button("Analyze"):
        # Perform sentiment analysis
        if text:
            sentiment = predict_sentiment(model, text)
            result = "Positive sentiment" if sentiment == 1 else "Negative sentiment"
            st.success(result)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
