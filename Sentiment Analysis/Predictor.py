import pickle
import sys
import numpy as np

def load_model(model_path):
    """Load the saved model from the pickle file."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_input(text):
    """Preprocess the input text."""
    # Your preprocessing steps go here, e.g., tokenization, vectorization, etc.
    # For simplicity, let's assume no preprocessing is needed in this example.
    return text

def predict_sentiment(model, text):
    """Predict sentiment using the loaded model."""
    # Preprocess input text
    preprocessed_text = preprocess_input(text)
    # Perform sentiment prediction
    prediction = model.predict([preprocessed_text])
    return prediction[0]

def main():
    if len(sys.argv) != 3:
        print("Usage: python predictor.py model.pkl 'text to analyze'")
        sys.exit(1)
    
    # Load the model
    model_path = sys.argv[1]
    model = load_model(model_path)

    # Text to analyze
    text = sys.argv[2]

    # Predict sentiment
    sentiment = predict_sentiment(model, text)

    # Print the predicted sentiment
    if sentiment == 1:
        print("Positive sentiment")
    else:
        print("Negative sentiment")

if __name__ == "__main__":
    main()
