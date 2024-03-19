# Sentiment Analysis Model using IMDB Data

This repository contains code for a sentiment analysis model trained on the IMDB movie review dataset. The model predicts the sentiment (positive or negative) of movie reviews.

## Usage

### 1. Data Preprocessing

The `preprocessing.ipynb` Jupyter Notebook provides the code for preprocessing the IMDB dataset. It includes steps such as tokenization, padding, and data splitting. Execute the notebook to preprocess the data before training the model.

### 2. Model Training

The `model_training.ipynb` Jupyter Notebook contains code for training the sentiment analysis model using the preprocessed IMDB dataset. The notebook utilizes a deep learning architecture (e.g., LSTM, CNN) for sentiment classification.

### 3. Model Evaluation and Selection

After training, evaluate the performance of the trained models using appropriate metrics (e.g., accuracy, F1-score) on a validation set. Select the model with the best validation score for further use.

### 4. Model Deployment

The `predictor.py` Python script is responsible for loading the selected model and providing predictions for new movie reviews. Once the model is loaded, it can be called to predict the sentiment of input text data.

### 5. Saving the Model

After selecting the best-performing model, save it as a pickle file using the `pickle` library in Python. The saved model file can then be loaded for deployment without needing to retrain the model.

### 6. Streamlit App

Use the `streamlit_app.py` script to create a web application for the sentiment analysis model. This app allows users to input movie reviews and get real-time predictions of their sentiment. The saved model pickle file is imported into the Streamlit app for inference.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributors

- [Aditya Khamitkar](https://github.com/TheNaiveSamosa)
- [Tushar Jagtap](https://github.com/Tushar-Jagatap)
- [Vedant Deshmukh](https://github.com/vedant4687)

## Issues

If you encounter any issues or have suggestions for improvement, please feel free to open an issue on the GitHub repository. We welcome contributions from the community.

## Acknowledgments

- This project utilizes the IMDB movie review dataset, which is publicly available.
- We thank the open-source community for their contributions to libraries and frameworks used in this project.

## Disclaimer

This sentiment analysis model is trained on the IMDB movie review dataset and may not generalize well to other domains or types of text data. Use it at your discretion.