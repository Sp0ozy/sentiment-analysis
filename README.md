# IMDB Sentiment Analysis

A deep learning project that classifies movie reviews as **positive** or **negative**. Built as part of my data science learning journey.

## What it does

Users can type any movie review into a simple web app, and the model predicts whether the sentiment is positive or negative. The app is built with Streamlit and runs a pre-trained RNN model under the hood.

## ML Approach

The model is a **Simple RNN (Recurrent Neural Network)** — a good starting point for learning how sequential models process text.

**Architecture:**
- `Embedding` layer — converts words into 128-dimensional vectors (vocabulary of 10,000 words)
- `SimpleRNN` layer — 128 hidden units with ReLU activation
- `Dense` output — single neuron with Sigmoid for binary classification

**Training setup:**
- Loss: Binary Crossentropy
- Optimizer: Adam
- Input sequences padded to 500 tokens
- Early stopping to prevent overfitting and save time

**Dataset:** Keras built-in IMDB dataset (~25,000 training reviews). The training set is relatively small for NLP, so the model reaches around **77–78% accuracy** - decent, but not great. A larger dataset or a more advanced architecture, like LSTM or a Transformer, would improve this significantly.

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Building and training the RNN model |
| NumPy / Pandas | Data handling |
| Scikit-learn / SciKeras | ML utilities and Keras wrapper |
| Matplotlib | Plotting training metrics |
| TensorBoard | Visualizing training runs |
| Streamlit | Web app interface |

## Project Structure

```
├── embedding.ipynb     # Exploring word embeddings and one-hot encoding
├── simplernn.ipynb     # Model training
├── prediction.ipynb    # Running inference on new examples
├── app.py              # Streamlit web app
└── simple_rnn_imdb.keras  # Saved model weights
```

## Skills Learned

- How to preprocess text for deep learning (tokenization, padding, vocabulary indexing)
- Building and training RNN models with Keras
- Understanding word embeddings
- Deploying a trained model in a simple web app with Streamlit
