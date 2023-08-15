# Amazon Fine Food Sentiment Analysis with BERT

This repository contains code for a sentiment analysis demo that predicts the sentiment of Amazon fine food reviews using a finetuned BERT Base model from the Hugging Face Transformers library. The demo also includes an interface built using Gradio, allowing users to interactively input reviews and receive sentiment predictions.

## About the Demo

The sentiment analysis model is trained on the Amazon Fine Food Reviews dataset, which includes:

- Number of reviews: 568,454
- Number of users: 256,059
- Number of products: 74,258
- Timespan: Oct 1999 — Oct 2012
- Number of Attributes/Columns in data: 10

## Model Architecture

**Model Architecture:**

- `self.bert`: BERT Base model loaded from pre-trained weights.
- `self.drop`: Dropout layer applied for regularization.
- `self.out`: Linear layer mapping BERT hidden size to sentiment classes.

**Files in the Repository:**

- `amazon_finefood_sentiment_analysis_training.ipynb`: Code for training the sentiment analysis model.
- `amazon_finefood_sentiment_analysis_interface.ipynb`: Code for building the Gradio interface.
- `sentiment_analysis_finetune_bert.pkl`: Trained sentiment analysis model in pkl format.
- `rnn_lstm_gru.py`:code for Training sentiment analysis model using LSTM GRU RNN.
- result.ipynb`:code of result of comparison between LSTM, GRU, RNN and finetuned BERT Base.

**Usage:**

To run the code and interact with the sentiment analysis demo:

1. Open `amazon_finefood_sentiment_analysis_interface.ipynb`.
2. Set the file path to `sentiment_analysis_finetune_bert.pkl`.
3. Execute the notebook cells to set up the Gradio interface and make predictions.

Feel free to experiment with the interface, input different reviews, and observe sentiment predictions and confidence scores.

For questions/issues, open an issue in this repository.

**Model Achievements**

- Gated Recurrent Unit (GRU): Achieved an accuracy of 94.8%.
- Long Short-Term Memory (LSTM): Implemented an architecture with an accuracy of 93.2%.
- BERT Base Model Fine-Tuning: Achieved an accuracy of 96.4% after finetuning.

**Training Details**

All experiments were performed on a single NVIDIA RTX 2070 GPU. The training times are as follows:

- GRU Model: Trained for 10 epochs, took approximately 10+ hours.
- LSTM Model: Trained for 10 epochs, took around 10+ hours.
- BERT Base Model Fine-Tuning: Trained for 10 epochs, took around 15+ hours.

**Acknowledgments:**

The sentiment analysis model uses BERT architecture from Hugging Face Transformers. The Amazon Fine Food Reviews dataset is for training. Gradio is used for the interactive interface.

