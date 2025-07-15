# Riot-Gamer-Churn-Prediction
An end to end pipeline for predicting gamer churn using Riot match data. It retrieves data via the Riot API, extracts player level features, and applies a Bidirectional LSTM model to detect churn supporting retention analytics through temporal modeling and performance evaluation.


This project implements an end-to-end pipeline for **predicting player churn** based on match activity retrieved from the **Riot Games API**. The system is designed to help understand player behavior, identify at-risk users, and support retention strategies in competitive online gaming environments.


##  Description

Churn is defined here as a player who does not return to the game after a long break (e.g., >10 hours between matches). Using match-level metadata, we engineer temporal and behavioral features for each player over their last few games. These features are used to train a **Bidirectional LSTM** model to classify whether a player has churned.

This project covers:
- **Automated data retrieval** via Riot’s public API
- **Custom churn labeling** based on inter-match time gaps
- **Feature extraction** from player match sequences
- **LSTM-based churn prediction** using time-series modeling
- **Evaluation** using precision, recall, F1-score, and bootstrap confidence intervals


##  Execution pipeline

python Riot_API_Data_Retrieving.py    # Optional: Fetch raw match data
python clean.py                       # Clean and structure the match data
python feature_extraction.py          # Generate player feature sequences
python model.py

## Model Details

Architecture: Bidirectional LSTM with batch normalization and dropout
Input: Sequence of match level player features
Output: Binary churn prediction
Evaluation Metrics: Accuracy, Precision, Recall, F1-score



