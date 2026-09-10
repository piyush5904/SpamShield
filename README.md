# SpamShield — SMS Spam Detection

SpamShield is a machine learning project that classifies SMS messages as **spam** or **not spam (ham)**.

The project uses the UCI SMS Spam Collection dataset and a machine learning pipeline built with TF-IDF and Logistic Regression.

## How It Works

```text
SMS Message
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Spam / Ham Prediction
TF-IDF

TF-IDF converts text messages into numerical features that the machine learning model can understand.

The model uses both individual words and two-word combinations (unigrams and bigrams).

Logistic Regression

Logistic Regression is used as the classification algorithm to determine whether a message is spam or ham.

Class weighting is used to handle the imbalance between spam and ham messages.

Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains 5,572 SMS messages labeled as:

ham
spam
Results

The final model achieved approximately:

Accuracy: 97.67%
Spam Precision: 90%
Spam Recall: 93%
Spam F1-score: 91%
Confusion Matrix
Actual	Predicted Ham	Predicted Spam
Ham	951	15
Spam	11	138
Features
TF-IDF text vectorization
Unigram + bigram features
Logistic Regression classifier
Class balancing
Classification report
Confusion matrix
Saved trained model
Separate prediction script
Interactive message classification
Prediction confidence scores
Project Structure
Spam_Detection/
├── spam_detection.py
├── predict.py
├── README.md
└── .gitignore

The dataset and trained model are excluded from the repository using .gitignore.

Installation

Install the required libraries:

pip install pandas scikit-learn matplotlib joblib
Training the Model

Run:

python spam_detection.py

This will:

Load the dataset
Split the data into training and testing sets
Convert messages into TF-IDF features
Train the Logistic Regression model
Evaluate the model
Generate the confusion matrix
Save the trained model as spamshield_model.pkl
Making Predictions

After training, run:

python predict.py

You can then enter your own SMS messages and SpamShield will classify them as spam or not spam.

The model also displays its predicted probability for each class.

Technologies
Python
Pandas
Scikit-learn
Matplotlib
Joblib
What I Learned

This project helped me understand the complete machine learning workflow for a text classification problem — from loading and splitting a real dataset to feature extraction, model training, evaluation, model persistence, and inference on new data.