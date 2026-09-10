import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. LOAD THE DATASET
# -----------------------------------

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

print("Dataset size:", len(df))


# -----------------------------------
# 2. SEPARATE INPUT AND TARGET
# -----------------------------------

X = df["message"]
y = df["label"]


# -----------------------------------
# 3. SPLIT INTO TRAINING AND TESTING
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 4. CREATE THE ML PIPELINE
# -----------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# -----------------------------------
# 5. TRAIN THE MODEL
# -----------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete!")

joblib.dump(model, "spamshield_model.pkl")

print("Model saved!")

# -----------------------------------
# 6. TEST THE MODEL
# -----------------------------------

predictions = model.predict(X_test)


# -----------------------------------
# 7. EVALUATE THE MODEL
# -----------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# -----------------------------------
# 8. CONFUSION MATRIX
# -----------------------------------

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["ham", "spam"]
)

disp.plot()
plt.title("SpamShield - Confusion Matrix")
plt.show()

# -----------------------------------
# 9. USER INPUT
# -----------------------------------

print("\n==============================")
print("       SPAMSHIELD")
print("==============================")
print("Enter a message to classify.")
print("Type 'exit' to stop.")


while True:

    message = input("\nEnter your message: ")

    if message.lower() == "exit":
        print("Goodbye!")
        break

    # Make prediction
    prediction = model.predict([message])[0]

    # Get confidence probabilities
    probabilities = model.predict_proba([message])[0]

    print("\nModel confidence:")

    for label, probability in zip(model.classes_, probabilities):
        print(f"{label}: {probability:.2%}")

    # Display result
    if prediction == "spam":
        print("🚨 SPAM")
    else:
        print("✅ NOT SPAM")