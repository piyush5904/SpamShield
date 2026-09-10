import joblib

# Load the trained model
model = joblib.load("spamshield_model.pkl")

print("==============================")
print("       SPAMSHIELD")
print("==============================")
print("Enter a message to classify.")
print("Type 'exit' to stop.")

while True:

    message = input("\nEnter your message: ")

    if message.lower() == "exit":
        print("Goodbye!")
        break

    prediction = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]

    print("\nModel confidence:")

    for label, probability in zip(model.classes_, probabilities):
        print(f"{label}: {probability:.2%}")

    if prediction == "spam":
        print("🚨 SPAM")
    else:
        print("✅ NOT SPAM")