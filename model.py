# here's wehere i put the code to train a digit recognition model

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump((model, digits.target_names), "digit_model.pkl")
    print("✅ Model trained and saved as 'digit_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()
