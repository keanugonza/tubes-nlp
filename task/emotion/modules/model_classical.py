import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pandas as pd

class ClassicalMLLabeler:
    def __init__(self):
        # Pipeline: TF-IDF Vectorizer -> Linear Support Vector Machine
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('clf', LinearSVC(dual='auto', random_state=42, class_weight='balanced'))
        ])
        self.is_trained = False

    def train(self, train_df: pd.DataFrame):
        """Trains the SVM model."""
        print("Training Classical Model (SVM)...")
        X = train_df['text']
        y = train_df['label']
        self.model.fit(X, y)
        self.is_trained = True
        print("Training Complete.")

    def save_model(self, path: str):
        """
        Saves the trained SVM pipeline to disk.
        
        Args:
            path: Directory path where the model will be saved
        """
        if not self.is_trained:
            raise Exception("Cannot save untrained model!")
        
        print(f"Saving classical model to {path}...")
        if not os.path.exists(path):
            os.makedirs(path)
        
        model_file = os.path.join(path, "classical_model.pkl")
        joblib.dump(self.model, model_file)
        print(f"Model saved successfully to {model_file}")

    def predict(self, texts: list) -> list:
        """Returns integer labels for a list of texts."""
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        return self.model.predict(texts)

    def evaluate(self, test_df: pd.DataFrame, target_names: list):
        """Prints classification report."""
        preds = self.predict(test_df['text'])
        print("\n--- Classical Model Performance ---")
        print(classification_report(test_df['label'], preds, target_names=target_names, zero_division=0))