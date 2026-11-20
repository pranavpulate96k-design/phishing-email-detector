import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

def extract_heuristic_features(text: str) -> list:
    has_urgent = 1 if re.search(r'\b(urgent|immediate|verify|account suspended|action required)\b', text, re.I) else 0
    has_link = 1 if re.search(r'http[s]?://', text) else 0
    num_exclamations = text.count('!')
    return [has_urgent, has_link, num_exclamations]

def train_phishing_model():
    dataset = [
        ("Dear customer, your account is suspended. Click http://bit.ly/verify immediately!", 1),
        ("Urgent action required: update your password at http://phish.com now!!", 1),
        ("Verify your banking details to avoid closure http://secure-login.net", 1),
        ("Congratulations! You won a prize. Claim here http://win.money", 1),
        ("Hey, are we still meeting for lunch tomorrow at 12?", 0),
        ("Please review the attached project document and send feedback.", 0),
        ("Weekly team sync meeting notes and updates.", 0),
        ("Hi mom, I will call you later this evening.", 0)
    ]
    texts, labels = zip(*dataset)
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(texts).toarray()
    
    heuristics = np.array([extract_heuristic_features(t) for t in texts])
    X_combined = np.hstack((X_tfidf, heuristics))
    
    model = LogisticRegression()
    model.fit(X_combined, labels)
    
    predictions = model.predict(X_combined)
    
    print("Accuracy:", accuracy_score(labels, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(labels, predictions))
    print("\nClassification Report:\n", classification_report(labels, predictions))
    
    return vectorizer, model

def predict_email(email_text: str, vectorizer, model) -> str:
    tfidf_feat = vectorizer.transform([email_text]).toarray()
    heuristics_feat = np.array([extract_heuristic_features(email_text)])
    combined = np.hstack((tfidf_feat, heuristics_feat))
    pred = model.predict(combined)[0]
    return "Phishing" if pred == 1 else "Legitimate"

if __name__ == "__main__":
    vec, model = train_phishing_model()
    sample = "URGENT: Click http://verify-account.com immediately to secure your bank account!"
    print(f"\nSample Email: '{sample}'")
    print("Prediction:", predict_email(sample, vec, model))
