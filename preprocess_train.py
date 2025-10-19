import pandas as pd
import numpy as np
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- I. Initial Setup and NLTK Downloads (CORRECTED BLOCK) ---

print("Checking and downloading necessary NLTK packages...")

# Use a list to check and download required packages
required_nltk_packages = ['stopwords', 'wordnet'] 

for package in required_nltk_packages:
    try:
        # Tries to find the package; if not found, it raises a LookupError
        nltk.data.find(f'corpora/{package}')
        print(f"'{package}' found locally.")
    except LookupError:
        # If not found, explicitly download it
        print(f"'{package}' not found. Downloading...")
        nltk.download(package)

lemmatizer = WordNetLemmatizer()


# --- II. Data Loading and EDA ---

# Note: Ensure SMSSpamCollection file is in the same directory
try:
    df = pd.read_csv('SMSSpamCollection', 
                     sep='\t', 
                     header=None, 
                     names=['label', 'message'], 
                     encoding='latin-1')
except FileNotFoundError:
    print("\nError: SMSSpamCollection file not found.")
    print("Please ensure the SMSSpamCollection file is in the same directory.")
    exit()

# 1. EDA: Feature Engineering (Length)
df['length'] = df['message'].apply(len)

# 2. EDA: Class Distribution
print("\n--- EDA Summary ---")
print("Total Messages:", len(df))
print("Class Distribution (Counts):\n", df['label'].value_counts())
print("Average Message Length (Ham vs. Spam):\n", df.groupby('label')['length'].mean().round(2))
print("---------------------\n")


# --- III. Text Preprocessing Function ---

def preprocess_text(text):
    """Cleans, tokenizes, removes stopwords, and lemmatizes a message."""
    # 1. Lowercase and remove punctuation
    text = text.lower()
    # Use str.translate to efficiently remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 2. Tokenization and Stopword Removal
    tokens = text.split()
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop_words]
        
    # 3. Lemmatization
    # Only lemmatize tokens that are alphabetic (to skip numbers/single characters)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    
    return " ".join(tokens)

# Apply preprocessing
df['cleaned_message'] = df['message'].apply(preprocess_text)


# --- IV. Feature Extraction (TF-IDF) and Data Splitting ---
print("Applying TF-IDF Feature Extraction and splitting data...")
X = df['cleaned_message']
y = df['label']

# Split 80/20 with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize, fit on training data, and transform
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# --- V. Model Training and Evaluation ---
print("Training and evaluating models...")

def evaluate_model(y_true, y_pred, model_name):
    """Calculates and prints key classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    # Target the minority class 'spam' for Precision/Recall/F1
    prec = precision_score(y_true, y_pred, pos_label='spam', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label='spam', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='spam', zero_division=0)
    
    print(f"--- Results for {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 30)
    # Return F1 score for model selection
    return f1

# 1. Train Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)
nb_f1 = evaluate_model(y_test, nb_predictions, "Multinomial Naive Bayes")

# 2. Train Logistic Regression
lr_model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)
lr_f1 = evaluate_model(y_test, lr_predictions, "Logistic Regression")

# Select best model (MNB is preferred due to high precision/F1)
best_model = nb_model


# --- VI. Artifact Saving ---
print("\nSaving best model (Multinomial Naive Bayes) and TF-IDF Vectorizer...")

# 1. Save the best model
with open('spam_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
    
# 2. Save the fitted TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Trained model and vectorizer saved successfully as .pkl files.")