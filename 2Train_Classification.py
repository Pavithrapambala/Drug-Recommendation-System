import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load your dataset
df = pd.read_csv('D:/Project_2024/Drug_Rec/medicine.csv')

# Combine 'Reason' and 'Description' for text classification
df['text'] = df['Reason'] + ' ' + df['Description']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['Drug_Name'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

# Save the model and vectorizer to .pkl files
with open('classifier_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
