import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load dataset
df = pd.read_csv('spam (1).csv', encoding='latin-1')
df = df[['v1', 'v2']].copy()
df.columns = ['target', 'text']

# Encode labels: spam=1, ham=0
df['target'] = df['target'].map({'spam': 1, 'ham': 0})

# Remove duplicates
df = df.drop_duplicates(keep='first')

# Apply text preprocessing
print("Preprocessing text...")
df['transformed_text'] = df['text'].apply(transform_text)

# Vectorize
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, precision_score
y_pred = model.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")

# Save new pickle files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nDone! vectorizer.pkl and model.pkl have been re-saved.")
