import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 10px;
}

.sub {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 15px;
    border: 2px solid #38bdf8;
    font-size: 16px;
}

.stButton button {
    width: 100%;
    background: linear-gradient(to right, #06b6d4, #3b82f6);
    color: white;
    border: none;
    padding: 12px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

.stButton button:hover {
    background: linear-gradient(to right, #3b82f6, #06b6d4);
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.spam {
    background-color: #7f1d1d;
    color: #fecaca;
}

.notspam {
    background-color: #14532d;
    color: #bbf7d0;
}

</style>
""", unsafe_allow_html=True)

# -------------------- TEXT PROCESSING --------------------
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

# -------------------- LOAD MODEL --------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -------------------- UI --------------------
st.markdown('<div class="title">📩 Spam Message Detector</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="sub">Check whether your Email or SMS is Spam or Not</div>',
    unsafe_allow_html=True
)

input_sms = st.text_area(
    "Enter Your Message",
    height=180,
    placeholder="Type your SMS or Email here..."
)

# -------------------- PREDICT BUTTON --------------------
if st.button('🚀 Analyze Message'):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # preprocess
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = tfidf.transform([transformed_sms])

        # predict
        result = model.predict(vector_input)[0]

        # display
        if result == 1:
            st.markdown(
                '<div class="result-box spam">🚨 SPAM MESSAGE</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box notspam">✅ NOT SPAM</div>',
                unsafe_allow_html=True
            )

# -------------------- FOOTER --------------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align:center; color:#94a3b8;'>
        Built with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
