import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

ps = PorterStemmer()

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

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

try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except FileNotFoundError:
    st.error("Error: Model files (vectorizer.pkl, model.pkl) not found. Please ensure they are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading models: {e}")
    st.stop()

st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: #333333;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }

    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #007bff;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    h1.css-10trblm {
        color: #007bff;
        text-align: center;
        margin-bottom: 20px;
        font-size: 2.8em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    .result-spam {
        background-color: #ffe0e0;
        color: #d32f2f;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.8em;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #d32f2f;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .result-not-spam {
        background-color: #e0ffe0;
        color: #388e3c;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.8em;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #388e3c;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }

    .example-button button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 8px 15px;
        border: none;
        margin: 5px 0;
        cursor: pointer;
        transition: 0.3s;
        width: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .example-button button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    
    hr {
        border-top: 2px solid #cccccc;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📧 Email/SMS Spam Classifier")

st.write(
    """
    Welcome to the Email/SMS Spam Classifier! This tool helps you quickly determine if a message is spam or not.
    Simply enter any email or SMS text into the box below and click 'Predict'.
    """
)

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

input_sms = st.text_area("Enter the message here:", height=150, key="message_input",
                         placeholder="Type or paste your message...")

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button('Predict')

def clear_message_callback():
    st.session_state.message_input = ""

with col2:
    clear_button = st.button('Clear Message', on_click=clear_message_callback)

st.markdown("---")
st.subheader("💡 Try these examples:")
example_cols = st.columns(2)

example_spam_1 = "Congratulations! You've won a FREE iPhone! Click here to claim your prize."
example_spam_2 = "URGENT! Your bank account has been compromised. Verify your details immediately."
example_ham_1 = "Hi team, please find the updated project report attached. Let me know your thoughts."
example_ham_2 = "Just checking in, how are you doing today? Let's catch up soon."

def load_example_message(message):
    st.session_state.message_input = message

with example_cols[0]:
    st.markdown("<div class='example-button'>", unsafe_allow_html=True)
    st.button("Load Spam Example 1", key="ex_spam_1", on_click=load_example_message, args=(example_spam_1,))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='example-button'>", unsafe_allow_html=True)
    st.button("Load Ham Example 1", key="ex_ham_1", on_click=load_example_message, args=(example_ham_1,))
    st.markdown("</div>", unsafe_allow_html=True)

with example_cols[1]:
    st.markdown("<div class='example-button'>", unsafe_allow_html=True)
    st.button("Load Spam Example 2", key="ex_spam_2", on_click=load_example_message, args=(example_spam_2,))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='example-button'>", unsafe_allow_html=True)
    st.button("Load Ham Example 2", key="ex_ham_2", on_click=load_example_message, args=(example_ham_2,))
    st.markdown("</div>", unsafe_allow_html=True)

if predict_button:
    if input_sms:
        transformed_sms = transform_text(input_sms)
        
        vector_input_sparse = tfidf.transform([transformed_sms])
        
        vector_input_dense = vector_input_sparse.toarray()

        result_label = model.predict(vector_input_dense)[0]
        probabilities = model.predict_proba(vector_input_dense)[0]

        st.markdown("---")
        st.subheader("Prediction Result:")
        
        if result_label == 1:
            st.markdown(f"<div class='result-spam'>🛑 SPAM!</div>", unsafe_allow_html=True)
            st.write(f"Confidence (Spam): **{probabilities[1]*100:.2f}%**")
            st.write(f"Confidence (Not Spam): {probabilities[0]*100:.2f}%")
        else:
            st.markdown(f"<div class='result-not-spam'>✅ NOT SPAM (HAM)</div>", unsafe_allow_html=True)
            st.write(f"Confidence (Not Spam): **{probabilities[0]*100:.2f}%**")
            st.write(f"Confidence (Spam): {probabilities[1]*100:.2f}%")
    else:
        st.warning("Please enter a message to classify.")

st.markdown("---")
st.info("This classifier uses a machine learning model trained on text data. Its accuracy may vary. For educational purposes only.")