import pandas as pd
import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "faqs.csv"
THRESHOLD = 0.30

FALLBACK = (
    "Sorry, I couldn't find a relevant answer. "
    "Please try asking in a different way."
)

EXAMPLES = [
    "What are the college timings?",
    "How can I apply for admission?",
    "Is attendance compulsory?",
    "Does the college have a library?",
    "Are placement opportunities available?",
    "Do you have Wi-Fi?"
]


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="SmartFAQ",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# NLP SETUP
# ============================================================

try:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

except Exception:
    STOP_WORDS = set(ENGLISH_STOP_WORDS)


tokenizer = RegexpTokenizer(r"[a-z0-9]+")
stemmer = PorterStemmer()


def preprocess(text):

    text = str(text).lower()

    text = text.replace("wi-fi", "wifi")
    text = text.replace("wi fi", "wifi")

    tokens = tokenizer.tokenize(text)

    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(tokens)


# ============================================================
# LOAD FAQ DATA
# ============================================================

@st.cache_resource
def load_model():

    df = pd.read_csv(CSV_FILE)

    required_columns = {
        "question",
        "answer",
        "category"
    }

    if not required_columns.issubset(df.columns):

        raise ValueError(
            "faqs.csv must contain: "
            "question, answer, category"
        )

    df = df.dropna().reset_index(drop=True)

    cleaned_questions = df["question"].apply(
        preprocess
    )

    vectorizer = TfidfVectorizer()

    faq_vectors = vectorizer.fit_transform(
        cleaned_questions
    )

    return df, vectorizer, faq_vectors


# ============================================================
# FIND BEST ANSWER
# ============================================================

def get_answer(
    user_question,
    df,
    vectorizer,
    faq_vectors
):

    if not user_question or not user_question.strip():

        return {
            "found": False,
            "answer": "Please type a question first.",
            "score": 0
        }

    cleaned_question = preprocess(
        user_question
    )

    if len(cleaned_question) < 3:

        return {
            "found": False,
            "answer": (
                "Your question is too short. "
                "Please ask a complete question."
            ),
            "score": 0
        }

    user_vector = vectorizer.transform(
        [cleaned_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_index = similarity_scores.argmax()

    best_score = float(
        similarity_scores[best_index]
    )

    if best_score < THRESHOLD:

        return {
            "found": False,
            "answer": FALLBACK,
            "score": best_score
        }

    best_row = df.iloc[best_index]

    return {
        "found": True,
        "answer": best_row["answer"],
        "question": best_row["question"],
        "category": best_row["category"],
        "score": best_score
    }


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


if "dark_mode" not in st.session_state:

    st.session_state.dark_mode = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 SmartFAQ")

    st.caption(
        "College information, answered instantly."
    )

    st.divider()

    st.subheader("Appearance")

    st.session_state.dark_mode = st.toggle(
        "🌙 Dark mode",
        value=st.session_state.dark_mode
    )

    st.divider()

    st.subheader("Project Overview")

    st.metric(
        "FAQ Questions",
        "36"
    )

    st.metric(
        "Categories",
        "16"
    )

    st.metric(
        "NLP Method",
        "TF-IDF"
    )

    st.metric(
        "Matching",
        "Cosine Similarity"
    )

    st.divider()

    st.subheader("Try a Question")

    selected_question = None

    for example in EXAMPLES:

        if st.button(
            example,
            use_container_width=True
        ):

            selected_question = example


# ============================================================
# LOAD MODEL
# ============================================================

try:

    df, vectorizer, faq_vectors = load_model()

except FileNotFoundError:

    st.error(
        "faqs.csv was not found."
    )

    st.info(
        "Keep faqs.csv in the same folder as app.py."
    )

    st.stop()

except Exception as error:

    st.error(
        f"Could not load FAQ data: {error}"
    )

    st.stop()


# ============================================================
# MAIN TITLE
# ============================================================

if st.session_state.dark_mode:

    st.title("🌙 SmartFAQ")

    st.caption(
        "NLP Based FAQ Retrieval Chatbot • Dark Mode"
    )

else:

    st.title("🎓 SmartFAQ")

    st.caption(
        "NLP Based FAQ Retrieval Chatbot"
    )


st.divider()


# ============================================================
# INTRODUCTION
# ============================================================

st.header(
    "Ask. Match. Get the Answer."
)

st.write(
    "SmartFAQ understands your question and finds "
    "the most relevant answer from the college FAQ "
    "knowledge base using NLP similarity matching."
)


# ============================================================
# STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "FAQ Knowledge",
        f"{len(df)} questions"
    )


with col2:

    st.metric(
        "Categories",
        f"{df['category'].nunique()} topics"
    )


with col3:

    st.metric(
        "Technique",
        "TF-IDF"
    )


with col4:

    st.metric(
        "Matching",
        "Cosine Similarity"
    )


st.divider()


# ============================================================
# ASK QUESTION
# ============================================================

st.header("Ask SmartFAQ")

st.caption(
    "Type your question naturally. "
    "You do not need to use the exact wording "
    "from the FAQ."
)


with st.form(
    "question_form",
    clear_on_submit=True
):

    user_question = st.text_input(
        "Your question",
        placeholder=(
            "Example: Does the college provide Wi-Fi?"
        )
    )

    submit = st.form_submit_button(
        "🔎 Find Answer",
        use_container_width=True
    )


# ============================================================
# PROCESS QUESTION
# ============================================================

question = None


if selected_question:

    question = selected_question

elif submit:

    question = user_question


if question:

    result = get_answer(
        question,
        df,
        vectorizer,
        faq_vectors
    )

    st.session_state.history.append(
        (question, result)
    )


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander(
    "⚙️ How SmartFAQ Works"
):

    st.write(
        "The chatbot follows a simple NLP pipeline:"
    )

    st.write(
        "1. User enters a question"
    )

    st.write(
        "2. Text is cleaned and preprocessed"
    )

    st.write(
        "3. TF-IDF converts text into numerical vectors"
    )

    st.write(
        "4. Cosine similarity compares the questions"
    )

    st.write(
        "5. The most similar FAQ is selected"
    )

    st.write(
        "6. The corresponding answer is displayed"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if not st.session_state.history:

    st.info(
        "💬 Ask a question above or select an "
        "example question from the sidebar."
    )


for question, result in reversed(
    st.session_state.history
):

    # User question

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.write(question)


    # Bot answer

    with st.chat_message(
        "assistant",
        avatar="🎓"
    ):

        st.write(
            result["answer"]
        )


        if result["found"]:

            col1, col2 = st.columns(2)

            with col1:

                st.caption("Category")

                st.write(
                    result["category"]
                )

            with col2:

                st.caption("Similarity Score")

                st.write(
                    f"{result['score'] * 100:.1f}%"
                )

            st.caption(
                "Matched FAQ"
            )

            st.write(
                result["question"]
            )

        else:

            st.warning(
                "No confident FAQ match was found."
            )


# ============================================================
# FOOTER INFORMATION
# ============================================================

st.divider()

st.caption(
    "SmartFAQ • NLP FAQ Retrieval System • "
    "TF-IDF + Cosine Similarity"
)