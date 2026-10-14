"""
SMARTFAQ - NLP-Powered College Information Assistant (Streamlit user interface).

All NLP / retrieval logic lives in engine.py. This file only draws the interface.
Raw HTML is never used for content: only one <style> block is injected for theming.
"""

import logging
import re

import streamlit as st

from engine import (MIN_KNOWN_FRACTION, OUTCOME_LABELS, THRESHOLD, DataError, FAQEngine,
                    evaluate, preprocess_steps)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("smartfaq")

APP_NAME = "SMARTFAQ"
SUBTITLE = "NLP-Powered College Information Assistant"
TAGLINE_1, TAGLINE_2 = "Your College.", "One Answer Away."
PAGES = ["Chat", "FAQ Categories", "How It Works", "Evaluation", "About"]
MAX_HISTORY = 20
SUGGESTIONS = [
    "What documents are required for admission?",
    "What are the college timings?",
    "Is attendance compulsory?",
    "Does the college provide Wi-Fi?",
    "Are placement opportunities available?",
    "What scholarships are available?",
]

# ------------------------------------------------------------------ theme
# Colours meet WCAG AA contrast (4.5:1) for text. Button fill is a little darker than the
# dark-mode accent so that white button text stays readable.
LIGHT = {"bg": "#F8FAFC", "card": "#FFFFFF", "border": "#CBD5E1", "text": "#0F172A",
         "muted": "#475569", "input": "#FFFFFF", "accent": "#1D4ED8", "button": "#1D4ED8",
         "tint": "#EFF6FF"}
DARK = {"bg": "#080D18", "card": "#111827", "border": "#334155", "text": "#F8FAFC",
        "muted": "#A8B5C7", "input": "#0F172A", "accent": "#60A5FA", "button": "#2563EB",
        "tint": "#16233B"}

CSS = """
html, body, .stApp, [data-testid="stAppViewContainer"] {background: var(--bg); color: var(--text);
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {background: var(--card);}
[data-testid="stSidebar"] {border-right: 1px solid var(--border);}
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 800px; padding-top: 2.2rem; padding-bottom: 3rem;}

/* text colours: every kind of text follows the theme */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label, .stApp span,
[data-testid="stMarkdownContainer"] *, [data-testid="stText"], [data-testid="stText"] * {color: var(--text);}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color: var(--muted) !important;}
.stApp h1 {font-family: Georgia, "Times New Roman", serif; font-weight: 700;
  font-size: 2.7rem; line-height: 1.1; letter-spacing: -0.01em; padding: 0.1rem 0;}
.stApp h2, .stApp h3 {font-family: Georgia, "Times New Roman", serif; font-weight: 700;}
[data-testid="stText"] {font-family: ui-monospace, Consolas, monospace; font-size: 0.9rem;}
hr {border-color: var(--border);}
[data-testid="stVerticalBlockBorderWrapper"] {border-color: var(--border) !important;
  background: var(--card); border-radius: 10px;}

/* alerts and expanders keep readable text in both themes */
[data-testid="stAlert"] {background: var(--card); border: 1px solid var(--border); border-radius: 8px;}
[data-testid="stAlert"] * {color: var(--text) !important;}
[data-testid="stExpander"] {border: 1px solid var(--border); border-radius: 8px; background: var(--card);}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color: var(--text) !important;}
[data-testid="stExpander"] svg {fill: var(--text); color: var(--text);}
[data-testid="stMarkdownContainer"] table {border-collapse: collapse; width: 100%; font-size: 0.9rem;}
[data-testid="stMarkdownContainer"] th {background: var(--tint); border: 1px solid var(--border);}
[data-testid="stMarkdownContainer"] td {border: 1px solid var(--border);}

/* sidebar open/close arrows must be visible on both backgrounds */
[data-testid="stSidebarCollapseButton"] *, [data-testid="stExpandSidebarButton"] *,
[data-testid="collapsedControl"] * {color: var(--text) !important; fill: var(--text) !important;}

/* text input: readable in both themes */
.stTextInput input {background: var(--input); color: var(--text) !important;
  -webkit-text-fill-color: var(--text); caret-color: var(--text); font-size: 1.05rem; padding: 0.85rem 1rem;}
.stTextInput input::placeholder {color: var(--muted) !important; -webkit-text-fill-color: var(--muted); opacity: 1;}
.stTextInput div[data-baseweb="input"], .stTextInput div[data-baseweb="base-input"] {
  background: var(--input); border-radius: 8px;}
.stTextInput div[data-baseweb="input"] {border: 1px solid var(--border);}
.stTextInput div[data-baseweb="input"]:focus-within {border-color: var(--accent); box-shadow: 0 0 0 3px var(--tint);}
[data-testid="stForm"] {border: none; padding: 0; background: transparent;}

/* buttons */
.stButton > button {background: var(--card); color: var(--text); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.6rem 0.9rem; font-weight: 500; text-align: left;
  justify-content: flex-start; transition: border-color 0.15s, color 0.15s;}
.stButton > button p {color: inherit; text-align: left;}
.stButton > button:hover {border-color: var(--accent); color: var(--accent);}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--button); border-color: var(--button); color: #FFFFFF;}
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p {color: #FFFFFF;}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  color: #FFFFFF; filter: brightness(1.1);}
[data-testid="stFormSubmitButton"] button {background: var(--button); color: #FFFFFF;
  border: 1px solid var(--button); border-radius: 8px; min-height: 3rem; font-weight: 600;}
[data-testid="stFormSubmitButton"] button p {color: #FFFFFF;}
[data-testid="stFormSubmitButton"] button:hover {filter: brightness(1.1); color: #FFFFFF;}
button:focus-visible, input:focus-visible, [role="radio"]:focus-visible {
  outline: 3px solid var(--accent) !important; outline-offset: 2px;}

/* chat messages */
[data-testid="stChatMessage"] {background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 0.75rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {background: var(--tint);}
.stProgress > div > div {background: var(--border);}
.stProgress > div > div > div {background: var(--accent);}

@media (max-width: 640px) {
  .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem;}
  .stApp h1 {font-size: 2.1rem;}
}
"""


def apply_theme(dark):
    """Inject the colour variables and CSS (styling only, no content)."""
    palette = DARK if dark else LIGHT
    variables = ":root{" + ";".join(f"--{k}:{v}" for k, v in palette.items()) + "}"
    st.markdown(f"<style>{variables}{CSS}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def plain(text):
    """Escape Markdown characters so user text is always shown literally (no links, images, styling)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$<&])", r"\\\1", str(text))


@st.cache_resource(show_spinner=False)
def load_engine():
    """Build the TF-IDF index once per server process (not on every click)."""
    return FAQEngine()


@st.cache_resource(show_spinner=False)
def run_evaluation():
    return evaluate(load_engine())


# ---------------------------------------------------------------- callbacks
def ask(question):
    """Answer a question and add it to the conversation (runs before the page redraws)."""
    result = load_engine().ask(question)
    st.session_state.notice = ""
    if result.status == "empty":
        st.session_state.notice = result.message
    else:
        st.session_state.history = (st.session_state.history + [(question, result)])[-MAX_HISTORY:]
    st.session_state.page = "Chat"


def submit_form():
    ask(st.session_state.q_input)


def clear_chat():
    st.session_state.history = []
    st.session_state.notice = ""


def set_category(name):
    st.session_state.cat = name


# ----------------------------------------------------------- shared widgets
def question_form():
    with st.form("ask_form", clear_on_submit=True):
        col_input, col_button = st.columns([5, 1])
        col_input.text_input("Your question", key="q_input", label_visibility="collapsed",
                             max_chars=300, placeholder="Ask SMARTFAQ about your college...")
        col_button.form_submit_button("Ask", on_click=submit_form, use_container_width=True)
    if st.session_state.notice:
        st.warning(st.session_state.notice)


def render_details(result):
    """'Why did I get this answer?' - the retrieval evidence."""
    with st.expander("Why did I get this answer?"):
        st.markdown(f"**Your question after preprocessing:** {plain(result.processed_query) or '(nothing left)'}")
        st.markdown(f"**Keywords found in the FAQ vocabulary:** {result.known_fraction:.0%}")
        st.markdown(f"**Similarity threshold:** {THRESHOLD:.0%} (below this, SMARTFAQ refuses to answer)")
        st.markdown("**Closest FAQs:**")
        st.markdown("\n".join(
            f"{i}. “{plain(c.question)}” · {plain(c.category)} · **{c.score:.1%}**"
            for i, c in enumerate(result.candidates, 1)))


def render_message(question, result):
    with st.chat_message("user", avatar="👤"):
        st.markdown(plain(question))
    with st.chat_message("assistant", avatar="🎓"):
        if result.status == "answer":
            st.caption("ANSWER")
            st.markdown(plain(result.answer))
            st.divider()
            c1, c2, c3 = st.columns([3, 1.5, 1.2])
            c1.caption("MATCHED FAQ")
            c1.markdown(f"“{plain(result.question)}”")
            c2.caption("CATEGORY")
            c2.markdown(plain(result.category))
            c3.caption("SIMILARITY")
            c3.markdown(f"**{result.score:.1%}**")
            st.progress(min(result.score, 1.0))
            if result.matched_phrase != result.question:
                st.caption(f"Matched stored wording: “{result.matched_phrase}”")
            if result.note:
                st.caption(result.note)
            render_details(result)
        else:
            st.markdown(f"**{result.message}**")
            if result.status == "no_match":
                st.write("SMARTFAQ only answers from its FAQ knowledge base. Try rewording your question, "
                         "or browse the FAQ Categories page.")
                if result.note:
                    st.caption(result.note)
                if result.score > 0:
                    st.caption(f"Closest FAQ: “{result.question}” at {result.score:.1%} similarity, "
                               f"below the {THRESHOLD:.0%} threshold, so it was not used.")
                render_details(result)


# -------------------------------------------------------------------- pages
def page_chat(engine):
    history = st.session_state.history
    if not history:
        st.caption(f"{APP_NAME} · {SUBTITLE.upper()}")
        st.markdown(f"# {TAGLINE_1}\n# {TAGLINE_2}")
        st.write("Ask about admissions, academics, attendance, placements, facilities and campus life. "
                 "SMARTFAQ finds the most similar question in the college FAQ knowledge base.")
        question_form()
        st.markdown("#### How can SMARTFAQ help?")
        st.caption("SUGGESTED QUESTIONS · TAP ONE TO TRY IT")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTIONS):
            cols[i % 2].button(q, key=f"sug_{i}", on_click=ask, args=(q,), use_container_width=True)
        st.caption(f"{engine.faq_count} FAQs · {len(engine.category_counts)} categories · "
                   f"TF-IDF + cosine similarity · demo data")
    else:
        st.markdown("### Ask SMARTFAQ")
        question_form()
        for question, result in reversed(history):     # newest first, so the answer is next to the input
            render_message(question, result)


def page_categories(engine):
    st.markdown("## FAQ Categories")
    st.caption("Browse the knowledge base by topic. Select a question to ask it in Chat.")
    counts = engine.category_counts
    cols = st.columns(3)
    for i, name in enumerate(counts.index):
        kind = "primary" if st.session_state.get("cat") == name else "secondary"
        cols[i % 3].button(f"{name} · {counts[name]}", key=f"cat_{i}", type=kind,
                           on_click=set_category, args=(name,), use_container_width=True)
    selected = st.session_state.get("cat")
    if selected in counts.index:
        st.divider()
        st.caption(f"{selected.upper()} · {counts[selected]} FAQ(S)")
        for i, q in enumerate(engine.questions_in(selected)):
            st.button(q, key=f"catq_{i}", on_click=ask, args=(q,), use_container_width=True)
    else:
        st.caption("Select a category above to see its questions.")


def page_how(engine):
    st.markdown("## How SMARTFAQ finds your answer")
    st.caption("A classic information-retrieval pipeline. It finds the closest stored FAQ; it does not write new text.")
    steps = [
        ("01", "QUESTION", "The student types a question in natural language."),
        ("02", "PREPROCESS", "Lowercase, normalise spellings (Wi-Fi = WiFi), split into words, remove "
                             "stopwords such as 'the' and 'is', and stem words ('timings' becomes 'time')."),
        ("03", "TF-IDF", "Each word gets a weight: high if it is frequent in this question but rare across "
                         "the FAQs, low if it appears everywhere. A question becomes a vector of weights."),
        ("04", "COSINE SIMILARITY", "The angle between the question vector and every stored FAQ wording is "
                                    "measured. 1.0 means identical word weights, 0.0 means no shared words."),
        ("05", "BEST MATCH", f"The highest-scoring FAQ wins, but only if its similarity is at least "
                             f"{THRESHOLD:.0%} and most of the question's keywords are known."),
        ("06", "ANSWER", "The stored answer is shown with the matched FAQ, its category and the similarity "
                         "score. Otherwise a fallback message is shown."),
    ]
    for number, title, text in steps:
        with st.container(border=True):
            st.markdown(f"**{number} · {title}**")
            st.caption(text)

    st.markdown("### Try the pipeline yourself")
    sample = st.text_input("Type any question to see each stage", value="When does college start?",
                           key="inspect", max_chars=300)
    stages = preprocess_steps(sample)
    shown = engine.ask(sample)
    st.text("\n".join([
        f"1. Lowercase + normalise : {stages['normalised']}",
        f"2. Tokens                : {stages['tokens']}",
        f"3. Without stopwords     : {stages['without_stopwords']}",
        f"4. After stemming        : {stages['stemmed']}",
        f"5. Best FAQ              : {shown.candidates[0].question if shown.candidates else '(none)'}",
        f"6. Similarity            : {shown.candidates[0].score:.1%}" if shown.candidates else "6. Similarity            : -",
        f"7. Decision              : {'ANSWER' if shown.status == 'answer' else 'FALLBACK'}",
    ]))

    st.markdown("### Design decisions worth knowing")
    st.markdown(
        f"- **Threshold ({THRESHOLD:.2f}):** a nearest FAQ always exists, even for 'What is the capital of "
        "Japan?'. The threshold turns 'nearest' into 'relevant enough'. See the Evaluation page for how it behaves.\n"
        f"- **Known-keyword rule:** TF-IDF ignores words it has never seen, so a question can shrink to one "
        f"matching word. Unless more than {MIN_KNOWN_FRACTION:.0%} of a question's keywords are known, SMARTFAQ refuses.\n"
        "- **Alternative wordings:** each FAQ is stored with several ways of asking it. The best wording "
        "counts, so 'When does college start?' reaches the timings FAQ.\n"
        "- **Ties:** equal scores are resolved by file order, so results are repeatable. A near-tie is flagged."
    )


def outcome_table(rows):
    lines = ["| Query | Expected FAQ | Retrieved FAQ | Similarity | Result |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {plain(r['query'])} | {plain(r['expected'] or '(none: should refuse)')} | "
                     f"{plain(r['retrieved'] or '-')} | {r['score']:.1%} | {OUTCOME_LABELS[r['outcome']]} |")
    return "\n".join(lines)


def page_evaluation(engine):
    st.markdown("## Evaluation")
    st.caption("Measured live on a built-in test set. Nothing on this page is typed in by hand.")
    report = run_evaluation()
    overall = report["overall"]
    st.markdown(f"### Overall: {overall['good']} of {overall['total']} correct ({overall['accuracy']:.1%})")
    names = {"demo": "Demo questions", "paraphrase": "Reworded questions (not stored in the FAQ file)",
             "out_of_scope": "Unrelated questions (should be refused)"}
    for key, label in names.items():
        g = report["groups"][key]
        st.markdown(f"- **{label}:** {g['good']} / {g['total']} = {g['accuracy']:.1%}")
    st.caption("Accuracy = (correct answers + correct refusals) / all test queries. The reworded and unrelated "
               "groups are the meaningful ones: demo questions are close to stored wordings, so they are easy.")

    st.markdown("### Threshold study")
    lines = ["| Threshold | Accuracy | Missed | Wrong FAQ | False accepts |", "|---|---|---|---|---|"]
    for s in report["sweep"]:
        mark = " (used)" if abs(s["threshold"] - report["threshold"]) < 1e-9 else ""
        lines.append(f"| {s['threshold']:.2f}{mark} | {s['accuracy']:.1%} | {s['missed']} | "
                     f"{s['wrong_answer']} | {s['false_accept']} |")
    st.markdown("\n".join(lines))
    st.caption("Missed = a valid question was refused. False accept = an unrelated question was answered. "
               "Very high thresholds refuse valid questions; very low ones risk answering unrelated ones.")

    with st.expander("Show every test query"):
        st.markdown(outcome_table(report["rows"]))
    st.markdown("**Honest notes**")
    st.markdown(
        f"- The test set has only {overall['total']} queries, written by the developer. It is indicative, not a benchmark.\n"
        "- Most errors come from words with no overlap, for example 'shift' vs 'change' or 'scooter' vs 'bike'. "
        "TF-IDF matches words, not meaning.\n"
        "- Neighbouring FAQs that share words (scholarship and admission forms) are the other error source."
    )


def page_about(engine):
    st.markdown("## About SMARTFAQ")
    st.write("SMARTFAQ is an NLP-based FAQ retrieval system for a college. A student asks a question in "
             "their own words and SMARTFAQ returns the answer of the most similar stored question.")
    st.markdown("### Why this project?")
    cols = st.columns(2)
    for i, (title, text) in enumerate([
        ("FAST", "Answers appear instantly. The TF-IDF index is built once and reused."),
        ("NATURAL", "Ask in different words: each FAQ is stored with several alternative wordings."),
        ("TRANSPARENT", "Every answer shows the matched FAQ, its category and the similarity score."),
        ("LIGHTWEIGHT", "No paid API, no GPU, no database. It runs on a free hosting plan.")]):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)
    st.markdown("### Technology")
    st.markdown("Python · Streamlit · scikit-learn (TF-IDF, cosine similarity) · NLTK (Porter stemmer) · pandas")
    st.markdown("### What it is not")
    st.write("SMARTFAQ is not generative AI. It uses no LLM, no RAG and no neural network. "
             "It cannot write new answers, only return stored ones.")
    st.markdown("### Limitations")
    st.markdown(
        "- TF-IDF compares words, not meaning. Synonyms that are not stored (for example 'scooter' and 'bike') may not match.\n"
        "- It only knows what is in the FAQ file. Accuracy depends on FAQ quality.\n"
        "- Spelling mistakes are not corrected.\n"
        "- The conversation has no memory: each question is answered on its own."
    )
    st.markdown("### Future scope")
    st.write("Sentence embeddings for semantic search, multilingual FAQs, voice input, an admin page to edit FAQs, "
             "usage analytics and a larger knowledge base.")
    st.markdown("### Demo data")
    st.write(f"The {engine.faq_count} FAQs, phone numbers and timings are sample information for demonstration. "
             "They do not describe any real college.")


# --------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title=f"{APP_NAME} - {SUBTITLE}", page_icon="🎓",
                       layout="centered", initial_sidebar_state="expanded")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("page", "Chat")
    st.session_state.setdefault("notice", "")
    apply_theme(st.session_state.get("dark", False))

    try:
        engine = load_engine()
    except DataError as error:                       # friendly, no traceback, no file paths
        st.error(str(error))
        st.stop()
    except Exception:
        LOG.exception("Unexpected start-up failure")
        st.error("SMARTFAQ could not start. Please reload the page or contact the maintainer.")
        st.stop()

    with st.sidebar:
        st.markdown(f"## {APP_NAME}")
        st.caption(SUBTITLE)
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        if st.session_state.history:
            st.button("Clear conversation", on_click=clear_chat, use_container_width=True)
        st.divider()
        st.toggle("Dark mode", key="dark")
        st.caption("Demo knowledge base: sample data, not real college policy.")
        st.caption(f"{APP_NAME} v2.0")

    try:
        page = st.session_state.page
        if page == "Chat":
            page_chat(engine)
        elif page == "FAQ Categories":
            page_categories(engine)
        elif page == "How It Works":
            page_how(engine)
        elif page == "Evaluation":
            page_evaluation(engine)
        else:
            page_about(engine)
    except Exception:
        LOG.exception("Unexpected error while drawing page")
        st.error("Something went wrong while showing this page. Please try again.")


if __name__ == "__main__":
    main()
