"""
SMARTFAQ engine: the NLP and retrieval logic (no Streamlit code in here).

Pipeline
    question -> preprocessing -> TF-IDF vector -> cosine similarity
             -> best FAQ -> threshold check -> answer or fallback

Run `python engine.py` to print the evaluation report in the terminal.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer          # pure Python, needs no downloaded data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------- settings
CSV_PATH = Path(__file__).with_name("faqs.csv")   # relative to this file: works locally and on the cloud
THRESHOLD = 0.30        # minimum cosine similarity needed to give an answer (see README / Evaluation page)
MIN_KNOWN_FRACTION = 0.5  # more than this share of the question's keywords must exist in the FAQ vocabulary
TOP_K = 3               # how many candidate FAQs to report for transparency
MAX_QUERY_CHARS = 300   # longer input is truncated, nobody needs 5000 characters

NO_MATCH_MESSAGE = "I couldn't find a strong match for that question."
EMPTY_MESSAGE = "Please type a question first."
TOO_SHORT_MESSAGE = ("Your question has no meaningful keywords left after cleaning. "
                     "Try something like \"What are the college timings?\"")

# ------------------------------------------------------------ preprocessing
# Common English function words. They appear in almost every sentence, so they
# say nothing about the topic. A fixed list is used (instead of downloading
# NLTK's list) so the app behaves identically on every machine, offline too.
STOP_WORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further get had has
have having he her here hers him his how i if in into is it its me more most my no nor of off on
once only or other our out over own same she should so some such than that the their them then
there these they this those through to too under until up us very was we were what when where
which while who whom why will with would you your yours
""".split())

# "college" appears in almost every FAQ, so it carries no information about WHICH
# FAQ is wanted. Without this, "Which college is best in Delhi?" would partly match
# every FAQ through that single word.
DOMAIN_STOP_WORDS = frozenset({"college"})

# Spelling variants that must become one token (applied after lowercasing).
NORMALISATION_RULES = [
    (re.compile(r"\bwi[\s-]?fi\b"), "wifi"),
    (re.compile(r"\be[\s-]?mail\b"), "email"),
    (re.compile(r"\btime[\s-]?table\b"), "timetable"),
    (re.compile(r"\binstal+ments?\b"), "installment"),     # British / American spelling
    (re.compile(r"\bprogramme(s?)\b"), r"program\1"),
]
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")   # letters and digits only: this removes punctuation
STEMMER = PorterStemmer()                  # "timings" -> "time", "classes" -> "class"


def preprocess_steps(text):
    """Run the preprocessing pipeline and return every intermediate stage (used by the UI)."""
    lowered = str(text).lower()
    normalised = lowered
    for pattern, replacement in NORMALISATION_RULES:
        normalised = pattern.sub(replacement, normalised)
    tokens = TOKEN_PATTERN.findall(normalised)
    kept = [t for t in tokens if t not in STOP_WORDS and t not in DOMAIN_STOP_WORDS]
    stemmed = [STEMMER.stem(t) for t in kept]
    return {"lowercase": lowered, "normalised": normalised, "tokens": tokens,
            "without_stopwords": kept, "stemmed": stemmed}


def preprocess(text):
    """Return the cleaned text as one string, e.g. 'When does college start?' -> 'start'."""
    return " ".join(preprocess_steps(text)["stemmed"])


# --------------------------------------------------------------- data classes
class DataError(Exception):
    """The FAQ file is missing or unusable. The message is safe to show to users."""


@dataclass
class Candidate:
    question: str
    category: str
    score: float
    matched_phrase: str     # the stored wording that matched best (question or one of its variations)


@dataclass
class Answer:
    status: str             # "answer" | "no_match" | "empty" | "too_short"
    message: str = ""
    answer: str = ""
    question: str = ""
    category: str = ""
    score: float = 0.0
    matched_phrase: str = ""
    processed_query: str = ""
    known_fraction: float = 1.0     # share of the question's keywords that the FAQ vocabulary contains
    candidates: list = field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------- data loading
def load_faqs(path=CSV_PATH):
    """Read and validate the FAQ file. Raises DataError with a friendly message."""
    name = Path(path).name
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except FileNotFoundError:
        raise DataError(f"'{name}' was not found. Keep it in the same folder as app.py.")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise DataError(f"'{name}' could not be read. Check that it is a valid UTF-8 CSV file.")

    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ("question", "answer", "category") if c not in df.columns]
    if missing:
        raise DataError(f"'{name}' must have the columns question, answer, category "
                        f"(optional: variations). Missing: {', '.join(missing)}.")
    if "variations" not in df.columns:
        df["variations"] = ""

    df = df.apply(lambda col: col.str.strip())
    df = df[(df["question"] != "") & (df["answer"] != "") & (df["category"] != "")]
    df = df.drop_duplicates(subset="question", keep="first").reset_index(drop=True)
    if df.empty:
        raise DataError(f"'{name}' contains no usable FAQ rows.")
    return df


# ------------------------------------------------------------------- the engine
class FAQEngine:
    """Builds the TF-IDF index once, then answers questions."""

    def __init__(self, csv_path=CSV_PATH, threshold=THRESHOLD):
        self.threshold = threshold
        self.faqs = load_faqs(csv_path)

        # Every FAQ is stored under its main question PLUS optional alternative wordings
        # (the "variations" column, separated by "|"). Each wording is one row of the index.
        self.phrases, owners = [], []
        for faq_id, row in self.faqs.iterrows():
            wordings = [row["question"]] + [v.strip() for v in row["variations"].split("|") if v.strip()]
            self.phrases += wordings
            owners += [faq_id] * len(wordings)
        self.phrase_owner = np.array(owners)

        cleaned = [preprocess(p) for p in self.phrases]
        self.vectorizer = TfidfVectorizer(token_pattern=r"\S+")   # text is already tokenised
        try:
            self.phrase_vectors = self.vectorizer.fit_transform(cleaned)
        except ValueError:
            raise DataError("The FAQ questions contain no usable words.")

    # -- basic facts about the knowledge base (shown in the UI)
    @property
    def faq_count(self):
        return len(self.faqs)

    @property
    def category_counts(self):
        return self.faqs["category"].value_counts().sort_index()

    @property
    def vocabulary_size(self):
        return len(self.vectorizer.vocabulary_)

    def questions_in(self, category):
        return self.faqs.loc[self.faqs["category"] == category, "question"].tolist()

    # -- retrieval
    def retrieve(self, query):
        """Return (preprocessing stages, top candidate FAQs, known-word fraction). No decision is made here."""
        steps = preprocess_steps(query)
        words = steps["stemmed"]
        if not words:
            return steps, [], 0.0
        # TF-IDF can only "see" words that exist in the FAQ vocabulary; unknown words are silently
        # dropped. So "Who won the cricket world cup?" would shrink to just "cricket" and match the
        # cricket-team FAQ strongly. Counting the unknown words lets us detect that situation.
        known_fraction = sum(w in self.vectorizer.vocabulary_ for w in words) / len(words)
        query_vector = self.vectorizer.transform([" ".join(steps["stemmed"])])
        scores = cosine_similarity(query_vector, self.phrase_vectors)[0]

        # Highest score first. Ties keep CSV order (stable sort), so results are repeatable.
        candidates, seen = [], set()
        for idx in np.argsort(-scores, kind="stable"):
            faq_id = int(self.phrase_owner[idx])
            if faq_id in seen:
                continue                      # a FAQ is reported once, with its best wording
            seen.add(faq_id)
            row = self.faqs.iloc[faq_id]
            candidates.append(Candidate(row["question"], row["category"],
                                        float(scores[idx]), self.phrases[idx]))
            if len(candidates) == TOP_K:
                break
        return steps, candidates, known_fraction

    def ask(self, query):
        """Full pipeline: validate input, retrieve, apply the threshold."""
        if not isinstance(query, str) or not query.strip():
            return Answer(status="empty", message=EMPTY_MESSAGE)
        query = " ".join(query.split())[:MAX_QUERY_CHARS]     # normalise whitespace, cap length

        steps, candidates, known_fraction = self.retrieve(query)
        processed = " ".join(steps["stemmed"])
        if not candidates:
            return Answer(status="too_short", message=TOO_SHORT_MESSAGE, processed_query=processed)

        best = candidates[0]
        covered = known_fraction > MIN_KNOWN_FRACTION
        note = ""
        if len(candidates) > 1 and best.score - candidates[1].score < 0.05 and best.score >= self.threshold:
            note = "Another FAQ scored almost the same. Check the alternatives below."

        if best.score < self.threshold or not covered:
            reason = ("Most of your keywords do not appear in the FAQ knowledge base."
                      if not covered else "")
            return Answer(status="no_match", message=NO_MATCH_MESSAGE, score=best.score,
                          question=best.question, processed_query=processed, note=reason,
                          known_fraction=known_fraction, candidates=candidates)
        row = self.faqs[self.faqs["question"] == best.question].iloc[0]
        return Answer(status="answer", answer=row["answer"], question=best.question,
                      category=best.category, score=best.score, matched_phrase=best.matched_phrase,
                      processed_query=processed, known_fraction=known_fraction,
                      candidates=candidates, note=note)


# ----------------------------------------------------------------- evaluation
# Test queries. "expected" is the FAQ that should be returned, or None when the
# system should refuse to answer.
#   demo         : wordings used in the project brief / faculty demo
#   paraphrase   : the same intent worded differently from anything stored in faqs.csv
#   out_of_scope : questions the knowledge base cannot answer, must be rejected
EVAL_SET = [
    ("demo", "What are the college timings?", "What are the college timings?"),
    ("demo", "When does college start?", "What are the college timings?"),
    ("demo", "What time do classes begin?", "What are the college timings?"),
    ("demo", "When does college finish?", "What are the college timings?"),
    ("demo", "How do I apply for admission?", "How can I apply for admission?"),
    ("demo", "What documents are required?", "What documents are required for admission?"),
    ("demo", "Is attendance compulsory?", "Is attendance compulsory?"),
    ("demo", "Does the college have a library?", "Does the college have a library?"),
    ("demo", "Are placement opportunities available?", "Are placement opportunities available?"),
    ("demo", "Do you have Wi-Fi?", "Does the college provide Wi-Fi?"),
    ("demo", "What scholarships are available?", "What scholarships are available?"),

    ("paraphrase", "At what time are lectures over for the day?", "What are the college timings?"),
    ("paraphrase", "Do you work on Saturdays?", "Which days is the college open?"),
    ("paraphrase", "Who is allowed to apply?", "What is the eligibility criteria for admission?"),
    ("paraphrase", "Till when can I fill the admission form?", "What is the last date to apply for admission?"),
    ("paraphrase", "What papers must I submit while enrolling?", "What documents are required for admission?"),
    ("paraphrase", "Which programmes can I study here?", "What courses does the college offer?"),
    ("paraphrase", "Total duration of the BCA degree?", "How long is the BCA course?"),
    ("paraphrase", "Can I shift to a different course?", "Can I change my course after admission?"),
    ("paraphrase", "Do I need to attend a minimum percentage of lectures?", "Is attendance compulsory?"),
    ("paraphrase", "I missed many classes because of fever", "What if my attendance is low?"),
    ("paraphrase", "When do the final exams happen?", "When are the semester exams held?"),
    ("paraphrase", "Where will I get my marks?", "How can I check my exam results?"),
    ("paraphrase", "How many marks do I need to pass a subject?", "What are the passing marks?"),
    ("paraphrase", "How much does the course cost per year?", "What is the fee structure?"),
    ("paraphrase", "Can I pay in two parts?", "Can I pay fees in instalments?"),
    ("paraphrase", "Is UPI accepted for fee payment?", "How can I pay my fees online?"),
    ("paraphrase", "Where do I find the lecture schedule?", "Where can I find the class timetable?"),
    ("paraphrase", "How many days can I keep a borrowed book?", "How many books can I borrow from the library?"),
    ("paraphrase", "Does the college place students in companies?", "Are placement opportunities available?"),
    ("paraphrase", "Is there help to find an internship?", "Does the college help with internships?"),
    ("paraphrase", "How can I apply for financial aid?", "How do I apply for a scholarship?"),
    ("paraphrase", "Can I park my scooter?", "Is there parking for bikes and cars?"),
    ("paraphrase", "Is the internet free for students?", "Does the college provide Wi-Fi?"),
    ("paraphrase", "Do you have a lab for programming?", "Are computer labs available?"),
    ("paraphrase", "What if I lose my student card?", "What should I do if I lose my ID card?"),
    ("paraphrase", "Are there any cultural programs?", "What events and fests are held in the college?"),
    ("paraphrase", "How do I report a complaint against a teacher?", "Whom do I contact for a complaint or personal problem?"),

    ("out_of_scope", "What is the capital of Japan?", None),
    ("out_of_scope", "Who won the cricket world cup?", None),
    ("out_of_scope", "How do I bake a chocolate cake?", None),
    ("out_of_scope", "What is the weather today?", None),
    ("out_of_scope", "Tell me a joke", None),
    ("out_of_scope", "Explain quantum computing", None),
    ("out_of_scope", "How do I renew my passport?", None),
    ("out_of_scope", "Which phone should I buy?", None),
    ("out_of_scope", "Is there a college in France?", None),
    ("out_of_scope", "Which is the best college in Delhi?", None),
    ("out_of_scope", "Where can I buy cheap books online?", None),
    ("out_of_scope", "How do I get a driving licence?", None),
]

OUTCOME_LABELS = {"correct": "Correct", "wrong_answer": "Wrong FAQ", "missed": "Missed (rejected)",
                  "correct_rejection": "Correctly rejected", "false_accept": "False accept"}
SWEEP_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def _classify(expected, predicted, score, known_fraction, threshold):
    answered = predicted is not None and score >= threshold and known_fraction > MIN_KNOWN_FRACTION
    if expected is None:
        return "false_accept" if answered else "correct_rejection"
    if not answered:
        return "missed"
    return "correct" if predicted == expected else "wrong_answer"


def _summarise(rows):
    total = len(rows)
    good = sum(r["outcome"] in ("correct", "correct_rejection") for r in rows)
    return {"total": total, "good": good, "accuracy": good / total if total else 0.0}


def evaluate(engine, eval_set=None, sweep=SWEEP_THRESHOLDS):
    """Measure retrieval quality on the test queries. Everything returned is measured, not assumed."""
    eval_set = eval_set or EVAL_SET
    known = set(engine.faqs["question"])
    retrievals = []
    for kind, query, expected in eval_set:
        if expected is not None and expected not in known:
            raise ValueError(f"Evaluation expects an FAQ that is not in the dataset: {expected!r}")
        _, candidates, known_fraction = engine.retrieve(query)
        top = candidates[0] if candidates else None
        retrievals.append((kind, query, expected, top.question if top else None,
                           top.score if top else 0.0, known_fraction))

    def rows_at(threshold):
        return [{"kind": k, "query": q, "expected": e, "retrieved": p, "score": s,
                 "outcome": _classify(e, p, s, kf, threshold)} for k, q, e, p, s, kf in retrievals]

    rows = rows_at(engine.threshold)
    groups = {kind: _summarise([r for r in rows if r["kind"] == kind])
              for kind in ("demo", "paraphrase", "out_of_scope")}
    sweep_rows = []
    for t in sweep:
        r = rows_at(t)
        counts = {o: sum(x["outcome"] == o for x in r) for o in OUTCOME_LABELS}
        sweep_rows.append({"threshold": t, "accuracy": _summarise(r)["accuracy"], **counts})
    return {"threshold": engine.threshold, "rows": rows, "overall": _summarise(rows),
            "groups": groups, "sweep": sweep_rows}


if __name__ == "__main__":
    engine = FAQEngine()
    report = evaluate(engine)
    print(f"FAQs: {engine.faq_count} | categories: {len(engine.category_counts)} | "
          f"indexed wordings: {len(engine.phrases)} | threshold: {engine.threshold}\n")
    for r in report["rows"]:
        print(f"{OUTCOME_LABELS[r['outcome']]:<19} {r['score']:.3f}  [{r['kind']}] {r['query']}"
              f"  ->  {r['retrieved']}")
    print()
    for name, g in report["groups"].items():
        print(f"{name:<13} {g['good']}/{g['total']} = {g['accuracy']:.1%}")
    o = report["overall"]
    print(f"{'overall':<13} {o['good']}/{o['total']} = {o['accuracy']:.1%}\n\nThreshold sweep:")
    for s in report["sweep"]:
        print(f"  {s['threshold']:.2f}  accuracy {s['accuracy']:.1%}  missed {s['missed']}  "
              f"wrong {s['wrong_answer']}  false accepts {s['false_accept']}")
