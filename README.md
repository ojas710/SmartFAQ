# SMARTFAQ

**NLP-Powered College Information Assistant** · *Your College. One Answer Away.*

SMARTFAQ lets a student ask a question in their own words and returns the answer of the most
similar question in a college FAQ knowledge base. Every answer shows *why* it was chosen:
the matched FAQ, its category and the similarity score.

> It is a **retrieval** system. It is not generative AI: no LLM, no RAG, no neural network.
> The 49 FAQs are **sample/demo data**, not real college information.

## Features
- Natural-language questions, including different wordings of the same intent
- Transparent results: matched FAQ, category, similarity score, top-3 candidates, preprocessing output
- Responsible fallback: unrelated questions (e.g. "What is the capital of Japan?") are refused
- FAQ category browser with counts
- "How it works" page with a live pipeline inspector
- Evaluation page that measures accuracy and a threshold study on a built-in test set
- Light and dark themes, keyboard-friendly, responsive layout, graceful error handling

## NLP pipeline
```
question -> lowercase + spelling normalisation -> tokenise -> stopword removal -> stemming
         -> TF-IDF vector -> cosine similarity with every stored FAQ wording
         -> best FAQ -> threshold + known-keyword check -> answer or fallback
```
| Step | Why it exists |
|---|---|
| Normalisation | `Wi-Fi`, `wifi`, `Wi Fi` and `programme`/`program` must become one token |
| Stopword removal | Words like "the", "is", "what" say nothing about the topic. "college" is also removed because it is in nearly every FAQ |
| Porter stemming | `timings`, `timing` and `time` share one root |
| TF-IDF | Rare, topic-specific words count more than common ones |
| Cosine similarity | Compares direction of word-weight vectors, so question length matters little |
| Threshold (0.30) | A nearest FAQ always exists; the threshold decides whether it is relevant enough |
| Known-keyword rule | TF-IDF ignores unseen words, so "Who won the cricket world cup?" would shrink to "cricket". If half or more of a question's keywords are unknown, SMARTFAQ refuses |

Each FAQ has a main `question` plus optional `variations` (alternative wordings separated by `|`).
Every wording is indexed; the best-matching wording represents the FAQ.

## Project structure
```
SMARTFAQ/
├── app.py              Streamlit interface only
├── engine.py           NLP, retrieval and evaluation (no Streamlit code)
├── faqs.csv            Knowledge base: question, answer, category, variations
├── requirements.txt
├── .gitignore
├── README.md
└── .streamlit/config.toml   Fixed light base theme, no usage tracking
```
`engine.py` is separate so the NLP can be explained, tested and run without the UI (`python engine.py`).

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Print the evaluation report in the terminal: `python engine.py`

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository (`app.py` at the repo root).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set **Main file path** to `app.py`.
4. Click **Deploy**. No secrets or environment variables are needed.

## Editing the FAQs
Edit `faqs.csv` (UTF-8). Required columns: `question`, `answer`, `category`. Optional: `variations`.
Add 3 to 5 natural alternative wordings per FAQ. Do not copy wordings from the test set in `engine.py`.

## Test plan
Run each question in Chat and check the matched FAQ:

| Question | Expected FAQ |
|---|---|
| What are the college timings? / When does college start? / What time do classes begin? | College timings |
| How do I apply for admission? | How can I apply for admission? |
| What documents are required? | Documents for admission |
| Is attendance compulsory? | Is attendance compulsory? |
| Does the college have a library? | Library |
| Are placement opportunities available? | Placements |
| Do you have Wi-Fi? | Wi-Fi |
| What scholarships are available? | Scholarships |
| What is the capital of Japan? | Refused (fallback) |
| (empty), `hi`, `????` | Friendly message, no crash |

## Evaluation (measured with `python engine.py` or the Evaluation page)
50 test queries written by the developer, threshold 0.30:

| Group | Result |
|---|---|
| Demo questions | 11 / 11 |
| Reworded questions not stored in `faqs.csv` | 18 / 27 (66.7%) |
| Unrelated questions (should be refused) | 12 / 12 |
| **Overall** | **41 / 50 (82.0%)** |

These numbers were measured in development with a stand-in Porter stemmer because NLTK could not be installed
in the build sandbox. Run the Evaluation page on your machine for the authoritative figures; tiny differences are possible.
The set is small and was inspected while improving the dataset, so treat it as indicative, not a benchmark.
Threshold study: accuracy was identical for thresholds 0.10 to 0.35 and fell at 0.40 and above, so 0.30 is inside the stable range.

## Limitations
- TF-IDF matches words, not meaning. Unstored synonyms ("shift" vs "change", "scooter" vs "bike") can fail.
- It only knows what is in the FAQ file; accuracy depends on FAQ quality.
- No spelling correction and no conversation memory.
- Not a generative chatbot: it cannot compose new answers.

## Future scope
Sentence-embedding (semantic) search, multilingual FAQs, voice input, an admin page for FAQs, usage analytics,
a larger knowledge base.
