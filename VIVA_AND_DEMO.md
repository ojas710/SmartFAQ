# SMARTFAQ: Viva answers and demo script

## Viva questions (short, technically correct answers)
1. **Why NLP?** Students phrase the same question in many ways. NLP preprocessing turns free text into a form a computer can compare.
2. **Why TF-IDF?** The FAQ set is small and vocabulary-specific. TF-IDF is fast, needs no training data, is fully explainable, and gives a score I can show. Heavier models would be overkill.
3. **What is TF-IDF?** Term Frequency x Inverse Document Frequency. A word's weight is high if it is frequent in this text and rare across all FAQ wordings. scikit-learn uses the smoothed form idf = ln((1+N)/(1+df)) + 1 and normalises each vector to length 1.
4. **What is cosine similarity?** The cosine of the angle between two vectors: (A.B)/(|A||B|). 1 means same direction (same weighted words), 0 means no shared words.
5. **Why cosine?** It ignores vector length, so a long and a short question about the same topic still match. Euclidean distance would be affected by length.
6. **Why not an LLM?** The task is retrieval of approved answers. An LLM can invent wrong policy, costs money, needs an API and is hard to explain. TF-IDF returns only stored, verified answers.
7. **What if similarity is low?** If it is below 0.30, or half or more of the keywords are unknown, SMARTFAQ shows a fallback and the closest FAQ is shown only as information.
8. **Why remove stopwords?** Words like "the", "is", "what" appear everywhere and add noise to the similarity score.
9. **Why stemming?** It maps "timings", "timing", "time" to one root so different word forms match. Porter is a standard rule-based stemmer.
10. **Limitations?** Lexical only (no synonyms unless stored), limited to the FAQ file, no spell-check, no memory, small test set.
11. **How to improve?** Sentence embeddings for semantic matching, spell correction, more FAQ wordings, an admin page, usage analytics.
12. **How is it different from a generative chatbot?** It selects an existing answer; a generative model composes new text. Mine cannot hallucinate, but it cannot answer anything outside the FAQs.
13. **How did you evaluate it?** 50 test queries in three groups (demo, reworded, unrelated). I count correct answers plus correct refusals. Result: 41/50 = 82%, with 12/12 refusals and 18/27 on reworded questions. The failures are lexical (no word overlap). The set is small and developer-written.
14. **Why is the threshold required?** A nearest FAQ always exists, even for nonsense. The threshold converts "nearest" into "relevant". My threshold study shows 0.30 sits in the stable range.
15. **Two FAQs with similar scores?** The higher score wins; exact ties keep CSV order, so results are repeatable. If the gap is under 0.05 the app flags a near-tie and shows the alternatives.

Bonus: **Why the known-keyword rule?** TF-IDF ignores unseen words, so "Who won the cricket world cup?" became "cricket" and matched the cricket-team FAQ at 71%. Requiring more than half of the keywords to be known fixed this (false accepts went from 5 to 0 on my test set).

## 3 to 5 minute demo script
1. **Open SMARTFAQ (20 s).** Point to the tagline, suggested questions and the "demo data" note.
2. **Ask "What are the college timings?" (30 s).** Show answer, matched FAQ, category, similarity.
3. **Ask "When does college start?" then "What time do classes begin?" (40 s).** Same FAQ, different wording. Open "Why did I get this answer?" and point at the matched stored wording.
4. **Ask "What is the capital of Japan?" (30 s).** Show the refusal. Say: "A nearest FAQ always exists, the threshold and keyword rule stop me from forcing a wrong answer."
5. **Open How It Works (60 s).** Walk the six steps, then type "When does college start?" in the live inspector and read tokens, stopwords removed and stems.
6. **Open Evaluation (45 s).** Say the numbers are computed live. Show the threshold table and be upfront about the 18/27 reworded result and why it fails.
7. **Open FAQ Categories (20 s).** Click a category, click a question.
8. **Toggle Dark mode (10 s).** Finish with: "Honest limits: lexical matching, only the FAQ file, upgrade path is embeddings."
