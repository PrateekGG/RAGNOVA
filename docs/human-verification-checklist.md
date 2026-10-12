# What still needs a person: the checklist

Everything the code and the measurements could settle has been settled and written down in `data/README.md`'s results log. What is left needs a person, a microphone, a network cable, or a stranger. This page makes each of those a short, ordered job, says exactly what to record, and says where. Nothing here asks you to trust a claim: each item ends with something you can see or a file you can open.

Order matters a little: do **1** before any of the others, and **4** before **5** is not required (they are independent).

---

## 1. Before any session, on the machine that will run it

Ten minutes, once per machine.

1. **Ollama.** Open Ollama's Settings and switch **Auto-download updates** and **Cloud** off. (On 2026-10-05 an update was left half applied and the model silently stopped starting until it was reinstalled; nothing in the app noticed until a question was asked.) Do not change any Ollama setting while a model is downloading: it restarts the server and ends the download.
2. **The model.** `ollama list` shows `llama3.2:3b`. If not: `ollama pull llama3.2:3b` (about 2 GB).
3. **Start the page from the project root**, so it reads `.streamlit/config.toml` (localhost only, no usage statistics, no email prompt). Use the module form, which runs through the Python interpreter instead of the `streamlit.exe` launcher:
   ```bash
   .venv\Scripts\python -m streamlit run src/app.py
   ```
   **If Windows says "An Application Control policy has blocked this file"** (seen 2026-10-11 on `streamlit.exe`, and on native libraries such as scikit-learn's, scipy's, `av` and `pybase64`), that is Smart App Control (Windows Security, App & browser control; state 1 = on here). It blocked each unfamiliar file the first time it was loaded and allowed it on a later try (`import sentence_transformers` failed once and then imported cleanly four times in a row), so: use the module form above, and if an import is blocked, wait a minute and start again. **Do a full warm-up on the demo machine while it is online** (start the page, ask a text question, an image question, a voice question, upload one file) so every library has loaded once. Whether a cold start with the network off is blocked has not been tested; item 3 below is where to find out. Smart App Control has no allow-list: the only switch is Windows Security, and turning it off cannot be undone without resetting Windows, so that is your decision, not a script's.
   The browser does not open by itself; open the URL it prints (`http://127.0.0.1:8501`).
4. **Press "Test the model" in the sidebar** and wait for `llama3.2:3b answered: 'OK'`. The first call after Ollama starts can take a full minute (measured 2026-10-07), so do this before anyone is watching. If it shows an error, fix that before anything else; the error text is Ollama's own.
5. Check the sidebar's counts match the data folder (documents, images, audio) and the vector index line. If you added files, run `python scripts/build_index.py` first.

**Record:** nothing, unless it fails; then the error text in the results log.

---

## 2. The microphone path

The code under it (`transcribe_audio_bytes`, then `transcribe_query`) was run on a synthetic recording by `scripts/verify_offline.py`. What has never run is a **real voice through the browser's microphone and the page's recorder**; that needs you.

1. In the page, open **Spoken / Voice Query** and allow the microphone if the browser asks (it will treat `127.0.0.1` as a secure origin).
2. Record, then press **Transcribe & Submit Audio**, for each of these two questions, spoken at a normal pace:
   - "How many marks does the prototype carry?"
   - "How many books can an undergraduate borrow, and for how long?"
3. For each, note: the transcript the page shows as your message (word for word), whether the answer is right, whether the citation points at `notice.pdf` page 2 / `library_hours.pdf` page 1, and roughly how many seconds it took.
4. Try once with some background noise (a fan, a room) and note whether the transcript degrades.

**Pass:** the transcript is recognisable and the right source is cited for both quiet recordings.
**Record:** one results-log row in `data/README.md` ("Microphone path, N questions, transcripts, pass or fail, any misrecognised words").

---

## 3. The real network-off demonstration

Methodology 8.5. The software side is proven (a guard inside Python refused every outside connection and the run needed none, `data/eval/offline_2026-10-05.txt`). What it cannot prove is the machine with its network really off, through the page.

**Before** disconnecting (once, while online): run the page and one query so every model file is downloaded and cached, then put `RAGNOVA_OFFLINE=1` in `.env` so the Hugging Face libraries do not even try to reach the Hub. (Without it, loading the embedding model made 31 Hub attempts and stalled the first question about 49 s.)

1. Confirm the baseline: `python scripts/verify_offline.py --offline-switch` ends with zero blocked connection attempts.
2. **Switch the network off yourself**: Windows Settings, Network and internet, turn Wi-Fi off and unplug any cable (or use airplane mode). The app runs on this machine only and needs no network.
3. Start the page as in section 1 and, in order, with the network off:
   1. upload a **new** document, image or audio file in the sidebar and press Save & Index File; the counter rises;
   2. ask a text question about it, and one about an existing document;
   3. attach a query image and ask a question about it;
   4. ask one question by voice (section 2);
   5. open the citations of each answer.
4. Note anything that errored, stalled, or looked different from the online run.

**Pass:** every step completes with no functional degradation (the methodology's wording).
**Record:** a results-log row: date, which steps passed, and the longest wait.

---

## 4. Feedback sessions with outside testers

Chapter 3 §3.5.3's rule: the team does not grade itself. The page has a rating form under every real answer; this section is the protocol around it.

**Set up (2 minutes):** section 1, then leave the engine on **Live Multimodal RAG**. Never leave it on the canned engine for a session: its answers are hand-written and are logged as `scaffold-mock`, which the summary should exclude.

**Each tester (about 15 minutes), a card with these ten tasks:**
- six text questions picked from the gold set (a mix of `data/README.md`'s T rows, including at least one paraphrased and one about a paper);
- two questions about an image (for example I1, I9);
- one spoken question;
- one question the corpus cannot answer (for example N1), where the right behaviour is a refusal.

For each answer the tester rates it **1 to 5** in the form and may type a comment. Testers enter **initials, not names**.

**Consent and privacy.** Tell testers the rating, their initials and any comment are saved on this machine. The raw log (`feedback/feedback.jsonl`) is **git-ignored on purpose** (it holds initials and free text); only the summary is committed. If the team decides to publish the raw log, ask each tester first.

**After:** `python scripts/summarize_feedback.py` prints the count, the mean, a histogram, the split by tester (team members and outsiders reported separately), and every answer rated 2 or lower in full. Paste the numbers into `docs/feedback-log.md`, and open the low-rated answers: each one is either a retrieval miss, a model limit, or a gold-set mistake, and the log should say which.

**Record:** `docs/feedback-log.md` plus one results-log row (testers, answers rated, mean, what the low ratings were).

---

## 5. Independent rating of the answers

Every answer-quality figure so far was rated by the AI assistant that helped build the system (4.21/5 on 29 answers, 2026-10-05), which methodology section 9 lists as a threat to validity. A blind sheet makes the independent version a half-hour job.

```bash
python scripts/export_rating_sheet.py export data/eval/answers_2026-10-07_images-default.txt sheet.csv key.csv
```

Send `sheet.csv` to the rater (it opens in Excel), **keep `key.csv` yourself**. The sheet shows only the question, the answer and the sources the system cited; the expected answer, whether the system refused and the gold ids are hidden, and the rows are shuffled. The rater fills four columns per row, 1 to 5: **faithfulness** (is every claim supported by the cited sources), **relevance** (does it address the question), **context** (were the retrieved sources useful), **citation** (do the markers point at sources that support the claims), and may add a comment. For a correct refusal, rate relevance and faithfulness on whether refusing was right.

```bash
python scripts/export_rating_sheet.py import sheet.csv key.csv
```

prints the mean per dimension, by kind of question, and every answer with a rating of 2 or lower. It refuses a sheet with a blank or out-of-range rating and names the row and column. Compare with the old figure **by average only**: it was one combined score and its per-answer values were not kept.

Methodology 8.2 asks for at least three raters, one external, and a written rubric agreed beforehand; one rater is a first step, not the protocol.

**Record:** a results-log row with the rater's role (external or not), n, the four means, and the low-rated answers.

---

## 6. Python 3.11

The project's documentation names Python 3.11 and it has been run on 3.13 (the pinned `requirements.txt`) and 3.14. 3.11 itself has not been tried. If you install it:

```bash
py -3.11 -m venv .venv311
.venv311\Scripts\python -m pip install -r requirements.txt
.venv311\Scripts\python -m pytest -q
```

**Pass:** the install completes and the suite passes as it does on 3.13 (all passed, one expected failure). If a pinned package has no wheel for 3.11, note which: that is the finding.
**Record:** a results-log row. Do not point this environment at the real `chroma_db/`: an index built by one ChromaDB version cannot be read by another (the error message says so); use a scratch `CHROMA_PERSIST_DIR`.

---

## 7. The pinned environment, and your own files in the page

Two jobs from 2026-10-10 (ADR-016) that only a person at this machine can finish.

**7a. Run the pinned environment's tests once it will start.** On 2026-10-10 `.venv-pinned313` (Python 3.13, Streamlit 1.41.1, ChromaDB 0.5.23) could not import scikit-learn: Windows reported *"An Application Control policy has blocked this file"* for a native library (`_argkmin_classmode`, and once `pyduccfft`), reproduced by `python -I -c "import sentence_transformers"` with no repository code. That is a Windows security policy (Smart App Control or an organisation policy), so it is yours to decide about; nothing in the repository tried to get around it. Until it runs, the 52 new tests in `test_scoped_sources.py`, `test_attachments.py` and `test_app.py` have passed only in `.venv`.

```bash
.venv-pinned313\Scripts\python -m pytest -q
```

**Pass:** all passed, one expected failure. The test to watch is `test_a_picked_file_that_is_no_longer_indexed_is_dropped_instead_of_crashing_the_page`: Streamlit 1.41.1 *raises* on a stored pick that is not among the options (1.65 quietly drops it), so this guard can only fail there.
**Record:** a results-log row with the count, or the error text if it is still blocked.

**7b. Try your own file, as the first person to use it did.** Start the page as in section 1.

1. Add a document of your own with **Add to Corpus** and **Save & Index File**. Expect: a green line naming the file and its chunk count, a banner "Answering from *file* only", and the file selected under **Answer from**.
2. Ask "who is this person?" or "what is in this document?" (whatever fits the file), then press one of the three one-click questions. Expect an answer that cites the file.
3. Ask something the file cannot answer ("what is the capital of France?"). Expect the refusal, with no sources listed under it. **If it answers anyway, that is the finding to record:** inside a scope the model's own refusal is the only guard (four of four were refused in the check, a small sample).
4. Press **Use everything** and ask the same "who is this person?". Expect the refusal: that is the old corpus-wide behaviour, kept on purpose.
5. Open **Visual / Query Image**, attach a screenshot that contains text, and ask "what does this image say?". Expect the note "read N words of text from it" and an answer that cites the attachment as [1]; the answer is often only the headline. A photo with no text should say that it has none.
6. **Names, typos, follow-ups (ADR-018; on by default).** With no scope, ask "who is Alex Morgan?" (should answer from the ID card), "who is Dr. S. Sharma?" (should refuse: no such person), "tell me lab rul no. 2" (should give the credential-sharing rule), then "tell me more" (should continue, not refuse), and "what is greedy algorithm tell me in brief" (a short paragraph, with citations). Under each answer only the cited sources should be listed. **Record** what each showed.
7. **Afterwards, delete the file you uploaded** from `data/documents/` (or `images/`, `audio/`): uploads land in the same folders as the shipped corpus, so a personal file sits there untracked and `git add data/` would stage it. Removing it from the index as well needs `python scripts/build_index.py`.

**Record:** a results-log row: what you added, what each step showed, and anything that surprised you.

---

## When something here fails

Write the failure down in the results log with the real output, in the same words the log's other rows use: what you ran, what you saw, what you did not check. A recorded failure is evidence; an unrecorded one is just a rumour.
