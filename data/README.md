# Demo Corpus & Gold-Standard Evaluation Set

This folder holds the files RAGNova will index, plus the test questions we use to *measure* whether retrieval works.

**Fill this file in on Day 1, before writing any code.** Writing the test set before the system is deliberate: it stops us from unconsciously tuning the system to flatter itself. See Chapter 1 §3.3 and §1.10.

---

## Folder layout

```
data/
├── documents/   PDFs and DOCX files
├── images/      PNG / JPG — include screenshots and photos containing text
└── audio/       WAV / MP3 clips, 30 s to 3 min
```

Target: 10–15 PDFs, 5–8 DOCX, 10–15 images (3+ screenshots, 3+ photos with visible text), 3–5 audio clips.

## Deliberate cross-modal test material

Make sure the corpus contains, on purpose:

- [x] An image **and** a document about the same topic → proves image→document retrieval (`images/screenshot_wifi_setup.png` and `documents/it_onboarding.docx`).
- [x] An audio clip mentioning a topic that also appears in a PDF → proves audio→document retrieval (`audio/hod_project_announcement.wav` and `documents/notice.pdf`).
- [x] A screenshot whose content is described in a document → proves text→image retrieval with a visible payoff (`images/screenshot_synopsis_portal.png` and `documents/notice.pdf`).

## File inventory

> **Corpus status: Multimodal corpus populated.** The documents folder contains the Ch6 validated starter set (preserving contract and retrieval test invariants). `images/` contains 15 images (5 screenshots of portals/WiFi dialogs with secure CA certificate validation, 4 campus posters with visible OCR text, 2 architecture/map diagrams, and 4 physical photo assets with visible text). `audio/` contains 4 spoken speech clips generated via speech synthesis matching the campus academic guidelines.

| File | Modality | What it contains | Notes |
|---|---|---|---|
| `documents/notice.pdf` | pdf | 2-page dept notice — synopsis submission process, deadlines, mid-term evaluation weighting | Generated (Ch6). Exercises single & multi-chunk paths. |
| `documents/library_hours.pdf` | pdf | 1-page library policy — hours, borrowing limits, fines | Generated (Ch6). |
| `documents/it_onboarding.docx` | docx | 2-page IT onboarding guide, one explicit page break | Generated (Ch6) — exercises ADR-008's page-break detection. |
| `images/screenshot_portal_login.png` | image | SSO login screen with student email and password fields | Screenshot with OCR-readable authentication guidelines. |
| `images/screenshot_wifi_setup.png` | image | Wi-Fi settings dialog for `RAGNOVA-STUDENT` (WPA2-Enterprise) | Screenshot matching `it_onboarding.docx`. Configures secure 802.1X certificate validation (`ragnova.edu`). |
| `images/screenshot_error_403.png` | image | 403 Forbidden screen warning against unauthorized downloads | Screenshot matching Acceptable Use Policy in `it_onboarding.docx`. |
| `images/screenshot_synopsis_portal.png` | image | Synopsis portal upload screen with 21st August deadline | Screenshot matching `notice.pdf`. |
| `images/screenshot_vpn_client.png` | image | SecureConnect VPN client connected to remote library gateway | Screenshot for off-campus journal access. |
| `images/notice_seminar_poster.png` | image | Department seminar poster on Multimodal RAG (Auditorium Hall B) | Poster notice with date, venue, topics. |
| `images/notice_library_fines.png` | image | Central Library borrowing rules and overdue fine schedule | Noticeboard graphic (2 Rs/day, 200 Rs cap, 4 books/14 days). |
| `images/notice_lab_rules.png` | image | AIML lab rules and account sharing prohibition | Noticeboard graphic matching IT policy. |
| `images/notice_midterm_schedule.png` | image | Final year project evaluation timetable & marking breakdown | Noticeboard graphic (40% prototype, 35% report, 25% viva). |
| `images/diagram_rag_architecture.png` | image | Architectural block diagram of RAGNova pipeline | System diagram covering PyMuPDF, Whisper, OpenCLIP, ChromaDB. |
| `images/diagram_campus_map.png` | image | Schematic campus map showing Library, IT Centre, AIML Block | Blueprint diagram for cross-modal queries. |
| `images/photo_id_card_sample.png` | image | Sample Student ID Card with photo, barcode, borrowing limits | Physical ID card photo asset for student identification. |
| `images/photo_lab_door_sign.png` | image | AIML Research Lab entrance door sign (Room 302, Dr. S. Rao) | Physical lab sign photo asset with visible OCR text. |
| `images/photo_library_desk_sign.png` | image | Central Library circulation counter standing desk sign | Physical counter sign photo asset (4 books/14 days, fines, drop box). |
| `images/photo_campus_building_plaque.png` | image | Academic Block B directory wall plaque | Physical plaque photo asset detailing department floor directory. |
| `audio/hod_project_announcement.wav` | audio | Spoken briefing on project evaluation weighting & August 21 deadline | Speech audio matching `notice.pdf` (40% demo, 35% report, 25% viva). |
| `audio/library_orientation_excerpt.wav` | audio | Orientation speech on library hours, 4 books loan, overdue fines | Speech audio matching `library_hours.pdf`. |
| `audio/it_helpdesk_wifi_instructions.wav` | audio | Helpdesk voice guide on connecting to RAGNOVA-STUDENT network | Speech audio matching `it_onboarding.docx` (secure CA cert guidance). |
| `audio/lab_assistant_briefing.wav` | audio | Briefing on VPN installation for remote journals & no pirated media | Speech audio matching Acceptable Use Policy. |

---

## Gold-standard question set

The evaluation set from Chapter 1 §1.10. Start with 5 questions on Day 1; grow to **20 text questions + 10 cross-modal queries** by Chapter 7, when we first measure Recall@5 and MRR.

**Current state (Ch12, 2026-09-29): 13 text questions, 8 cross-modal.** The remaining gap to 20 text questions isn't a content-quality shortcut away — the 3-file starter corpus (`notice.pdf`, `library_hours.pdf`, `it_onboarding.docx`) has been mined for essentially every distinct, cleanly-extractable fact; T6-T13 already cover what was left after T1-T5. Reaching 20 for real needs the corpus itself to grow, which is why the team's fuller target (10-15 PDF, 5-8 DOCX) further up this file still matters, not just this table.

### Text → text/document queries

| # | Question | Expected source file | Expected page / timestamp | Why it's a good test |
|---|---|---|---|---|
| T1 | If I don't get my system actually running by evaluation day, how many marks am I giving up? | `documents/notice.pdf` | page 2 | Pure paraphrase — shares no words with "the working prototype... will carry forty percent." Answer: 40%. |
| T2 | As an undergrad, how many items can I check out from the library at once, and for how long? | `documents/library_hours.pdf` | page 1 | Paraphrase — shares no words with "borrow up to four books... fourteen days." Answer: 4 books, 14 days. |
| T3 | What happens the first time someone gets caught sharing their login with a friend? | `documents/it_onboarding.docx` | page 2 | Paraphrase — shares no words with "network access will be suspended for a period of two weeks." |
| T4 | What software must students set up if they want to read digital journals from home? | `documents/it_onboarding.docx` | page 1 | Paraphrase — refers to the institute virtual private network client (VPN). |
| T5 | Where will the guide allotment list be announced? | `documents/notice.pdf` | page 1 | Paraphrase — department noticeboard and student portal. |
| T6 | How many people are allowed to work together on this project, at most? | `documents/notice.pdf` | page 1 | Paraphrase — shares no words with "teams of two or three members." Answer: 3. |
| T7 | Is there a maximum length for the synopsis, not counting the cover page or references? | `documents/notice.pdf` | page 1 | Paraphrase — shares no words with "should not exceed three pages, excluding the cover page and references." Answer: 3 pages. |
| T8 | If someone misses their individual viva without approval beforehand, what mark do they get for that part? | `documents/notice.pdf` | page 2 | Paraphrase — shares no words with "absent without prior approval... will receive a zero for that component." |
| T9 | What happens if part of our submission turns out to be copied from somewhere else, even just a small section? | `documents/notice.pdf` | page 2 | Paraphrase — shares no words with "treated as a case of academic dishonesty and referred to the disciplinary committee, independent of how small the copied portion is." |
| T10 | On a Sunday, what time does the library close? | `documents/library_hours.pdf` | page 1 | Paraphrase — shares no words with "nine in the morning until six in the evening on weekends and public holidays." Answer: 6 PM. |
| T11 | Can I extend my library loan if nobody else wants that book? | `documents/library_hours.pdf` | page 1 | Paraphrase — shares no words with "renewal of an existing loan is permitted once, provided no other student has placed a reservation." |
| T12 | Before I can start using my new institute email account, what do I need to do first? | `documents/it_onboarding.docx` | page 1 | Paraphrase — shares no words with "must be activated manually before first use... using the temporary password printed on your admission letter." |
| T13 | Am I allowed to download movies or paid software through the campus network without a license? | `documents/it_onboarding.docx` | page 2 | Paraphrase — shares no words with "downloading copyrighted material... without a valid licence is not permitted." Answer: no. |

> Include at least three questions whose wording shares **no keywords** with the source text (pure paraphrase). Those are the questions that prove semantic search beats Ctrl+F — and they are the ones to demo.

### Text → image queries (cross-modal)

| # | Query text | Expected image | Why it's a good test |
|---|---|---|---|
| I1 | Where can I see the Wi-Fi authentication screen for campus wireless? | `images/screenshot_wifi_setup.png` | Semantic text query retrieving the wireless configuration screenshot. |
| I2 | Find the diagram showing how ChromaDB and OpenCLIP connect together | `images/diagram_rag_architecture.png` | Conceptual query matching the system architecture diagram. |
| I3 | What poster shows the upcoming AI and Machine Learning seminar venue? | `images/notice_seminar_poster.png` | Text search retrieving the guest lecture seminar poster. |
| I4 | Where is the AI & Machine Learning Research Laboratory located and who heads it? | `images/photo_lab_door_sign.png` | Visual sign retrieval locating Room 302 and Dr. S. Rao. |

### Image → document queries (cross-modal)

| # | Query image | Expected document(s) | Why it's a good test |
|---|---|---|---|
| M1 | `images/screenshot_synopsis_portal.png` | `documents/notice.pdf` | Screenshot of the portal links directly to the written notice guidelines. |
| M2 | `images/screenshot_wifi_setup.png` | `documents/it_onboarding.docx` | Wi-Fi dialog screenshot links to the onboarding guide explaining SSID and login steps. |

### Audio → anything queries

| # | Audio clip (or spoken query) | Expected result | Why it's a good test |
|---|---|---|---|
| A1 | `audio/hod_project_announcement.wav` | `documents/notice.pdf` / `images/notice_midterm_schedule.png` | Spoken announcement retrieves the written evaluation weighting guidelines. |
| A2 | `audio/library_orientation_excerpt.wav` | `documents/library_hours.pdf` / `images/notice_library_fines.png` | Spoken orientation retrieves the library hours and fine policies. |

### Negative controls (should return "not found in the provided sources")

Questions the corpus genuinely cannot answer. These test whether the system **refuses to hallucinate** — objective O4. The UI scaffold and retrieval pipeline must refuse these without claiming false grounding or emitting fake citations.

| # | Question | Expected behaviour |
|---|---|---|
| N1 | What is the hostel mess menu for Wednesday lunch? | Refuses / states the sources don't cover it (empty citations) |
| N2 | How do I apply for a refund on tuition fees? | Refuses / states the sources don't cover it (empty citations) |

---

## Results log

Fill in as each chapter's evaluation runs. Numbers, not adjectives (Chapter 1 §1.10 rule).

| Date | Chapter | Recall@5 | MRR | Cross-modal Recall@5 | Notes / what changed |
|---|---|---|---|---|---|
| 2026-09-15 | Ch7 | 1.00 | 1.00 | N/A (no images/audio yet) | First real semantic search, 3 text queries (T1–T3) against the 6-chunk Ch6 starter corpus. All 3 hit at rank 1. Read alongside Ch7 §3.3: with only 3 questions, one miss would swing Recall@5 to 0.67 — promising, not yet strong evidence. Grow the gold set (T4 onward) before trusting this number in a report. |
| 2026-09-23 | Ch10 | N/A (this row is generation, not retrieval) | N/A | N/A | Answer-quality check (`scripts/evaluate_answers.py`), 7 questions (T1–T5, N1–N2), real Ollama (`llama3.2:3b`) + real text_index. Self-rated 1–5 (Ch1 §1.10's faithfulness criterion): T1/T2/T3/N1/N2 = 5 (correct, well-cited or correctly refused); T4/T5 = 4 (correct *answer text* — 5 related systems, Rs. 200 cap — but a citation-number error: T4 cited `[2]` for a fact actually on the `[1]` chunk, T5 cited `[2]` when only one chunk (`[1]`) had cleared the relevance threshold, i.e. the model invented an out-of-range number). Average 4.71/5. Both errors were caught automatically by `answer_query()`'s citation-range check (a logged warning, not a crash) — see Ch10 §6.3. Real, disclosed limitation: a low citation number is not independently verified against the claim it's attached to; only the **Sources list itself** (built from real Chunk metadata, never model text) is guaranteed correct. |
| 2026-09-28 | Ch12 | N/A (cross-modal only, see next column) | N/A | 1.00 | First real cross-modal measurement (`scripts/evaluate_cross_modal.py`), 8 questions (I1–I4 text→image, M1–M2 image→document via real Tesseract OCR, A1–A2 audio-topic→anything) against the real corpus — real CLIP (`ViT-B-32`), real OCR, real Whisper-indexed audio, real `retrieve()` with per-collection floors + RRF merge (ADR-007/009/010). All 8 hit, scores 0.236–0.713, comfortably above `MIN_IMAGE_RELEVANCE_SCORE=0.2` — no evidence the ADR-010 floor needs adjusting. Same caveat as the Ch7 row: n=8 is fragile evidence, one miss would swing any category's Recall@5 by 0.25–0.50 — a real first measurement, not yet a number to cite confidently in a report. Grow the cross-modal gold set before trusting this further. |
| 2026-09-29 | Ch12 | 1.00 | 0.81 | N/A (unchanged, see above row) | Re-measured (`scripts/evaluate_retrieval.py`) after growing the text gold set from 3 to 13 questions (T1–T13, all real corpus content, zero keyword overlap with source text). All 13 hit within top 5 — Recall@5 still 1.00, but MRR dropped from the earlier 1.00 to a more honest **0.81** (several hits land at rank 2, not always rank 1), which is exactly the kind of real signal a 3-question sample was too small to ever show. Read alongside the 2026-09-15 row's own caveat: 13 is a real improvement over 3 but still short of the file's own 20-question target — the 3-file starter corpus has been mined for essentially every distinct extractable fact, so reaching 20 needs the corpus itself to grow, not more questions squeezed from the same 3 files. |

---

## Licensing / privacy note

Use only files you own or that are freely shareable. Do not commit anything confidential — this folder may end up in the submitted report or a public repository.
