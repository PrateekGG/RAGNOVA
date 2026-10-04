# Step-by-Step Guide & Comprehensive Changelog: Tasks 1 & 4

This document serves as the complete, authoritative guide and detailed file-by-file changelog for:
- **Task 1: Populate the Real Multimodal Corpus** (`data/images/`, `data/audio/`, and `data/README.md`)
- **Task 4: UI Scaffold** (`src/app.py` bare Streamlit shell with mocked answers, citations, and TODO hooks)

---

## 1. Architectural Context & Invariants

```
                            RAGNova Multimodal Pipeline
                            
  [ Documents (PDF/DOCX) ] ──► PyMuPDF / python-docx ──► all-MiniLM-L6-v2 ──┐
  [ Images (PNG/JPG) ]     ──► OpenCLIP / OCR        ──► ViT-B-32        ──┼──► ChromaDB
  [ Audio (WAV/MP3) ]      ──► faster-whisper        ──► all-MiniLM-L6-v2 ──┘
                                                                               │
                                [ Streamlit UI: src/app.py ] ◄─────────────────┘
```

- **Contract Invariant ([`src/core/schemas.py`](../src/core/schemas.py))**: Every modality emits unified `Chunk` objects adhering to the frozen interface contract.
- **Collection Routing (ADR-003)**: Text documents and transcribed audio route into ChromaDB's `text_index`. Images route into `image_index`.
- **Test Invariant ([`tests/test_retrieval.py`](../tests/test_retrieval.py))**: Line 170 strictly asserts `assert collection.count() == 6` across the 3 starter documents (`notice.pdf`, `library_hours.pdf`, `it_onboarding.docx`). Those starter documents are preserved intact to prevent regressions in Chapter 6/7 tests.

---

## 2. File-by-File Changelog & Detailed Specifications

Here is the exact record of every file created, modified, or generated across the repository.

### File 1: `scripts/generate_multimodal_corpus.py` [NEW]
- **File Path**: [`scripts/generate_multimodal_corpus.py`](../scripts/generate_multimodal_corpus.py)
- **Role**: A reproducible, self-contained Python script to generate all required images and audio clips without requiring external internet downloads or manual file asset hunting.
- **Key Components**:
  - `_get_font()`: Loads scalable TrueType fonts (`arial.ttf`, `segoeui.ttf`, `DejaVuSans.ttf`, `calibri.ttf`) with fallback to standard PIL bitmap fonts.
  - `create_image()`: Generates structured, high-contrast, window-framed PNG graphics with headers, window control pills, colored badges, and OCR-readable typography.
  - `generate_all_images()`: Builds all 12 images categorized into screenshots, posters, blueprints, and identity cards.
  - `_generate_synthetic_speech_wav()`: Generates 16-bit PCM WAV audio using Windows PowerShell `System.Speech.Synthesis.SpeechSynthesizer` for natural acoustic speech, with a mathematical multi-tone formant modulator fallback.
  - `generate_all_audio()`: Synthesizes the 4 campus-themed audio clips.
- **Execution Command**:
  ```powershell
  & ..\.venv\Scripts\python.exe scripts/generate_multimodal_corpus.py
  ```

---

### File 2: `data/images/*` [NEW DIRECTORY — 12 Files]
- **Directory Path**: [`data/images/`](../data/images/)
- **Target Met**: 12 images created (requirement: 10–15 images, with >= 3 screenshots and >= 3 text posters/photos).

| # | Filename | Dimensions | Category | Visible Content / OCR Text | Semantic / Cross-Modal Purpose |
|---|---|---|---|---|---|
| 1 | `screenshot_portal_login.png` | 700×480 | Screenshot | Single Sign-On, Student ID field, password input, admission letter instructions | Links with IT onboarding credentials policy |
| 2 | `screenshot_wifi_setup.png` | 650×450 | Screenshot | `RAGNOVA-STUDENT` SSID, WPA2-Enterprise (802.1X PEAP), CA certificate validation | Direct cross-modal pair with `it_onboarding.docx` (secure Wi-Fi) |
| 3 | `screenshot_error_403.png` | 650×400 | Screenshot | 403 Forbidden alert, mass downloading restriction, Acceptable Use Policy §3.2 | Proves retrieval on policy infractions |
| 4 | `screenshot_synopsis_portal.png` | 700×500 | Screenshot | Synopsis portal, August 21st deadline, 3-page limit, 40%/35%/25% weighting | Direct cross-modal pair with `notice.pdf` |
| 5 | `screenshot_vpn_client.png` | 640×420 | Screenshot | SecureConnect VPN client, status Connected, digital library journal access | Matches remote journal access in `it_onboarding.docx` |
| 6 | `notice_seminar_poster.png` | 600×520 | Poster | Dept of CSE-AIML seminar, Multimodal RAG, Sept 28, Auditorium Hall B | Exercises visual layout and date/venue OCR |
| 7 | `notice_library_fines.png` | 620×460 | Notice | Undergrad borrowing limit 4 books / 14 days, fine 2 Rs/day, 200 Rs cap | Direct cross-modal pair with `library_hours.pdf` |
| 8 | `notice_lab_rules.png` | 620×440 | Notice | AIML lab rules, credential sharing prohibition, 2-week suspension | Matches Acceptable Use Policy in `it_onboarding.docx` |
| 9 | `notice_midterm_schedule.png` | 650×460 | Notice | Project timetable, 40% prototype, 35% report, 25% individual viva | Direct cross-modal pair with `notice.pdf` |
| 10 | `diagram_rag_architecture.png` | 720×520 | Diagram | System flow: PDF/Audio/Image -> PyMuPDF/Whisper/CLIP -> ChromaDB -> Ollama | Cross-modal technical architecture query |
| 11 | `diagram_campus_map.png` | 680×480 | Blueprint | Facility blueprint: Central Library (Bldg 3), IT Desk (Bldg 4), AIML (Bldg 2) | Tests spatial and facility entity queries |
| 12 | `photo_id_card_sample.png` | 600×380 | Photo Asset | Student ID card, Alex Morgan, Roll 23AIML042, barcode, valid thru 2027 | Tests photo ID card metadata extraction |
| 13 | `photo_lab_door_sign.png` | 620×390 | Photo Asset | Physical door plaque: AI Research Lab (Room 302), Dr. S. Rao, 9am-5pm | Tests physical sign / room location queries |
| 14 | `photo_library_desk_sign.png` | 620×390 | Photo Asset | Central Library circulation counter standing desk sign (4 books/14 days) | Tests desk policy and return drop box retrieval |
| 15 | `photo_campus_building_plaque.png` | 640×400 | Photo Asset | Academic Block B directory plaque (floors, departments, classrooms) | Tests building facility entity queries |

---

### File 3: `data/audio/*` [NEW DIRECTORY — 4 Files]
- **Directory Path**: [`data/audio/`](../data/audio/)
- **Target Met**: 4 spoken audio clips created via multi-backend speech synthesis with genuine speech (requirement: 3–5 audio clips, 15s to 3 min each).
- **Format**: Standard 16-bit PCM WAV, 16000 Hz, single channel mono (guaranteed host-independent via automated generator post-normalization across Windows, macOS, and Linux platforms, matching Whisper's native 16 kHz acoustic input).

| # | Filename | Size | Duration | Transcript Summary | Semantic Pairing |
|---|---|---|---|---|---|
| 1 | `hod_project_announcement.wav` | 1.06 MB | ~33s | Head of Dept briefing on 40% prototype demo, 35% project report, 25% individual viva, and August 21st deadline. | Cross-modal pair with `notice.pdf` and `notice_midterm_schedule.png` |
| 2 | `library_orientation_excerpt.wav` | 0.90 MB | ~28s | Central Library orientation on working hours (8am–10pm), undergrad borrowing quota (4 books for 14 days), and 2 Rs/day overdue fines. | Cross-modal pair with `library_hours.pdf` and `notice_library_fines.png` |
| 3 | `it_helpdesk_wifi_instructions.wav` | 1.10 MB | ~34s | Instructions on connecting to `RAGNOVA-STUDENT` with campus email, validating CA certificates, restarting adapters, and credential sharing prohibitions. | Cross-modal pair with `it_onboarding.docx` and `screenshot_wifi_setup.png` |
| 4 | `lab_assistant_briefing.wav` | 0.83 MB | ~26s | Briefing on installing the institute VPN client from the portal for journal access and acceptable network use. | Cross-modal pair with `it_onboarding.docx` and `screenshot_vpn_client.png` |

---

### File 4: `data/README.md` [MODIFIED]
- **File Path**: [`data/README.md`](../data/README.md)
- **Modifications**:
  1. **Deliberate Cross-Modal Checklist**: Marked `[x]` with explicit cross-modal citations linking images, audio clips, and documents:
     - Image & Document: `screenshot_wifi_setup.png` ↔ `it_onboarding.docx`
     - Audio & Document: `hod_project_announcement.wav` ↔ `notice.pdf`
     - Screenshot & Document: `screenshot_synopsis_portal.png` ↔ `notice.pdf`
  2. **File Inventory Table**: Replaced placeholder rows for `images/` and `audio/` with full metadata rows for all 15 images (including 4 photo assets) and 4 spoken audio clips.
  3. **Gold-Standard Evaluation Question Set**:
     - **Text queries**: Added T4 (VPN remote journals) and T5 (guide allotment announcement).
     - **Text → Image queries**: Added I1 (Wi-Fi authentication screenshot), I2 (RAGNova architecture diagram), I3 (seminar poster), I4 (AI lab door plaque).
     - **Image → Document queries**: Added M1 (`screenshot_synopsis_portal.png` ↔ `notice.pdf`), M2 (`screenshot_wifi_setup.png` ↔ `it_onboarding.docx`).
     - **Audio → Anything queries**: Added A1 (`hod_project_announcement.wav` ↔ `notice.pdf`), A2 (`library_orientation_excerpt.wav` ↔ `library_hours.pdf`).
     - **Negative controls**: Added N1 (mess menu inquiry) and N2 (fee refund inquiry) to verify refusal to hallucinate without false grounding.

---

### File 5: `src/app.py` [NEW]
- **File Path**: [`src/app.py`](../src/app.py)
- **Role**: Bare Streamlit UI scaffold satisfying Task 4.
- **Detailed Features**:
  1. **Page Settings & Styling**:
     - Modern dark-accent typography (`#1E293B`, `#64748B`).
     - Blue left-bordered citation cards (`#3B82F6`) with modality badges (`PDF`, `DOCX`, `IMAGE`, `AUDIO`).
  2. **Sidebar Metadata**:
     - Active settings read directly from `src.core.config.settings` (`OLLAMA_MODEL`, `TEXT_EMBEDDING_MODEL`, `CLIP_MODEL`, `WHISPER_MODEL_SIZE`, ChromaDB directory).
     - Dynamic corpus counter querying `data/documents/`, `data/images/`, and `data/audio/`.
     - Conversation reset button.
  3. **Chat Interface**:
     - Chat stream using `st.chat_message("user")` and `st.chat_message("assistant")`.
     - Bottom input using `st.chat_input()`.
     - Persistent chat history in `st.session_state.messages`.
  4. **Mock Processing Engine (`process_query`)**:
     - Returns grounded answers and citations for project marks, Wi-Fi connectivity (safe certificates), library rules, synopsis deadlines, lab locations, and ID cards.
     - Strictly honors negative controls (N1, N2) and unsupported queries by refusing without hallucination and returning empty citations (no false grounding).
     - Contains the exact integration anchor:
       ```python
       # =========================================================================
       # TODO: to wire the real call here
       # Once Track A / Chapter 10 implements retrieval and generation:
       #
       #   from src.pipelines.rag import answer_query
       #   response_text, retrieved_chunks = answer_query(user_query, top_k=settings.TOP_K)
       #   return response_text, [c.to_citation_dict() for c in retrieved_chunks]
       #
       # For now, return a mocked answer and fake citation string:
       # =========================================================================
       ```
  5. **Expandable Citation Cards**:
     - Renders expandable drawer (`st.expander("📚 Sources & Citations")`) showing source files, page numbers/timestamps, and grounded snippets.

---

### File 6: `tests/test_corpus_inventory.py` [NEW]
- **File Path**: [`tests/test_corpus_inventory.py`](../tests/test_corpus_inventory.py)
- **Role**: Automated verification test suite for Task 1 and Task 4 deliverables.
- **Test Cases**:
  1. `test_documents_corpus_present`: Verifies starter documents exist in `data/documents/`.
  2. `test_images_corpus_target_reached`: Verifies 10-15 images exist, >= 3 screenshots, >= 3 photo assets with visible text, all valid and readable by PIL.
  3. `test_audio_corpus_target_reached`: Verifies >= 3 audio files exist, file sizes > 1000 bytes, valid WAV headers, and duration >= 5.0 seconds.
  4. `test_app_scaffold_present_and_has_todo`: Verifies `src/app.py` exists, defines `process_query`, and includes `# TODO: to wire the real call here`.
  5. `test_app_process_query_mock_and_citations`: Tests `process_query` for marks queries, Wi-Fi queries, photo queries, negative controls refusal, and verified absence of false grounding citations.

---

## 3. Step-by-Step Execution Walkthrough

Follow these steps to reproduce or verify the work:

### Step 1: Generate the Corpus
```powershell
& ..\.venv\Scripts\python.exe scripts/generate_multimodal_corpus.py
```
*Expected Output*: 12 images written to `data/images/`, 4 WAV files written to `data/audio/`.

### Step 2: Run Verification Tests
```powershell
# Run the corpus inventory & UI scaffold test
& ..\.venv\Scripts\python.exe -m pytest tests/test_corpus_inventory.py -v

# Run the core contract test
& ..\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v

# Run document ingestion test (checks existing document pipeline)
& ..\.venv\Scripts\python.exe -m pytest tests/test_document_ingestion.py -v
```
*Expected Output*: All tests pass (100% pass rate, 0 errors).

### Step 3: Run the Streamlit UI
```powershell
& ..\.venv\Scripts\streamlit.exe run src/app.py
```
Open `http://localhost:8501` in your browser:
1. Type: `"How many marks does the working prototype carry?"` -> Verify assistant returns 40% prototype, 35% report, 25% viva, citing `notice.pdf` and `hod_project_announcement.wav`.
2. Type: `"How do I connect to campus wifi?"` -> Verify assistant returns `RAGNOVA-STUDENT` instructions, citing `it_onboarding.docx` and `screenshot_wifi_setup.png`.
3. Check the sidebar to verify document, image, and audio counts.
