"""
RAGNova — Streamlit chat UI (Chapter 11)

A unified multimodal Streamlit interface for the offline multimodal RAG system.
Features:
  - Text, voice (faster-whisper), and image (CLIP + OCR) query processing
  - Live token streaming via stream_answer() and st.write_stream()
  - Expandable, rich citations adhering to ADR-003, ADR-007, and ADR-008
  - Human feedback collection loop (Chapter 12) logging to feedback.jsonl
  - Live knowledge base upload and indexing for PDF, DOCX, Images, and Audio
  - Canned demo answers: opt-in only, always labelled, logged as "scaffold-mock"
  - System status and corpus overview in the sidebar

If the live system cannot answer (Ollama down, the model will not start, the
index is unreadable) the chat says so and shows the real error. It never swaps
in a canned answer: a tester would rate it as if the system had written it.

Run with:  streamlit run src/app.py
from the project root, so that .streamlit/config.toml applies (localhost only,
no usage telemetry).
"""

from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so src imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from src.core.config import settings
from src.pipelines.rag.answer import check_citations, extract_citation_numbers, is_refusal, search_question, stream_answer
from src.pipelines.rag.attachments import image_attachment
from src.ui.backend import (
    AUDIO_EXTS,
    DOCUMENT_EXTS,
    IMAGE_EXTS,
    add_to_corpus,
    attachment_note,
    check_model_generates,
    index_counts,
    kind_of,
    list_sources,
    ocr_available,
    ocr_image,
    ollama_status,
    transcribe_audio_bytes,
)
from src.ui.citations import CitationView, cited_views, citation_views
from src.ui.feedback import FeedbackEntry, model_label, record_feedback

ENGINE_LIVE = "Live Multimodal RAG"
ENGINE_CANNED = "Canned demo answers (not the real system)"
CANNED_BANNER = (
    "⚠️ Canned demo answer: hand-written text in src/app.py, not retrieved from "
    "the index and not written by the model. Do not rate it as a RAGNova answer."
)
SOURCE_ICONS = {"document": "📄", "image": "🖼️", "audio": "🎙️"}
GREETING = (
    "Hello! I am **RAGNova**, your offline multimodal assistant. "
    "You can ask me questions about campus policies, project deadlines, Wi-Fi configuration, "
    "or library hours. To ask about your own files, add them in the sidebar: the page "
    "then answers from that file alone, and **Answer from** switches back to everything."
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAGNova — Offline Multimodal RAG",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        margin-top: 0.8rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #E2E8F0;
        color: #334155;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def new_message(role: str, **fields) -> dict:
    """One chat-history entry. `engine` is "live" for the real pipeline and
    "scaffold" for canned demo text; `error` is a failure the user must see,
    and is the reason an answer is not offered for rating."""
    return {
        "role": role,
        "content": "",
        "citations": None,
        "views": None,
        "warning": None,
        "error": None,
        "engine": None,
        "image_note": None,
        **fields,
    }


def render_views(views: list[CitationView], expanded: bool = False) -> None:
    """The citation list for a real answer (one CitationView per chunk)."""
    with st.expander(f"📚 Sources & Citations ({len(views)})", expanded=expanded):
        for view in views:
            st.markdown(
                f"""
                <div class="citation-box">
                    <span class="badge-pill">{html.escape(view.modality_label.upper())}</span>
                    <strong>{html.escape(view.title)}</strong><br>
                    <div style="margin-top: 0.3rem;"><em>"{html.escape(view.excerpt)}"</em></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if view.note:
                st.caption(f"ℹ️ {view.note}")
            if view.modality_label == "Audio" and view.audio_start_s is not None and view.file_exists:
                st.audio(view.file_path, start_time=int(view.audio_start_s))
            elif view.modality_label == "Image" and view.file_exists:
                try:
                    st.image(view.file_path, width=320, caption=view.title)
                except Exception:
                    pass


def render_canned_citations(citations: list[dict], expanded: bool = False) -> None:
    """The hand-written citations that go with a canned demo answer."""
    with st.expander(f"📚 Sources & Citations ({len(citations)})", expanded=expanded):
        for cit in citations:
            st.markdown(
                f"""
                <div class="citation-box">
                    <span class="badge-pill">{html.escape(cit['modality'].upper())}</span>
                    <strong>[{cit['id']}]</strong> <code>{html.escape(cit['source'])}</code> {html.escape(cit.get('location', ''))}<br>
                    <em>"{html.escape(cit['snippet'])}"</em>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar: System Metadata, Ingestion & Corpus Stats
# ---------------------------------------------------------------------------
st.session_state.setdefault("kb_nonce", 0)            # changing an uploader's key empties it
st.session_state.setdefault("query_img_nonce", 0)

with st.sidebar:
    st.title("🌌 RAGNova System")
    st.caption("Offline Multimodal RAG (B.Tech CSE-AIML)")
    st.divider()

    st.subheader("⚙️ System Status")
    ollama_state, ollama_detail = ollama_status()
    if ollama_state == "ready":
        st.success(f"🟢 Ollama reachable, `{settings.OLLAMA_MODEL}` pulled")
        st.caption("That does not prove the model can start. The test button below does.")
    elif ollama_state == "missing_model":
        st.warning(f"⚠️ Ollama is running but `{settings.OLLAMA_MODEL}` is not pulled")
        st.caption(f"Run `ollama pull {settings.OLLAMA_MODEL}`.")
    else:
        st.warning("⚠️ Ollama is not reachable")
        st.caption("Run `ollama serve`. Answers need the model; questions the corpus cannot answer are still refused.")

    if st.button("🔬 Test the model", key="btn_model_check"):
        with st.spinner(f"Asking {settings.OLLAMA_MODEL} for a few tokens..."):
            worked, detail = check_model_generates()
        st.session_state.model_check = (worked, detail, datetime.now().strftime("%H:%M:%S"))
    model_check = st.session_state.get("model_check")
    if model_check:
        worked, detail, when = model_check
        (st.success if worked else st.error)(f"{when}: {detail}")

    st.write(f"**Text Embedder:** `{settings.TEXT_EMBEDDING_MODEL}`")
    st.write(f"**Image Embedder:** `{settings.CLIP_MODEL}`")
    st.write(f"**Speech-to-Text:** `faster-whisper ({settings.WHISPER_MODEL_SIZE})`")
    st.write(f"**ChromaDB Dir:** `{settings.CHROMA_PERSIST_DIR}`")

    st.divider()
    st.subheader("📁 Corpus Overview")
    docs_dir = PROJECT_ROOT / "data" / "documents"
    images_dir = PROJECT_ROOT / "data" / "images"
    audio_dir = PROJECT_ROOT / "data" / "audio"

    n_docs = len([p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in DOCUMENT_EXTS]) if docs_dir.exists() else 0
    n_images = len([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if images_dir.exists() else 0
    n_audio = len([p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]) if audio_dir.exists() else 0

    st.write(f"📄 **Documents:** {n_docs} files")
    st.write(f"🖼️ **Images:** {n_images} files")
    st.write(f"🎙️ **Audio Clips:** {n_audio} files")

    try:
        counts = index_counts()
        st.write(f"📊 **Vector Index:** {counts['text']} text/audio, {counts['image']} images")
    except Exception as err:
        st.warning(f"⚠️ Could not read the vector index: {err}")

    st.divider()
    st.subheader("📥 Add to Corpus")
    # The result is kept in session_state and shown on the next run: a message
    # drawn right before st.rerun() is gone before anyone can read it.
    kb_flash = st.session_state.pop("kb_flash", None)
    if kb_flash:
        (st.success if kb_flash[0] else st.error)(kb_flash[1])
    all_allowed = sorted([ext.lstrip(".") for ext in (DOCUMENT_EXTS | IMAGE_EXTS | AUDIO_EXTS)])
    uploaded_kb_file = st.file_uploader(
        "Upload document, image or audio",
        type=all_allowed,
        key=f"kb_uploader_{st.session_state.kb_nonce}",
    )
    if uploaded_kb_file is not None and st.button("Save & Index File", key="btn_index_upload"):
        with st.spinner(f"Indexing {uploaded_kb_file.name}..."):
            try:
                result = add_to_corpus(
                    uploaded_kb_file.name,
                    uploaded_kb_file.getvalue(),
                    data_root=PROJECT_ROOT / "data",
                )
                if result.focus:
                    st.session_state.focus_sources = result.focus
                st.session_state.kb_flash = (result.ok, result.message)
                st.session_state.kb_nonce += 1       # empty the uploader so it cannot be indexed twice by accident
            except Exception as err:
                st.session_state.kb_flash = (False, f"Failed to index file: {err}")
        st.rerun()

    st.divider()
    st.subheader("🎯 Answer from")
    try:
        listed_sources = list_sources()
    except Exception as err:
        listed_sources = []
        st.warning(f"⚠️ Could not list the indexed files: {err}")
    # A stored pick that is no longer in the index (a file removed, an index rebuilt) would crash the widget.
    st.session_state["focus_sources"] = [s for s in st.session_state.get("focus_sources", []) if s in listed_sources]
    st.multiselect(
        "Files to answer from",
        options=listed_sources,
        key="focus_sources",
        format_func=lambda path: f"{SOURCE_ICONS.get(kind_of(path), '📄')} {Path(path).name}",
        placeholder="Everything indexed",
        label_visibility="collapsed",
    )
    st.caption(
        "Nothing picked: all indexed files, with the relevance checks that refuse questions the files cannot answer. "
        "Files picked: only those files, with no relevance check; the model still says when they do not contain the answer."
    )

    st.divider()
    st.subheader("🛠️ Engine Mode")
    engine_mode = st.radio(
        "Response Engine",
        options=[ENGINE_LIVE, ENGINE_CANNED],
        index=0,
        help="Live RAG queries ChromaDB and streams Ollama. Canned demo answers are hand-written text for demos with no model; every one is labelled.",
    )
    if engine_mode == ENGINE_CANNED:
        st.warning("Canned mode: the answers are not produced by the system. Image queries are ignored.")

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = [new_message("assistant", content=GREETING)]
        st.session_state.feedback_submitted = set()
        st.rerun()

# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🌌 RAGNova Unified Query Interface</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions in plain language across your documents, screenshots, and audio recordings — 100% offline.</div>',
    unsafe_allow_html=True,
)

def clear_focus() -> None:
    # A callback, not code after the button: Streamlit forbids changing a widget's value
    # once it has been drawn in this run, and the picker in the sidebar is drawn first.
    st.session_state["focus_sources"] = []


focus = list(st.session_state.get("focus_sources", []))
if focus:
    banner, clear = st.columns([5, 1])
    banner.info("📌 Answering from **" + ", ".join(Path(s).name for s in focus) + "** only.")
    clear.button("Use everything", key="btn_clear_focus", on_click=clear_focus)

# Initialize conversation history and state
if "messages" not in st.session_state:
    st.session_state.messages = [new_message("assistant", content=GREETING)]

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()

if "tester_name" not in st.session_state:
    st.session_state.tester_name = ""


# Display existing messages
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("image_note"):
            st.caption(msg["image_note"])
        if msg.get("engine") == "scaffold":
            st.warning(CANNED_BANNER)
        if msg.get("error"):
            st.error(msg["error"])
        if msg.get("warning"):
            st.warning(msg["warning"])

        if msg.get("views"):
            render_views(msg["views"])
        elif msg.get("citations"):
            render_canned_citations(msg["citations"])

        # Human Feedback Form (Chapter 12). Not offered for a failure: there is no answer to rate.
        if msg["role"] == "assistant" and msg_idx > 0 and not msg.get("error"):
            fb_key = f"fb_{msg_idx}"
            if fb_key not in st.session_state.feedback_submitted:
                with st.expander("⭐ Rate this answer (Feedback Form)", expanded=False):
                    with st.form(key=f"form_{fb_key}"):
                        f_col1, f_col2 = st.columns([1, 2])
                        with f_col1:
                            rating = st.select_slider("Rating (1-5)", options=[1, 2, 3, 4, 5], value=5)
                            tester = st.text_input("Tester Name / Initials", value=st.session_state.tester_name)
                        with f_col2:
                            comment = st.text_input("Comment (optional)", placeholder="How helpful and accurate was this answer?")
                        if st.form_submit_button("Submit Rating"):
                            sources = [v.title for v in msg.get("views", [])] if msg.get("views") else [c.get("source", "") for c in msg.get("citations") or []]
                            user_q = st.session_state.messages[msg_idx - 1]["content"] if msg_idx > 0 else "N/A"
                            entry = FeedbackEntry(
                                query=user_q,
                                answer=msg["content"],
                                rating=rating,
                                sources=sources,
                                comment=comment,
                                tester=tester,
                                model=model_label(msg.get("engine")),
                            )
                            record_feedback(entry)
                            st.session_state.feedback_submitted.add(fb_key)
                            st.session_state.tester_name = tester
                            st.rerun()
            else:
                st.caption("✅ Feedback submitted for this response.")


# ---------------------------------------------------------------------------
# Canned demo answers (opt-in only; retained for demos with no model and for benchmark comparison)
# ---------------------------------------------------------------------------
def process_query(user_query: str) -> tuple[str, list[dict]]:
    """Canned demo answer for `user_query`: (answer_text, citations_list).

    NOT the RAG system. Keyword-matched, hand-written text with hand-written
    citations, kept for demos with no model and as a benchmark comparison.
    The chat calls it only when the user explicitly picks the canned engine,
    and labels every answer produced this way.
    """
    q = user_query.lower()

    # Negative controls (P2 fix): explicitly refuse without hallucinating or claiming false grounding
    if any(k in q for k in ["mess", "lunch", "dinner", "meal", "menu", "refund", "tuition fee", "fee refund"]):
        mock_answer = (
            "I could not find information about that in the provided offline sources (documents, images, or audio recordings). "
            "The indexed campus corpus does not cover hostel mess menus or tuition fee refund procedures. "
            "Under RAGNova Objective O4, the system refuses to hallucinate facts absent from the corpus."
        )
        return mock_answer, []

    # 1. Project Evaluation Weighting & Prototype Marks (Gold-standard T1, A1)
    is_proto_marks = (
        ("prototype" in q and any(k in q for k in ["mark", "demo", "percent", "weight", "score", "carry", "viva"]))
        or ("viva" in q and any(k in q for k in ["voce", "mark", "percent", "weight", "person", "eval"]))
        or ("evaluation" in q and any(k in q for k in ["weight", "mark", "scheme", "prototype", "breakdown"]))
        or ("how many marks" in q)
    )
    if is_proto_marks:
        mock_answer = (
            "According to the department notice and presentation timetable, the **working prototype** "
            "demonstrated on evaluation day carries **40% of the total marks** [1]. The written project "
            "report carries **35%**, and the individual **viva voce** carries the remaining **25%** [1][2]. "
            "Every team member must be present in person for the viva."
        )
        fake_citations = [
            {
                "id": 1,
                "modality": "pdf",
                "source": "data/documents/notice.pdf",
                "location": "Page 2",
                "snippet": "The working prototype demonstrated on the day of evaluation will carry forty percent of the total marks...",
            },
            {
                "id": 2,
                "modality": "audio",
                "source": "data/audio/hod_project_announcement.wav",
                "location": "Timestamp 00:08 - 00:22",
                "snippet": "Please remember that the working prototype carries forty percent... viva voce carries twenty-five percent...",
            },
        ]
        return mock_answer, fake_citations

    # 2. Campus Wi-Fi Configuration (Gold-standard T3, I1, M2)
    # Avoid matching generic words like 'network' or 'connect' in isolation (e.g. 'neural network')
    is_wifi = (
        ("wifi" in q or "wi-fi" in q or "ssid" in q or "ragnova-student" in q)
        or ("wireless" in q and any(k in q for k in ["adapter", "network", "connect", "setup", "setting", "lan"]))
        or ("connect" in q and any(k in q for k in ["campus", "internet", "wireless", "wifi", "wi-fi"]))
    )
    if is_wifi:
        mock_answer = (
            "To connect to campus Wi-Fi, select the SSID **`RAGNOVA-STUDENT`** [1]. Use your institute email "
            "address and the password set during account activation. The network uses WPA2-Enterprise (802.1X PEAP) "
            "with mandatory CA certificate validation (`ragnova.edu`) [2]. "
            "Never select 'Do not validate' for certificates to avoid credential theft attacks. "
            "If your device fails to connect, restart your wireless adapter before visiting the IT help desk."
        )
        fake_citations = [
            {
                "id": 1,
                "modality": "docx",
                "source": "data/documents/it_onboarding.docx",
                "location": "Page 1",
                "snippet": "Wireless internet on campus is available through the network named RAGNOVA-STUDENT...",
            },
            {
                "id": 2,
                "modality": "image",
                "source": "data/images/screenshot_wifi_setup.png",
                "location": "Dialog Window",
                "snippet": "Security Protocol: WPA2-Enterprise (802.1X PEAP), CA Certificate: Use system certificates (Domain: ragnova.edu)",
            },
        ]
        return mock_answer, fake_citations

    # 3. Library Borrowing, Quotas & Overdue Fines (Gold-standard T2, A2)
    # Avoid matching 'book' alone (e.g. 'book the AI lab') or 'fine' alone (e.g. 'fine-tuning')
    is_library = (
        ("library" in q)
        or ("borrow" in q)
        or ("overdue" in q)
        or ("fine" in q and any(k in q for k in ["overdue", "late", "rupee", "library", "return", "cap", "title"]))
        or ("book" in q and any(k in q for k in ["borrow", "return", "quota", "checkout", "overdue", "library", "title", "renew"]))
    )
    if is_library:
        mock_answer = (
            "Undergraduate students can borrow up to **4 books for 14 days** [1][2]. Overdue fines are **Rs 2 per day "
            "per title**, capped at a maximum of **Rs 200** [1][3]. The library is open from 8:00 AM to 10:00 PM on "
            "working days, with an after-hours return drop box beside the ground floor exit [2]."
        )
        fake_citations = [
            {
                "id": 1,
                "modality": "pdf",
                "source": "data/documents/library_hours.pdf",
                "location": "Page 1",
                "snippet": "Undergraduate students may borrow up to four books at a time for a period of fourteen days...",
            },
            {
                "id": 2,
                "modality": "image",
                "source": "data/images/photo_library_desk_sign.png",
                "location": "Desk Sign",
                "snippet": "Circulation Desk: Undergraduate Quota 4 books for 14 days, After-Hours Drop Box at Ground Floor Exit",
            },
            {
                "id": 3,
                "modality": "audio",
                "source": "data/audio/library_orientation_excerpt.wav",
                "location": "Timestamp 00:15 - 00:30",
                "snippet": "Overdue fines are two rupees per day, capped at two hundred rupees per title.",
            },
        ]
        return mock_answer, fake_citations

    # 4. Project Synopsis Submission (Gold-standard T5, M1)
    is_synopsis = (
        ("synopsis" in q)
        or ("submission deadline" in q and any(k in q for k in ["project", "august", "final year"]))
        or ("21st august" in q and "deadline" in q)
    )
    if is_synopsis:
        mock_answer = (
            "The submission deadline for the project synopsis is **21st August** [1]. The document should not exceed "
            "3 pages excluding the cover page and references, and must be submitted as a single PDF via the department "
            "portal [1][2]. Teams must comprise 2 or 3 members."
        )
        fake_citations = [
            {
                "id": 1,
                "modality": "pdf",
                "source": "data/documents/notice.pdf",
                "location": "Page 1",
                "snippet": "The submission deadline for the project synopsis is 21st August. The synopsis document should not exceed three pages...",
            },
            {
                "id": 2,
                "modality": "image",
                "source": "data/images/screenshot_synopsis_portal.png",
                "location": "Upload Portal",
                "snippet": "Submission Deadline: 21st August (Strict deadline), Document format: Single PDF file",
            },
        ]
        return mock_answer, fake_citations

    # 5. RAGNova Architecture & System Pipeline Overview
    # Avoid matching 'system' in isolation (e.g. 'operating system', 'database system')
    is_architecture = (
        ("what is this project" in q)
        or ("what is ragnova" in q)
        or ("tell me about ragnova" in q)
        or ("rag architecture" in q)
        or ("ragnova architecture" in q)
        or ("pipeline architecture" in q)
        or ("vector stores" in q or "chromadb collections" in q)
    )
    if is_architecture:
        mock_answer = (
            "**RAGNova** is an offline multimodal Retrieval-Augmented Generation system designed for campus environments [1]. "
            "It indexes documents (PDF/DOCX), images (screenshots and physical photo plaques via OCR and OpenCLIP), "
            "and audio briefings (via faster-whisper speech transcription) into dual ChromaDB vector collections [1][2]."
        )
        fake_citations = [
            {
                "id": 1,
                "modality": "image",
                "source": "data/images/diagram_rag_architecture.png",
                "location": "System Diagram",
                "snippet": "RAGNova Offline Multimodal Architecture: Documents, Images, and Audio pipelines with dual ChromaDB stores.",
            },
            {
                "id": 2,
                "modality": "pdf",
                "source": "data/documents/notice.pdf",
                "location": "Page 1",
                "snippet": "Department of Computer Science and Engineering (Artificial Intelligence and Machine Learning)...",
            },
        ]
        return mock_answer, fake_citations

    # 6. AIML Research Lab Location Plaque (Gold-standard I4)
    # Asking about location / Room 302 / faculty in-charge. (Booking the lab is not supported and will fall through).
    is_lab_location = (
        ("room 302" in q)
        or (
            any(k in q for k in ["ai lab", "aiml lab", "research lab", "computing lab"])
            and any(k in q for k in ["where", "locate", "in-charge", "faculty", "door", "sign", "plaque", "hours", "superintendent", "who heads"])
        )
    )
    if is_lab_location:
        mock_answer = (
            "The AI & Machine Learning Research Laboratory is located in **Academic Block B, Room 302** [1]. "
            "The faculty in-charge is **Dr. S. Rao**. Operating hours are 9:00 AM to 5:00 PM, and a smart card ID badge "
            "is required for entry [1]."
        )
        fake_citations = [
            {
                "id": 1,
                "modality": "image",
                "source": "data/images/photo_lab_door_sign.png",
                "location": "Room 302 Door Plaque",
                "snippet": "AI & Machine Learning Research Laboratory, Location: Academic Block B — Room 302, Faculty In-Charge: Dr. S. Rao",
            },
        ]
        return mock_answer, fake_citations

    # Catch-all for unsupported / unindexed queries:
    # Crucial fix for P2 (False Grounding): Do NOT claim false grounding or emit fake citations!
    mock_answer = (
        f"I could not find sufficient information in the indexed corpus to answer: \"{user_query}\".\n\n"
        "The offline knowledge base contains academic policies, library regulations, IT network setup, "
        "project evaluation guidelines, and campus facilities. Please query one of these indexed topics."
    )
    fake_citations = []
    return mock_answer, fake_citations


# ---------------------------------------------------------------------------
# Multimodal Query Controls
# ---------------------------------------------------------------------------
col_voice, col_img = st.columns([1, 1])

with col_voice:
    with st.expander("🎙️ Spoken / Voice Query", expanded=False):
        voice_clip = st.audio_input("Record question with microphone", key="mic_query_input")
        if voice_clip is not None:
            if st.button("Transcribe & Submit Audio", key="btn_transcribe_audio"):
                with st.spinner("Transcribing speech with faster-whisper..."):
                    try:
                        transcribed_text = transcribe_audio_bytes(voice_clip.getvalue(), suffix=".wav")
                        if transcribed_text.strip():
                            st.session_state["active_prompt_override"] = transcribed_text.strip()
                            st.rerun()
                        else:
                            st.warning("No intelligible speech detected in audio clip.")
                    except Exception as err:
                        st.error(f"Speech transcription failed: {err}")

query_img_file = None
with col_img:
    with st.expander("🖼️ Visual / Query Image", expanded=False):
        query_img_file = st.file_uploader(
            "Attach query image (searches via OCR + OpenCLIP)",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"query_img_uploader_{st.session_state.query_img_nonce}",
        )
    if query_img_file is not None:
        # Visible even with the expander closed, and cleared after one question:
        # an image that quietly stays attached rewrites every later question.
        st.info("📎 Query image attached. It is used for your next question only.")

# One-click questions for the files picked above (the "what is in this?" first move).
if focus:
    these = "this file" if len(focus) == 1 else "these files"
    suggestions = [
        f"Give me a short summary of {these}.",
        "What are the key points?",
        "List the names, dates and numbers it mentions." if len(focus) == 1
        else "List the names, dates and numbers they mention.",
    ]
    for column, suggestion in zip(st.columns(len(suggestions)), suggestions):
        if column.button(suggestion, key=f"suggest_{suggestion}"):
            st.session_state["active_prompt_override"] = suggestion

# Chat input
user_prompt = st.chat_input("Type your question here (e.g., 'How many marks does the prototype carry?')...")

# Check if there is an audio override prompt from the mic button
if st.session_state.get("active_prompt_override"):
    user_prompt = st.session_state.pop("active_prompt_override")


# ---------------------------------------------------------------------------
# Query Execution & Response Streaming
# ---------------------------------------------------------------------------
if user_prompt:
    query_pil = None
    attachments = []
    image_note = None
    if query_img_file is not None:
        if engine_mode == ENGINE_CANNED:
            image_note = "📎 Query image ignored: canned demo mode does not search."
        else:
            try:
                query_pil = Image.open(query_img_file)
                ocr_text = ocr_image(query_pil)
                attachments = [image_attachment(query_img_file.name, ocr_text)]
                image_note = attachment_note(query_img_file.name, ocr_text, ocr_available())
            except Exception as exc:
                query_pil = None
                attachments = []
                image_note = f"⚠️ Could not use the attached image ({exc}); the question was run without it."

    # The last answered exchange, for follow-ups ("tell me more"): (what that turn searched with, its answer).
    # Read before this question is appended; a turn that failed or was refused gives no history.
    history = None
    past = st.session_state.messages
    if len(past) >= 2 and past[-1]["role"] == "assistant" and past[-2]["role"] == "user":
        last_answer = past[-1]
        if last_answer.get("engine") == "live" and not last_answer.get("error") \
                and not is_refusal(last_answer["content"]):
            history = [(past[-2].get("search") or past[-2]["content"], last_answer["content"])]

    st.session_state.messages.append(new_message(
        "user", content=user_prompt, image_note=image_note,
        search=search_question(user_prompt, history, attachments, focus or None),
    ))
    with st.chat_message("user"):
        st.markdown(user_prompt)
        if image_note:
            st.caption(image_note)

    reply: dict = {}
    with st.chat_message("assistant"):
        if engine_mode == ENGINE_CANNED:
            answer_text, canned_citations = process_query(user_prompt)
            st.markdown(answer_text)
            reply = {"content": answer_text, "citations": canned_citations or None, "engine": "scaffold"}
        else:
            with st.spinner("Retrieving offline multimodal sources & streaming answer..."):
                try:
                    retrieved_chunks, token_stream = stream_answer(
                        user_prompt,
                        top_k=settings.TOP_K,
                        include_images=True,
                        query_image=query_pil,
                        sources=focus or None,
                        attachments=attachments,
                        history=history,
                    )
                    answer_text = st.write_stream(token_stream)

                    # Check citations for hallucination (Chapter 11 §1.3)
                    out_of_range = check_citations(answer_text, retrieved_chunks, user_prompt)
                    warning_msg = None
                    if out_of_range:
                        warning_msg = f"⚠️ Citation Alert: The model referenced source index {sorted(out_of_range)}, which was not in the retrieved context."
                    reply = {
                        "content": answer_text,
                        # A refusal lists no sources: with a scope or an attachment chunks are retrieved even
                        # when the model finds the answer is not in them, and "Sources" under a refusal reads
                        # as if the refusal were grounded in them.
                        # Otherwise only the sources the answer cites: a retrieved chunk the model did not use (an
                        # algorithms chapter under "tell me about internship fair") listed as [1] reads as a source.
                        "views": None if is_refusal(answer_text) else cited_views(
                            citation_views(retrieved_chunks), extract_citation_numbers(answer_text)
                        ),
                        "warning": warning_msg,
                        "engine": "live",
                    }
                except Exception as exc:
                    # No canned answer is substituted: it would be rated, logged and read as the system's.
                    reply = {
                        "content": "⚠️ The live system could not answer this question. No canned answer was substituted.",
                        "error": f"{type(exc).__name__}: {exc}",
                        "engine": "live",
                    }

    st.session_state.messages.append(new_message("assistant", **reply))
    if query_img_file is not None:
        st.session_state.query_img_nonce += 1       # the image is for this one question
    st.rerun()
