"""
RAGNova — Streamlit UI Scaffold (Chapter 11)

A bare Streamlit shell for the offline multimodal RAG system.
Features:
  - Text query input and submit handler
  - Mocked answer and citation response
  - Clearly marked '# TODO: to wire the real call here' for Chapter 10 integration
  - System status and corpus overview in the sidebar

Run with:  streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so src imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.core.config import settings

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
# Sidebar: System Metadata & Corpus Stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🌌 RAGNova System")
    st.caption("Offline Multimodal RAG (B.Tech CSE-AIML)")
    st.divider()

    st.subheader("⚙️ Active Configuration")
    st.write(f"**Local LLM:** `{settings.OLLAMA_MODEL}`")
    st.write(f"**Text Embedder:** `{settings.TEXT_EMBEDDING_MODEL}`")
    st.write(f"**Image Embedder:** `{settings.CLIP_MODEL}`")
    st.write(f"**Speech-to-Text:** `faster-whisper ({settings.WHISPER_MODEL_SIZE})`")
    st.write(f"**ChromaDB Dir:** `{settings.CHROMA_PERSIST_DIR}`")

    st.divider()
    st.subheader("📁 Corpus Overview")
    docs_dir = PROJECT_ROOT / "data" / "documents"
    images_dir = PROJECT_ROOT / "data" / "images"
    audio_dir = PROJECT_ROOT / "data" / "audio"

    n_docs = len(list(docs_dir.glob("*.*"))) if docs_dir.exists() else 0
    n_images = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
    n_audio = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0

    st.write(f"📄 **Documents:** {n_docs} files")
    st.write(f"🖼️ **Images:** {n_images} files")
    st.write(f"🎙️ **Audio Clips:** {n_audio} files")

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🌌 RAGNova Unified Query Interface</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions in plain language across your documents, screenshots, and audio recordings — 100% offline.</div>',
    unsafe_allow_html=True,
)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am **RAGNova**, your offline multimodal assistant. "
                "You can ask me questions about campus policies, project deadlines, Wi-Fi configuration, "
                "or library hours."
            ),
            "citations": None,
        }
    ]

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📚 Sources & Citations", expanded=False):
                for cit in msg["citations"]:
                    st.markdown(
                        f"""
                        <div class="citation-box">
                            <span class="badge-pill">{cit['modality'].upper()}</span>
                            <strong>[{cit['id']}]</strong> <code>{cit['source']}</code> {cit.get('location', '')}<br>
                            <em>"{cit['snippet']}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Query Processing Function (Mock / Scaffold)
# ---------------------------------------------------------------------------
def process_query(user_query: str) -> tuple[str, list[dict]]:
    """Process a user query and return (answer_text, citations_list).

    Currently returns a mocked answer with grounded citations for supported topics,
    and explicitly refuses without citations for unsupported queries and negative controls (Task 4 / O4).
    Once Chapter 10 (RAG Core) is implemented, the real retrieval and LLM call
    will replace the mock below.
    """
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
# Query Input Handler
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Type your question here (e.g., 'How many marks does the prototype carry?')...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt, "citations": None})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving offline multimodal sources..."):
            answer, citations = process_query(user_prompt)
            st.markdown(answer)
            if citations:
                with st.expander("📚 Sources & Citations", expanded=True):
                    for cit in citations:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <span class="badge-pill">{cit['modality'].upper()}</span>
                                <strong>[{cit['id']}]</strong> <code>{cit['source']}</code> {cit.get('location', '')}<br>
                                <em>"{cit['snippet']}"</em>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
