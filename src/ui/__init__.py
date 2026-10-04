"""Chapter 11 — the non-drawing half of the chat interface (Track C).

The Streamlit page itself is src/app.py (PR #4's scaffold). Everything the
page needs that isn't drawing lives here as plain functions, so it can be
tested with pytest without starting Streamlit, and so wiring the
scaffold's "TODO: to wire the real call here" is a small diff:

    from src.pipelines.rag import stream_answer
    from src.ui.citations import citation_views
    from src.ui.feedback import FeedbackEntry, record_feedback

  - backend.py   : save + index uploads, voice -> text, image -> OCR text,
                   index counts, is Ollama up
  - citations.py : what to show for each [n] (label, excerpt, file, caveats)
  - feedback.py  : the 1-5 rating log Chapter 12's human testing writes to
"""
