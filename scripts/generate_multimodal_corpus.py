"""
Generate the multimodal corpus for RAGNova:
  - data/images/: 12 images (screenshots, notice posters, diagrams, ID card)
  - data/audio/: 4 spoken audio clips (.wav) matching the campus corpus topics

Run with:  python scripts/generate_multimodal_corpus.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import wave
import math
import struct

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"


def _get_font(size: int = 16, bold: bool = False):
    font_names = ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "calibri.ttf"]
    if bold:
        font_names = ["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "calibrib.ttf"] + font_names
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def create_image(
    filename: str,
    width: int,
    height: int,
    bg_color: tuple[int, int, int],
    title: str,
    sections: list[tuple[str, str | list[str]]],
    border_color: tuple[int, int, int] | None = None,
    badge: str | None = None,
):
    """Render a clean, high-contrast image containing visible text suitable for OCR and CLIP."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    header_font = _get_font(16, bold=True)
    body_font = _get_font(14, bold=False)
    small_font = _get_font(12, bold=False)

    # Optional border
    if border_color:
        draw.rectangle([(0, 0), (width - 1, height - 1)], outline=border_color, width=2)

    # Header bar
    header_bg = (30, 41, 59) if bg_color != (30, 41, 59) else (15, 23, 42)
    draw.rectangle([(0, 0), (width, 42)], fill=header_bg)

    # Window control dots
    draw.ellipse([(12, 16), (22, 26)], fill=(239, 68, 68))
    draw.ellipse([(28, 16), (38, 26)], fill=(245, 158, 11))
    draw.ellipse([(44, 16), (54, 26)], fill=(16, 185, 129))

    # Header title text
    draw.text((68, 12), title, font=header_font, fill=(255, 255, 255))

    if badge:
        b_w = len(badge) * 8 + 16
        draw.rectangle([(width - b_w - 15, 10), (width - 15, 32)], fill=(59, 130, 246))
        draw.text((width - b_w - 7, 13), badge, font=small_font, fill=(255, 255, 255))

    y = 60
    margin_x = 30

    for sec_title, sec_content in sections:
        if sec_title:
            draw.text((margin_x, y), sec_title, font=header_font, fill=(30, 58, 138) if bg_color[0] > 200 else (147, 197, 253))
            y += 26

        if isinstance(sec_content, str):
            lines = sec_content.split("\n")
        else:
            lines = sec_content

        for line in lines:
            if y > height - 30:
                break
            text_color = (15, 23, 42) if bg_color[0] > 200 else (226, 232, 240)
            draw.text((margin_x + 10, y), line, font=body_font, fill=text_color)
            y += 22
        y += 12

    out_path = IMAGES_DIR / filename
    img.save(out_path, format="PNG")
    print(f"wrote image: {out_path.name} ({width}x{height})")


def create_photo_image(
    filename: str,
    width: int,
    height: int,
    bg_color: tuple[int, int, int],
    title: str,
    sections: list[tuple[str, str | list[str]]],
    border_color: tuple[int, int, int] | None = None,
    badge: str | None = None,
    photo_type: str = "PLAQUE",  # "ID_CARD", "PLAQUE", "SIGN"
):
    """Render a realistic physical photo asset (sign, plaque, or ID card) containing visible text."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    header_font = _get_font(18, bold=True)
    subhead_font = _get_font(14, bold=True)
    body_font = _get_font(13, bold=False)
    small_font = _get_font(11, bold=False)

    # Physical frame / outer matte border (photo asset, NOT an OS window)
    frame_width = 8
    frame_color = border_color if border_color else (180, 180, 180)
    for i in range(frame_width):
        draw.rectangle([(i, i), (width - 1 - i, height - 1 - i)], outline=frame_color)

    # Inner plaque margin
    pad = 20
    draw.rectangle([(pad, pad), (width - pad, height - pad)], outline=(148, 163, 184), width=1)

    # Top banner / institution bar
    banner_h = 42
    draw.rectangle([(pad + 1, pad + 1), (width - pad - 1, pad + banner_h)], fill=(30, 41, 59))
    draw.text((pad + 16, pad + 12), title, font=header_font, fill=(255, 255, 255))

    if badge:
        b_w = len(badge) * 8 + 16
        badge_bg = (225, 29, 72) if photo_type == "SIGN" else ((16, 185, 129) if photo_type == "ID_CARD" else (37, 99, 235))
        draw.rectangle([(width - pad - b_w - 12, pad + 9), (width - pad - 12, pad + 33)], fill=badge_bg)
        draw.text((width - pad - b_w - 4, pad + 13), badge, font=small_font, fill=(255, 255, 255))

    y = pad + banner_h + 16
    margin_x = pad + 16

    # If ID card, draw an ID photo avatar box
    if photo_type == "ID_CARD":
        avatar_w, avatar_h = 95, 115
        av_x = width - pad - avatar_w - 16
        av_y = y
        draw.rectangle([(av_x, av_y), (av_x + avatar_w, av_y + avatar_h)], fill=(226, 232, 240), outline=(148, 163, 184), width=2)
        draw.ellipse([(av_x + 32, av_y + 18), (av_x + 62, av_y + 48)], fill=(100, 116, 139))
        draw.pieslice([(av_x + 18, av_y + 55), (av_x + 77, av_y + 110)], 180, 360, fill=(100, 116, 139))
        draw.text((av_x + 12, av_y + avatar_h + 4), "[PHOTO ID]", font=small_font, fill=(100, 116, 139))

    for sec_title, sec_content in sections:
        if sec_title:
            draw.text((margin_x, y), sec_title, font=subhead_font, fill=(15, 23, 42))
            y += 24

        if isinstance(sec_content, str):
            lines = sec_content.split("\n")
        else:
            lines = sec_content

        for line in lines:
            if y > height - pad - 20:
                break
            draw.text((margin_x + 8, y), line, font=body_font, fill=(30, 41, 59))
            y += 20
        y += 10

    # Physical asset photo caption
    draw.text((pad + 12, height - pad - 15), "CAMPUS PHYSICAL ASSET PHOTOGRAPH — RAGNOVA CORPUS", font=small_font, fill=(148, 163, 184))

    out_path = IMAGES_DIR / filename
    img.save(out_path, format="PNG")
    print(f"wrote photo asset: {out_path.name} ({width}x{height})")


def generate_all_images():
    """Generates 15 deliberate images covering screenshots, posters, diagrams, and photos with visible text."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Portal Login Screenshot
    create_image(
        "screenshot_portal_login.png",
        700, 480, (248, 250, 252),
        "RAGNova Student Portal — Single Sign-On",
        [
            ("Student & Faculty Authentication", [
                "Service: Campus Central Single Sign-On (SSO)",
                "Notice: Activate your account before first use.",
                "",
                "[ Username / Institute Email ]",
                "  student.id@ragnova.edu",
                "",
                "[ Password ]",
                "  ••••••••••••••••",
                "",
                "Notice: Use the temporary password from your admission letter.",
                "Sharing credentials with another person violates institute policy.",
                "Support: Contact Information Technology help desk (9am - 5pm).",
            ])
        ],
        border_color=(203, 213, 225),
        badge="STUDENT PORTAL"
    )

    # 2. Wi-Fi Setup Screenshot (SECURE enterprise configuration)
    create_image(
        "screenshot_wifi_setup.png",
        650, 460, (241, 245, 249),
        "Network Settings — Wireless Connection",
        [
            ("Wireless Network: RAGNOVA-STUDENT", [
                "Status: Available across Hostel, Library, and Academic Blocks",
                "Security Protocol: WPA2-Enterprise (802.1X PEAP)",
                "Identity (Email): username@ragnova.edu",
                "Password: Same password as campus email activation",
                "CA Certificate: RAGNova Root CA (or System Trust Store)",
                "Server Domain Validation: ragnova.edu (Mandatory)",
                "Security Policy: Do NOT select 'Do not validate' (prevents rogue AP attacks)",
                "",
                "Troubleshooting:",
                "If connection fails repeatedly, restart the wireless adapter.",
                "Visit the IT help desk if persistent authentication errors occur."
            ])
        ],
        border_color=(148, 163, 184),
        badge="WIFI SETTINGS"
    )

    # 3. 403 Forbidden Access Error Screenshot
    create_image(
        "screenshot_error_403.png",
        650, 400, (254, 242, 242),
        "Access Denied — 403 Forbidden",
        [
            ("Institute Network Security Gateway", [
                "Your request to download academic content was blocked.",
                "Reason: Mass scraping or downloading copyrighted media is restricted.",
                "Policy Reference: Acceptable Use Policy §3.2 (Copyrighted Material)",
                "",
                "Notice of Penalties:",
                "First violation: Network access suspended for two weeks.",
                "Mandatory requirement: Complete IT awareness session before access restored.",
                "Second violation: Long-term suspension via Disciplinary Committee."
            ])
        ],
        border_color=(239, 68, 68),
        badge="SECURITY ALERT"
    )

    # 4. Synopsis Submission Portal Screenshot
    create_image(
        "screenshot_synopsis_portal.png",
        700, 500, (248, 250, 252),
        "CSE-AIML Final Year Project Submission Portal",
        [
            ("B.Tech Final Year Project — Synopsis Submission", [
                "Submission Deadline: 21st August (Strict deadline)",
                "Document format: Single PDF file (Max 3 pages excluding references)",
                "Team composition: 2 or 3 students per project team",
                "",
                "Evaluation Components:",
                "  - Working Prototype Demonstration: 40% weighting",
                "  - Written Project Report: 35% weighting",
                "  - Individual Viva Voce: 25% weighting",
                "",
                "File Selected: final_synopsis_team12.pdf (1.8 MB)",
                "[ SUBMIT SYNOPSIS FOR FACULTY GUIDE REVIEW ]"
            ])
        ],
        border_color=(59, 130, 246),
        badge="DEADLINE: 21 AUG"
    )

    # 5. VPN Client Screenshot
    create_image(
        "screenshot_vpn_client.png",
        640, 420, (243, 244, 246),
        "RAGNova SecureConnect VPN Client v2.4",
        [
            ("Remote Access to Campus Digital Library", [
                "Gateway: vpn.ragnova.edu:443",
                "Tunnel Status: Connected (AES-256 Encryption)",
                "Assigned IP: 10.120.44.18",
                "",
                "Accessible Services:",
                "  - Subscribed IEEE, ACM, and Springer Academic Journals",
                "  - Digital Resource Centre 2nd floor library catalog",
                "  - Department compute cluster SSH gateway",
                "",
                "Download client installer from the IT self-service portal."
            ])
        ],
        border_color=(107, 114, 128),
        badge="VPN CONNECTED"
    )

    # 6. Seminar Poster Notice
    create_image(
        "notice_seminar_poster.png",
        600, 520, (238, 242, 255),
        "Department of Computer Science & Engineering",
        [
            ("Guest Seminar: Multimodal Retrieval Augmented Generation", [
                "Date: September 28, 2026 | Time: 2:00 PM - 4:30 PM",
                "Venue: Main Auditorium Hall B, AIML Academic Block",
                "",
                "Key Topics Covered:",
                "  - Offline vector databases: ChromaDB and HNSW indexing",
                "  - Cross-modal retrieval using OpenCLIP embeddings",
                "  - Local LLM inference with Ollama and Llama 3.2 3B",
                "  - Verifiable citations from PDF, audio, and image sources",
                "",
                "Open to all B.Tech and M.Tech CSE-AIML students.",
                "Registration: Scan QR at Department Office or sign up on portal."
            ])
        ],
        border_color=(99, 102, 241),
        badge="SEMINAR NOTICE"
    )

    # 7. Library Fines Notice
    create_image(
        "notice_library_fines.png",
        620, 460, (255, 251, 235),
        "Central Library — Borrowing Policy and Overdue Fines",
        [
            ("Circulation Desk Policy Summary", [
                "Undergraduate borrowing quota: 4 books for 14 days",
                "Postgraduate and faculty quota: 8 books for 30 days",
                "One renewal permitted if no reservation is pending.",
                "",
                "Late Return Fine Schedule:",
                "  - Fine rate: 2 rupees per day per overdue book",
                "  - Maximum fine cap: 200 rupees per single title",
                "  - Borrowing privileges suspended when unpaid fines exceed cap.",
                "",
                "Working Hours: 8:00 AM to 10:00 PM on all working days.",
                "Help Desk: library@ragnova.edu | Ground Floor Counter"
            ])
        ],
        border_color=(245, 158, 11),
        badge="LIBRARY NOTICE"
    )

    # 8. Lab Rules Notice
    create_image(
        "notice_lab_rules.png",
        620, 440, (240, 253, 244),
        "AIML Computing Laboratory — Acceptable Use Rules",
        [
            ("Laboratory Guidelines & Account Security", [
                "1. Each account is strictly for individual student use.",
                "2. Credential sharing with any peer or friend is forbidden.",
                "3. No unlicensed media, game, or commercial downloading.",
                "",
                "Disciplinary Process:",
                "  - First violation: 2-week campus network suspension.",
                "  - Second violation: Referral to institute disciplinary committee.",
                "",
                "Laboratory Operating Hours: Monday to Friday 9:00 AM - 5:00 PM",
                "Report infractions to IT Help Desk, Ground Floor Academic Block."
            ])
        ],
        border_color=(34, 197, 94),
        badge="LAB SAFETY"
    )

    # 9. Midterm Schedule Notice
    create_image(
        "notice_midterm_schedule.png",
        650, 460, (245, 243, 255),
        "Final Year Project — Mid-Term Evaluation Timetable",
        [
            ("B.Tech Evaluation Weighting Scheme", [
                "Evaluation Panel Schedule:",
                "  - Team demonstrations: October 14th - 16th, 2026",
                "  - Venue: Conference Room 3, Department of CSE-AIML",
                "",
                "Marking Distribution:",
                "  - Working Prototype Demo: 40% (must run completely offline)",
                "  - Written Project Report: 35% (standard institute template)",
                "  - Individual Viva Voce: 25% (all team members must attend)",
                "",
                "Note: Absent members without prior dean approval receive zero for viva."
            ])
        ],
        border_color=(168, 85, 247),
        badge="EVALUATION DATES"
    )

    # 10. RAGNova Architecture Diagram
    create_image(
        "diagram_rag_architecture.png",
        720, 520, (15, 23, 42),
        "RAGNova Offline Multimodal Architecture Diagram",
        [
            ("SYSTEM PIPELINE FLOW", [
                "1. Input Modalities:",
                "   [ Documents: PDF/DOCX ]  [ Images: PNG/JPG ]  [ Audio: WAV/MP3 ]",
                "",
                "2. Ingestion & Embedding Layers:",
                "   - Text Pipeline: PyMuPDF / python-docx -> all-MiniLM-L6-v2 (384-dim)",
                "   - Vision Pipeline: Tesseract OCR + OpenCLIP ViT-B-32 (512-dim)",
                "   - Audio Pipeline: faster-whisper -> Text Chunks with timestamps",
                "",
                "3. Dual Vector Stores (ChromaDB):",
                "   [ text_index collection ]       [ image_index collection ]",
                "",
                "4. Retrieval & Local Generation:",
                "   - Cosine top-K retrieval -> Prompt with provenance chunks",
                "   - Ollama (Llama 3.2 3B) -> Grounded answer with numbered citations"
            ])
        ],
        border_color=(56, 189, 248),
        badge="ARCHITECTURE"
    )

    # 11. Campus Map Diagram
    create_image(
        "diagram_campus_map.png",
        680, 480, (248, 250, 252),
        "Campus Facility Blueprint & Directory",
        [
            ("Key Infrastructure Locations", [
                "[ Building 1 ] Administrative Block & Admission Registry",
                "[ Building 2 ] Department of CSE & AIML Labs (Auditorium Hall B)",
                "[ Building 3 ] Central Library & Digital Resource Centre (2nd Fl)",
                "[ Building 4 ] Information Technology Help Desk & Data Centre",
                "[ Building 5 ] Student Hostels & Residential Common Halls",
                "",
                "Wireless Coverage: RAGNOVA-STUDENT available in Buildings 1 through 5.",
                "Opening Hours:",
                "  - Central Library: 8:00 AM - 10:00 PM (24h during exams)",
                "  - IT Help Desk: 9:00 AM - 5:00 PM (Monday - Friday)"
            ])
        ],
        border_color=(100, 116, 139),
        badge="CAMPUS MAP"
    )

    # 12. Photo Asset 1: Student Identity Card (Photo with visible text)
    create_photo_image(
        "photo_id_card_sample.png",
        600, 380, (248, 250, 252),
        "RAGNova Institute of Technology",
        [
            ("Alex Morgan | Roll No: 23AIML042", [
                "Program: B.Tech Computer Science (AIML)",
                "Department: Computer Science & Engineering",
                "Library Barcode: *LIB-884920*",
                "Valid Session: 2023 - 2027",
                "Borrowing Privilege: 4 Books / 14 Days",
                "Campus Access: 24/7 Hostel & Library Reading Room"
            ])
        ],
        border_color=(79, 70, 229),
        badge="STUDENT ID",
        photo_type="ID_CARD"
    )

    # 13. Photo Asset 2: AI & ML Research Lab Door Sign (Photo with visible text)
    create_photo_image(
        "photo_lab_door_sign.png",
        620, 390, (245, 245, 245),
        "Department of Computer Science & Engineering",
        [
            ("AI & Machine Learning Research Laboratory", [
                "Location: Academic Block B — Room 302",
                "Faculty In-Charge: Dr. S. Rao (Associate Professor)",
                "Lab Superintendent: Mr. V. Sharma",
                "Operating Hours: Monday to Friday 09:00 AM - 05:00 PM",
                "Access Requirement: Smart Card ID Badge Required for Entry",
                "Safety Guideline: No food or unauthorized downloading."
            ])
        ],
        border_color=(30, 41, 59),
        badge="ROOM 302",
        photo_type="PLAQUE"
    )

    # 14. Photo Asset 3: Central Library Circulation Desk Sign (Photo with visible text)
    create_photo_image(
        "photo_library_desk_sign.png",
        620, 390, (254, 252, 232),
        "Central Library — Service Counter Notice",
        [
            ("Circulation Desk — Borrowing & Returns", [
                "Undergraduate Quota: 4 books for 14 days renewal",
                "Late Return Fine: Rs 2.00 per day per overdue book",
                "Overdue Fine Cap: Maximum Rs 200 per title",
                "After-Hours Drop Box: Located beside Ground Floor Main Exit",
                "Digital Resource Centre: Floor 2 (WiFi and Workstations)"
            ])
        ],
        border_color=(202, 138, 4),
        badge="CIRCULATION DESK",
        photo_type="SIGN"
    )

    # 15. Photo Asset 4: Campus Building B Directory Plaque (Photo with visible text)
    create_photo_image(
        "photo_campus_building_plaque.png",
        640, 400, (241, 245, 249),
        "Academic Block B Directory Plaque",
        [
            ("Department of CSE & AIML — Directory", [
                "Ground Floor: Department Office & Seminar Auditorium B",
                "Floor 1: B.Tech Classrooms & Faculty Cabins",
                "Floor 2: AI Research Lab (Room 302) & Compute Server Room",
                "Wireless Coverage: RAGNOVA-STUDENT (802.1X PEAP)",
                "IT Help Desk: Ground Floor Room 104 (Mon-Fri 9am-5pm)"
            ])
        ],
        border_color=(71, 85, 105),
        badge="BLOCK B",
        photo_type="PLAQUE"
    )


def _normalize_wav_to_16k_mono(filepath: Path) -> None:
    """Normalize WAV file to standard 16-bit PCM, 16000 Hz, single-channel mono.

    Guarantees host-independent, deterministic audio parameters across Windows,
    macOS, and Linux platforms, matching Whisper's native 16 kHz acoustic input.
    """
    import numpy as np

    with wave.open(str(filepath), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw_data = w.readframes(n_frames)

    if framerate == 16000 and n_channels == 1 and sampwidth == 2:
        return

    # Parse audio samples
    if sampwidth == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16)
    elif sampwidth == 1:
        samples = ((np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0).astype(np.int16)
    elif sampwidth == 4:
        samples = (np.frombuffer(raw_data, dtype=np.int32) // 65536).astype(np.int16)
    else:
        return

    # Downmix stereo to mono if necessary
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    # Resample to 16000 Hz via linear interpolation
    if framerate != 16000:
        target_len = int(len(samples) * 16000 / framerate)
        indices = np.linspace(0, len(samples) - 1, target_len)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.int16)

    with wave.open(str(filepath), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.tobytes())


def _generate_synthetic_speech_wav(filepath: Path, text: str) -> None:
    """Generate spoken speech .wav file using available platform TTS backends.

    Tries Windows PowerShell System.Speech, PowerShell Core, Windows SAPI VBScript,
    macOS 'say', Linux 'espeak-ng' / 'espeak', or 'pyttsx3'.
    Always wipes any pre-existing file before generating to avoid accepting stale audio.
    Fails loudly if no TTS engine is found: acoustic sine-wave tones contain NO speech
    and cannot be used for Whisper ASR.
    Normalizes generated audio to 16 kHz 16-bit mono PCM to prevent host-dependent formats.
    """
    # Remove any existing target file to prevent failed attempts from accepting stale audio
    filepath.unlink(missing_ok=True)

    # 1. Windows PowerShell System.Speech
    escaped_text = text.replace("'", "''")
    ps_cmd = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$synth.SetOutputToWaveFile('{filepath.as_posix()}'); "
        f"$synth.Speak('{escaped_text}'); "
        f"$synth.Dispose()"
    )
    for ps_bin in ["powershell", "pwsh"]:
        try:
            res = subprocess.run([ps_bin, "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and filepath.exists() and filepath.stat().st_size > 1000:
                _normalize_wav_to_16k_mono(filepath)
                print(f"wrote audio via {ps_bin} TTS: {filepath.name} ({filepath.stat().st_size} bytes, 16000Hz mono)")
                return
        except Exception:
            pass

    # 2. Windows SAPI via VBScript
    if sys.platform.startswith("win"):
        vbs_script = filepath.with_suffix(".vbs")
        try:
            clean_text = text.replace('"', '""')
            vbs_content = (
                f'Set voice = CreateObject("SAPI.SpVoice")\n'
                f'Set stream = CreateObject("SAPI.SpFileStream")\n'
                f'stream.Open "{filepath.as_posix()}", 3, False\n'
                f'Set voice.AudioOutputStream = stream\n'
                f'voice.Speak "{clean_text}"\n'
                f'stream.Close\n'
            )
            vbs_script.write_text(vbs_content, encoding="utf-8")
            res = subprocess.run(["cscript", "//nologo", str(vbs_script)], capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and filepath.exists() and filepath.stat().st_size > 1000:
                _normalize_wav_to_16k_mono(filepath)
                print(f"wrote audio via Windows SAPI: {filepath.name} ({filepath.stat().st_size} bytes, 16000Hz mono)")
                return
        except Exception:
            pass
        finally:
            if vbs_script.exists():
                vbs_script.unlink(missing_ok=True)

    # 3. macOS 'say' command
    if sys.platform == "darwin":
        try:
            res = subprocess.run(["say", "-o", str(filepath), "--data-format=LEI16@16000", text], capture_output=True, timeout=30)
            if res.returncode == 0 and filepath.exists() and filepath.stat().st_size > 1000:
                _normalize_wav_to_16k_mono(filepath)
                print(f"wrote audio via macOS say: {filepath.name} ({filepath.stat().st_size} bytes, 16000Hz mono)")
                return
        except Exception:
            pass

    # 4. Linux / cross-platform espeak-ng / espeak
    for espeak_bin in ["espeak-ng", "espeak"]:
        try:
            res = subprocess.run([espeak_bin, "-w", str(filepath), text], capture_output=True, timeout=30)
            if res.returncode == 0 and filepath.exists() and filepath.stat().st_size > 1000:
                _normalize_wav_to_16k_mono(filepath)
                print(f"wrote audio via {espeak_bin}: {filepath.name} ({filepath.stat().st_size} bytes, 16000Hz mono)")
                return
        except Exception:
            pass

    # 5. Python pyttsx3 (if installed in virtual environment)
    try:
        # pyrefly: ignore [missing-import]
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, str(filepath))
        engine.runAndWait()
        if filepath.exists() and filepath.stat().st_size > 1000:
            _normalize_wav_to_16k_mono(filepath)
            print(f"wrote audio via pyttsx3: {filepath.name} ({filepath.stat().st_size} bytes, 16000Hz mono)")
            return
    except Exception:
        pass

    # If no speech engine succeeded, NEVER emit sine-wave beeps.
    # Spoken audio is required for faster-whisper ASR.
    raise RuntimeError(
        f"Failed to generate speech audio for '{filepath.name}'. "
        f"No functional Text-to-Speech (TTS) engine was detected. "
        f"Tested: Windows PowerShell System.Speech, Windows SAPI, macOS 'say', Linux 'espeak-ng'/'espeak', and 'pyttsx3'. "
        f"Non-speech acoustic waveforms are rejected because faster-whisper requires spoken human language."
    )


def generate_all_audio():
    """Generates 4 campus audio clips (.wav) corresponding to the academic topics."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    audio_clips = [
        (
            "hod_project_announcement.wav",
            "Good morning final year students. This is an important announcement regarding your B.Tech CSE-AIML "
            "project evaluation. Please remember that the working prototype demonstrated on evaluation day carries "
            "forty percent of your total marks. The written project report carries thirty-five percent, and the "
            "individual viva voce with each team member carries the remaining twenty-five percent. Every member must "
            "attend the viva in person. Do not miss the synopsis submission deadline on August twenty-first."
        ),
        (
            "library_orientation_excerpt.wav",
            "Welcome to the Central Library orientation. Effective this semester, our working hours are eight in the "
            "morning until ten at night on working days, and nine in the morning until six in the evening on weekends. "
            "Undergraduate students may check out up to four books at a time for fourteen days. Overdue fines are two "
            "rupees per day, capped at two hundred rupees per title. The digital resource centre is located on the "
            "second floor."
        ),
        (
            "it_helpdesk_wifi_instructions.wav",
            "Hello students. If you are experiencing difficulty connecting to the campus wireless network named "
            "RAGNOVA-STUDENT, please follow these steps. Enter your full institute email address and the password set "
            "during account activation. Ensure your device validates server certificates using domain ragnova.edu. If the connection fails, restart your wireless adapter before visiting the help "
            "desk. Remember that sharing your credentials with any other student is strictly prohibited and results in a "
            "two-week network suspension."
        ),
        (
            "lab_assistant_briefing.wav",
            "Attention students in the AI and Machine Learning laboratory. If you need remote access to academic journal "
            "databases while working from home, please install the institute VPN client available on the self-service "
            "portal. Downloading copyrighted movies or commercial software over the campus network is strictly barred. "
            "The IT help desk is open Monday to Friday from nine in the morning until five in the evening."
        ),
    ]

    for filename, text in audio_clips:
        filepath = AUDIO_DIR / filename
        _generate_synthetic_speech_wav(filepath, text)


def main():
    print(f"Generating multimodal corpus under {DATA_DIR}...")
    generate_all_images()
    generate_all_audio()
    print("\nMultimodal corpus generation complete.")


if __name__ == "__main__":
    main()

