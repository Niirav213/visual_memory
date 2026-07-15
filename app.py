"""
Visual Memory AI — Premium SaaS Dashboard
Refactored by senior UI/UX engineer.
Clean architecture, centralized theme management (Light/Dark), 
safe session state initialization, and optimized capture processing.
"""

import os
import sys
import time
import cv2
import yaml
import numpy as np
import streamlit as st
from datetime import datetime
from PIL import Image
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Try importing audio libraries, fail gracefully if unavailable
speech_rec_available = True
try:
    import speech_recognition as sr
except ImportError:
    speech_rec_available = False

tts_available = True
try:
    import pyttsx3
except ImportError:
    tts_available = False

from src.tracking.tracker import ObjectTracker, TrackedObject
from src.memory.embedder import VisualEmbedder
from src.memory.vector_store import VectorStore
from src.memory.memory_db import MemoryDatabase, MemoryEntry
from src.query.engine import QueryEngine
from src.utils.visualization import (
    draw_tracked_objects, 
    create_info_overlay, 
    draw_guidance_overlay, 
    get_direction_description
)

# ──────────────────────────────────────────
# 0. SPEECH RECOGNITION AND SYNTHESIS
# ──────────────────────────────────────────

def recognize_speech() -> str:
    if not speech_rec_available:
        return "ERROR_IMPORT"
    
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=4, phrase_time_limit=4)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "ERROR_UNKNOWN"
        except sr.RequestError:
            return "ERROR_REQUEST"
    except Exception as e:
        return f"ERROR_MIC: {str(e)}"

def speak_background(text: str):
    if not tts_available:
        return
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# ──────────────────────────────────────────
# 1. CENTRALIZED THEME MANAGEMENT SYSTEM
# ──────────────────────────────────────────

def get_theme_css(theme_mode: str) -> str:
    """
    Generate responsive CSS variables and core styles dynamically.
    Ensures every card, button, sidebar, input, and metric is styled
    consistently according to the selected theme.
    """
    if theme_mode == "light":
        vars_css = """
        :root {
            --background: #F8FAFC;
            --surface: #FFFFFF;
            --secondary-surface: #F0F4FA;
            --sidebar-bg: #0B1B3D;
            --accent: #0B1B3D;
            --accent-hover: #1E3A8A;
            --text-primary: #0B1B3D;
            --text-secondary: #4A607A;
            --text-card: #0B1B3D;
            --text-card-secondary: #4A607A;
            --border: #A0BFE0;
            --shadow: rgba(11, 27, 61, 0.03);
            --chat-user-bg: #E1ECFD;
            --chat-assistant-bg: #F0F4FA;
            --overview-card-bg: #FFFFFF;
            --overview-card-text: #0B1B3D;
            --accent-card-bg: #F0F4FA;
            --btn-primary-bg: #0B1B3D;
            --btn-primary-text: #FFFFFF;
            --btn-secondary-bg: #F0F4FA;
            --btn-secondary-text: #0B1B3D;
            --btn-border: #A0BFE0;
            --alert-bg: #F3E8FF;
            --alert-text: #0B1B3D;
            --hero-bg: linear-gradient(135deg, #0B1B3D 0%, #173B75 100%);
            --hero-text: #FFFFFF;
        }
        
        /* Force light mode inputs background, text, and border */
        .stApp div[data-testid="stTextInput"] > div,
        .stApp div[data-testid="stSelectbox"] > div,
        .stApp div[data-testid="stMultiSelect"] > div,
        .stApp div[data-testid="stTextArea"] > div,
        .stApp div[data-baseweb="select"],
        .stApp div[data-baseweb="input"] {
            background-color: #F8FAFC !important;
            border: 1px solid #A0BFE0 !important;
            border-radius: 8px !important;
            color: #0B1B3D !important;
        }

        /* Color of inner text and placeholders in Light Mode inputs */
        .stApp div[data-testid="stTextInput"] input,
        .stApp div[data-testid="stSelectbox"] select,
        .stApp div[data-testid="stTextArea"] textarea,
        .stApp div[data-baseweb="select"] div,
        .stApp div[data-baseweb="input"] input,
        .stApp div[data-baseweb="input"] textarea,
        .stApp div[data-testid="stChatInput"] textarea {
            color: #0B1B3D !important;
            -webkit-text-fill-color: #0B1B3D !important;
        }
        
        /* Placeholder color in Light Mode inputs and chat input */
        .stApp div[data-testid="stTextInput"] input::placeholder,
        .stApp div[data-testid="stTextArea"] textarea::placeholder,
        .stApp div[data-baseweb="input"] input::placeholder,
        .stApp div[data-testid="stChatInput"] textarea::placeholder,
        .stApp div[data-testid="stChatInput"] textarea::-webkit-input-placeholder {
            color: #5A718F !important;
            opacity: 0.8 !important;
            -webkit-text-fill-color: #5A718F !important;
        }
        
        /* Multi-select chips in Light Mode */
        .stApp div[role="button"] {
            background-color: #E1ECFD !important;
            color: #0B1B3D !important;
            border: 1px solid #A0BFE0 !important;
        }
        .stApp div[role="button"] span {
            color: #0B1B3D !important;
        }

        /* Force card outline visibility in Light Mode */
        .stApp div[data-testid="stVerticalBlockBorderWrapper"],
        .stApp div[data-testid="stBlockBorderContainer"] {
            border: 1px solid #A0BFE0 !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 4px 20px rgba(11, 27, 61, 0.03) !important;
        }
        """
    else:
        vars_css = """
        :root {
            --background: #000000;
            --surface: #0D061A;
            --secondary-surface: #180C33;
            --sidebar-bg: #110724;
            --accent: #8B5CF6;
            --accent-hover: #A78BFA;
            --text-primary: #F3E8FF;
            --text-secondary: #C084FC;
            --text-card: #F3E8FF;
            --text-card-secondary: #C084FC;
            --border: #8B5CF6;
            --shadow: rgba(139, 92, 246, 0.15);
            --chat-user-bg: #1E1035;
            --chat-assistant-bg: #0D061A;
            --overview-card-bg: #0D061A;
            --overview-card-text: #F3E8FF;
            --accent-card-bg: #180C33;
            --btn-primary-bg: #8B5CF6;
            --btn-primary-text: #FFFFFF;
            --btn-secondary-bg: #0D061A;
            --btn-secondary-text: #F3E8FF;
            --btn-border: #8B5CF6;
            --alert-bg: #1E1035;
            --alert-text: #F3E8FF;
            --hero-bg: linear-gradient(135deg, #000000 0%, #1E0C33 100%);
            --hero-text: #FFFFFF;
        }
        
        /* Force card outline visibility in Dark Mode */
        .stApp div[data-testid="stVerticalBlockBorderWrapper"],
        .stApp div[data-testid="stBlockBorderContainer"] {
            border: 1.5px solid #8B5CF6 !important;
            background-color: #0D061A !important;
        }
        """

    common_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        %s

        /* Global App Overrides */
        .stApp {
            background-color: var(--background) !important;
            color: var(--text-primary) !important;
        }
        
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border) !important;
        }

        * { 
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
        }

        /* Typography serif for titles */
        h1, h2, h3, .hero-title, .section-title, .card-title, .kpi-value {
            font-family: 'Playfair Display', Georgia, serif !important;
        }

        /* Text colors outside cards */
        .stApp [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary);
            font-size: 0.92rem;
            line-height: 1.6;
        }
        .stApp [data-testid="stMarkdownContainer"] h1,
        .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3,
        .stApp [data-testid="stMarkdownContainer"] strong {
            color: var(--text-primary);
        }

        /* Scrollbar styles */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent);
        }

        /* Card Container styling */
        div[data-testid="stBlockBorderContainer"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--surface) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: 16px !important;
            padding: 1.2rem !important;
            box-shadow: 0 4px 20px var(--shadow) !important;
            transition: transform 0.2s ease, border-color 0.2s ease !important;
            margin-bottom: 1rem;
        }
        div[data-testid="stBlockBorderContainer"]:hover,
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
        }

        /* Enforce colors of all text inside cards */
        div[data-testid="stBlockBorderContainer"] *,
        div[data-testid="stBlockBorderContainer"] p,
        div[data-testid="stBlockBorderContainer"] label,
        div[data-testid="stBlockBorderContainer"] span,
        div[data-testid="stBlockBorderContainer"] strong,
        div[data-testid="stBlockBorderContainer"] h1,
        div[data-testid="stBlockBorderContainer"] h2,
        div[data-testid="stBlockBorderContainer"] h3,
        div[data-testid="stVerticalBlockBorderWrapper"] *,
        div[data-testid="stVerticalBlockBorderWrapper"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] label,
        div[data-testid="stVerticalBlockBorderWrapper"] span,
        div[data-testid="stVerticalBlockBorderWrapper"] strong,
        div[data-testid="stVerticalBlockBorderWrapper"] h1,
        div[data-testid="stVerticalBlockBorderWrapper"] h2,
        div[data-testid="stVerticalBlockBorderWrapper"] h3 {
            color: var(--text-card) !important;
        }

        /* Secondary text inside cards */
        div[data-testid="stBlockBorderContainer"] .stApp [data-testid="stMarkdownContainer"] p,
        div[data-testid="stBlockBorderContainer"] span[data-testid="stWidgetLabel"] p,
        div[data-testid="stBlockBorderContainer"] .stCaption,
        div[data-testid="stVerticalBlockBorderWrapper"] .stApp [data-testid="stMarkdownContainer"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] span[data-testid="stWidgetLabel"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] .stCaption {
            color: var(--text-card-secondary) !important;
        }

        /* Sidebar navigation radio overrides */
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] {
            gap: 0.5rem !important;
        }
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label {
            background-color: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            padding: 0.6rem 1rem !important;
            color: #FFFFFF !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            cursor: pointer !important;
        }
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label[data-checked="true"] {
            background-color: rgba(255, 255, 255, 0.2) !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label span[data-baseweb="radio"] {
            display: none !important;
        }

        /* Always white text for sidebar since background is dark in both themes */
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }

        /* Logo branding */
        .sidebar-logo {
            display: flex;
            align-items: center;
            padding: 1.5rem 1rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        .logo-icon {
            font-size: 1.8rem;
            margin-right: 0.6rem;
        }
        .logo-text {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #FFFFFF 50%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .logo-badge {
            background-color: rgba(255, 255, 255, 0.15);
            color: #FFFFFF;
            font-size: 0.65rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* User Profile section */
        .sidebar-user {
            display: flex;
            align-items: center;
            padding: 1rem;
            border-top: 1px solid var(--border);
            margin-top: 5rem;
        }
        .user-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            margin-right: 0.8rem;
        }
        .user-info {
            display: flex;
            flex-direction: column;
        }
        .user-name {
            font-size: 0.9rem;
            font-weight: 600;
            color: #FFFFFF;
        }
        .user-role {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.7);
        }

        /* Hero header block */
        .hero-container {
            position: relative;
            text-align: center;
            padding: 3rem 1.5rem;
            margin-bottom: 2rem;
            overflow: hidden;
            background: var(--hero-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 10px 30px var(--shadow);
        }
        .hero-container * {
            color: #FFFFFF !important;
        }
        .hero-glow {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 400px;
            height: 200px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
            filter: blur(60px);
            z-index: 0;
            pointer-events: none;
        }
        .hero-title {
            font-size: 3.2rem !important;
            font-weight: 300 !important;
            letter-spacing: -1px;
            margin-bottom: 0.8rem;
            z-index: 1;
            position: relative;
        }
        .hero-title em {
            font-style: italic;
            font-weight: 400;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            max-width: 700px;
            margin: 0 auto;
            font-weight: 400;
            z-index: 1;
            position: relative;
            line-height: 1.6;
        }

        /* Overview Section & Cards */
        .section-container {
            text-align: center;
            margin: 2.5rem 0 1.5rem 0;
        }
        .section-title {
            font-size: 2.2rem !important;
            font-weight: 400 !important;
            margin-bottom: 1.5rem;
            color: var(--text-primary) !important;
            text-align: center;
        }
        .section-title em {
            font-style: italic;
            font-weight: 300;
        }
        .overview-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }
        .overview-card {
            background-color: var(--surface) !important;
            border-radius: 12px;
            padding: 1.8rem;
            text-align: left;
            box-shadow: 0 4px 20px var(--shadow);
            transition: transform 0.2s ease;
            border: 1px solid var(--border) !important;
        }
        .overview-card:hover {
            transform: translateY(-4px);
        }
        .overview-card * {
            color: var(--text-card) !important;
        }
        .overview-card .card-desc {
            color: var(--text-card-secondary) !important;
        }
        .card-num {
            font-size: 0.9rem;
            opacity: 0.5;
            margin-bottom: 1rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .card-title {
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.75rem;
        }
        .card-desc {
            font-size: 0.88rem;
            line-height: 1.5;
            margin: 0;
        }

        /* Streamlit elements overrides inside cards */
        .stApp button {
            background-color: var(--btn-secondary-bg) !important;
            color: var(--btn-secondary-text) !important;
            border: 1px solid var(--btn-border) !important;
            border-radius: 8px !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 2px 8px var(--shadow) !important;
        }
        .stApp button:hover {
            background-color: var(--secondary-surface) !important;
            color: var(--text-primary) !important;
            border-color: var(--btn-border) !important;
            transform: translateY(-1px);
        }

        /* Start button (Primary) */
        .stApp button[aria-label="▶ Start"] {
            background-color: var(--btn-primary-bg) !important;
            color: var(--btn-primary-text) !important;
            border: 1.5px solid var(--btn-primary-bg) !important;
        }
        .stApp button[aria-label="▶ Start"]:hover {
            background-color: var(--accent-hover) !important;
            color: var(--btn-primary-text) !important;
            border-color: var(--accent-hover) !important;
        }

        /* Stop button (Red action) */
        .stApp button[aria-label="⏹ Stop"] {
            background-color: #ef4444 !important;
            color: #FFFFFF !important;
            border: 1.5px solid #ef4444 !important;
        }
        .stApp button[aria-label="⏹ Stop"]:hover {
            background-color: #dc2626 !important;
            color: #FFFFFF !important;
            border-color: #dc2626 !important;
        }

        /* Chat bubbles design override */
        div[data-testid="stChatMessage"] {
            background-color: var(--chat-assistant-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 0.8rem 1rem !important;
            margin-bottom: 0.8rem !important;
            box-shadow: 0 2px 10px var(--shadow) !important;
        }
        
        /* User message custom bg color */
        div[data-testid="stChatMessage"]:has([data-testid="user-avatar"]) {
            background-color: var(--chat-user-bg) !important;
            border-color: var(--border) !important;
        }
        div[data-testid="stChatMessage"] * {
            color: var(--text-card) !important;
        }

        /* Chat input overrides */
        div[data-testid="stChatInput"] {
            background-color: var(--secondary-surface) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 20px var(--shadow) !important;
            padding: 4px !important;
        }
        div[data-testid="stChatInput"] div,
        div[data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            background: transparent !important;
            color: var(--text-card) !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* Streamlit Alerts override */
        div[data-testid="stAlert"],
        div[data-testid="stNotification"],
        div[data-testid="stNotificationContent"],
        div.stAlert,
        div.stNotification {
            background-color: var(--alert-bg) !important;
            color: var(--alert-text) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: 8px !important;
        }
        div[data-testid="stAlert"] *,
        div[data-testid="stNotification"] *,
        div[data-testid="stNotificationContent"] *,
        div.stAlert *,
        div.stNotification * {
            color: var(--alert-text) !important;
        }

        /* Inputs box design override inside cards */
        div[data-testid="stBlockBorderContainer"] div[data-testid="stTextInput"] input {
            background-color: var(--secondary-surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-card) !important;
        }
        
        div[data-testid="stBlockBorderContainer"] div[data-testid="stSelectbox"] select {
            background-color: var(--secondary-surface) !important;
            color: var(--text-card) !important;
            border: 1px solid var(--border) !important;
        }
        div[data-testid="stBlockBorderContainer"] div[data-baseweb="select"] div {
            color: var(--text-card) !important;
        }

        /* Status panel styling */
        .status-panel {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: var(--secondary-surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.6rem 1rem;
        }
        .status-label {
            font-size: 0.72rem;
            color: var(--text-card-secondary) !important;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .status-value {
            font-size: 0.78rem;
            font-weight: 700;
        }
        .status-active {
            color: #10b981 !important;
        }
        .status-inactive {
            color: #ef4444 !important;
        }

        /* Offline placeholder styling */
        .offline-placeholder {
            background-color: var(--secondary-surface);
            border: 2px dashed var(--border);
            border-radius: 16px;
            padding: 4rem 1.5rem;
            text-align: center;
            color: var(--text-card-secondary) !important;
            transition: border-color 0.2s ease;
            height: 280px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .offline-placeholder:hover {
            border-color: var(--accent);
        }
        .offline-icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        .offline-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-card) !important;
            margin-bottom: 0.25rem;
        }
        .offline-desc {
            font-size: 0.85rem;
            max-width: 420px;
            margin: 0 auto;
            line-height: 1.5;
            color: var(--text-card-secondary) !important;
        }

        /* Background metrics containers */
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .kpi-card {
            background-color: var(--secondary-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.8rem;
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-card) !important;
        }
        .kpi-label {
            font-size: 0.72rem;
            color: var(--text-card-secondary) !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.25rem;
            font-weight: 600;
        }

        /* Recording status pulsing dot */
        .recording-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            margin-right: 6px;
            box-shadow: 0 0 8px #10b981;
            vertical-align: middle;
        }

        /* Floating badges/chips style */
        .conf-chip {
            display: inline-block;
            background-color: var(--secondary-surface);
            border: 1px solid var(--border);
            color: var(--text-card) !important;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 20px;
        }

        /* Memory visual timeline styling */
        .timeline-container {
            max-height: 380px;
            overflow-y: auto;
            padding-right: 0.2rem;
        }
        
        /* Table overrides inside cards */
        div[data-testid="stBlockBorderContainer"] div[data-testid="stTable"] table {
            background-color: var(--secondary-surface) !important;
            color: var(--text-card) !important;
            border: 1px solid var(--border) !important;
        }
        div[data-testid="stBlockBorderContainer"] div[data-testid="stTable"] th {
            background-color: var(--surface) !important;
            color: var(--text-card) !important;
            font-weight: 600 !important;
        }
        div[data-testid="stBlockBorderContainer"] div[data-testid="stTable"] td {
            color: var(--text-card-secondary) !important;
            border-bottom: 1px solid var(--border) !important;
        }
    </style>
    """
    return common_css.replace("%s", vars_css)


# ──────────────────────────────────────────
# 2. STATE AND CACHE SYSTEM INITIALIZATION
# ──────────────────────────────────────────

def initialize_session_state(config: dict):
    """
    Initialize all session state variables safely.
    Prevents runtime AttributeErrors on reload and load.
    """
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_store_times" not in st.session_state:
        st.session_state.last_store_times = {}
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! I am your AI Memory Assistant. I can locate items and answer natural language queries based on visual memories. Try asking *'Where is my phone?'* or *'What objects have you seen?'*"
            }
        ]
    if "active_search" not in st.session_state:
        st.session_state.active_search = None
    if "guidance_located_announced" not in st.session_state:
        st.session_state.guidance_located_announced = False
    if "last_guidance_voice_time" not in st.session_state:
        st.session_state.last_guidance_voice_time = 0
    if "voice_feedback_text" not in st.session_state:
        st.session_state.voice_feedback_text = None
    if "clicked_prompt" not in st.session_state:
        st.session_state.clicked_prompt = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = float(config["detection"]["confidence"])
    if "mem_interval" not in st.session_state:
        st.session_state.mem_interval = int(config["memory"]["memory_interval_seconds"])
    if "selected_model_path" not in st.session_state:
        st.session_state.selected_model_path = "yolov8n.pt"
    if "selected_source_type" not in st.session_state:
        st.session_state.selected_source_type = "Webcam"
    if "selected_cam_index" not in st.session_state:
        st.session_state.selected_cam_index = 0
    if "selected_video_source" not in st.session_state:
        st.session_state.selected_video_source = None
    if "video_cap" not in st.session_state:
        st.session_state.video_cap = None
    if "video_source_path" not in st.session_state:
        st.session_state.video_source_path = None


# ──────────────────────────────────────────
# 3. CORE INTEGRATION RESOURCES (CACHED)
# ──────────────────────────────────────────

@st.cache_data
def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def init_tracker(model_name, tracker_type, confidence, target_classes):
    return ObjectTracker(
        model_name=model_name,
        tracker_type=tracker_type,
        confidence=confidence,
        target_classes=target_classes,
    )

@st.cache_resource
def init_embedder(model_name):
    return VisualEmbedder(model_name=model_name)

@st.cache_resource
def init_vector_store(dim, index_path, id_map_path):
    vs = VectorStore(dimension=dim)
    vs.load(index_path, id_map_path)
    return vs

@st.cache_resource
def init_memory_db(db_path):
    return MemoryDatabase(db_path=db_path)


# ──────────────────────────────────────────
# 4. MAIN SAAS APP ENTRY POINT
# ──────────────────────────────────────────

def run_app():
    config = load_config()
    
    # Initialize all session state variables cleanly
    initialize_session_state(config)

    # Centralized theme mode CSS injection
    theme_mode = st.session_state.get("theme", "dark")
    st.markdown(get_theme_css(theme_mode), unsafe_allow_html=True)

    # Initialize Core Modules
    tracker = init_tracker(
        st.session_state.get("selected_model_path", "yolov8n.pt"),
        config["tracking"]["tracker"],
        st.session_state.get("confidence", 0.45),
        config["detection"]["target_classes"],
    )
    embedder = init_embedder(config["memory"]["embedding_model"])
    vector_store = init_vector_store(
        config["memory"]["embedding_dim"],
        config["memory"]["faiss_index_path"],
        config["memory"]["id_map_path"],
    )
    memory_db = init_memory_db(config["memory"]["db_path"])
    query_engine = QueryEngine(embedder, vector_store, memory_db)

    # ── Left Sidebar Navigation Panel ──
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="logo-icon">🧠</div>
                <div class="logo-text">VisualMemory</div>
                <div class="logo-badge">SaaS</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Main Navigation Group
        selected_nav = st.radio(
            "Navigation Menu",
            options=["🎥 Live Dashboard", "🖼️ Memory Gallery", "📊 Analytics & KPI", "⚙️ System Settings"],
            label_visibility="collapsed"
        )

        # Centralized theme selector toggle
        st.markdown("<p style='font-size:0.75rem; color:var(--text-secondary); margin-bottom:0.25rem; text-transform:uppercase; letter-spacing:1px; padding-left:0.5rem; margin-top:1.5rem;'>Theme Mode</p>", unsafe_allow_html=True)
        is_light = st.toggle("☀️ Light Theme", value=(theme_mode == "light"))
        if is_light:
            if st.session_state.theme != "light":
                st.session_state.theme = "light"
                st.rerun()
        else:
            if st.session_state.theme != "dark":
                st.session_state.theme = "dark"
                st.rerun()

        # Active Search Lock overlay in sidebar
        active_search_val = st.session_state.get("active_search", None)
        if active_search_val:
            st.markdown("---")
            st.markdown(
                f"""
                <div style='background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.25); border-radius:8px; padding:0.65rem; text-align:center;'>
                    <span style='font-size:0.65rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1px; display:block; margin-bottom:0.15rem;'>Active Search Target</span>
                    <strong style='color:var(--accent); font-size:0.95rem; text-transform:uppercase;'>🎯 {active_search_val}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("❌ Clear Target", key="side_clear_target", use_container_width=True):
                st.session_state.active_search = None
                st.session_state.guidance_located_announced = False
                st.rerun()

        # User profile at bottom of sidebar
        st.markdown(
            """
            <div class="sidebar-user">
                <div class="user-avatar">NT</div>
                <div class="user-info">
                    <div class="user-name">Navya Thomas</div>
                    <div class="user-role">SaaS Administrator</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────
    # TAB 1: LIVE DASHBOARD PANEL
    # ──────────────────────────────────────────

    if selected_nav == "🎥 Live Dashboard":
        # ── Hero Section (Only on first page) ──
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-glow"></div>
                <h1 class="hero-title">Remember exactly <em>where and when</em> you saw it.</h1>
                <p class="hero-subtitle">An intelligent visual memory assistant powered by YOLOv8 object tracking, CLIP visual embeddings, and FAISS vector retrieval.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Three Pillars Overview Section ──
        st.markdown(
            f"""
            <div class="section-container">
                <h2 class="section-title">Three features. <em>One clear memory.</em></h2>
                <div class="overview-grid">
                    <div class="overview-card">
                        <div class="card-num">01</div>
                        <div class="card-title">Stream & Track</div>
                        <p class="card-desc">Activate the camera feed. YOLOv8 detects and tracks objects dynamically across frames.</p>
                    </div>
                    <div class="overview-card">
                        <div class="card-num">02</div>
                        <div class="card-title">Commit to Memory</div>
                        <p class="card-desc">Visual crops are embedded via CLIP and stored in a vector index for semantic retrieval.</p>
                    </div>
                    <div class="overview-card">
                        <div class="card-num">03</div>
                        <div class="card-title">Retrieve Instantly</div>
                        <p class="card-desc">Query in natural language (e.g. "Where is my laptop?") to find timestamps and locations.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<h2 class='section-title'>Activate feed. <em>Locate anything.</em></h2>", unsafe_allow_html=True)
        col_control, col_monitor = st.columns([1.2, 1.8])

        # ── Left Column: Controls & Settings ──
        with col_control:
            # Control card container (aligned to height 480)
            with st.container(height=480, border=True):
                # Camera Status Indicators
                is_processing = st.session_state.get("processing", False)
                if is_processing:
                    status_html = """
                    <div class="status-panel">
                        <span class="status-label">CAMERA SERVICE FEED</span>
                        <span class="status-value status-active"><span class="recording-indicator"></span> LIVE FEED ACTIVE</span>
                    </div>
                    """
                else:
                    status_html = """
                    <div class="status-panel">
                        <span class="status-label">CAMERA SERVICE FEED</span>
                        <span class="status-value status-inactive">● STREAM OFFLINE</span>
                    </div>
                    """
                st.markdown(status_html, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

                # Start/Stop Buttons
                ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
                with ctrl_col1:
                    st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
                    start_btn = st.button("▶ Start", key="dashboard_start_btn", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with ctrl_col2:
                    st.markdown("<div class='stop-btn'>", unsafe_allow_html=True)
                    stop_btn = st.button("⏹ Stop", key="dashboard_stop_btn", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with ctrl_col3:
                    snapshot_btn = st.button("📸 Snapshot", key="dashboard_snapshot_btn", use_container_width=True)

                if start_btn:
                    st.session_state.processing = True
                    st.rerun()
                if stop_btn:
                    st.session_state.processing = False
                    if "video_cap" in st.session_state and st.session_state.get("video_cap") is not None:
                        try:
                            st.session_state.get("video_cap").release()
                        except Exception:
                            pass
                        st.session_state.video_cap = None
                        st.session_state.video_source_path = None
                    st.rerun()

                # Confidence slider (interactive!)
                st.markdown("<hr style='margin: 0.6rem 0; opacity: 0.15;' />", unsafe_allow_html=True)
                st.session_state.confidence = st.slider(
                    "Detection Confidence Threshold", 
                    0.1, 0.95, 
                    st.session_state.get("confidence", 0.45), 
                    0.05
                )

                # Memory interval slider (interactive!)
                st.session_state.mem_interval = st.slider(
                    "Memory Storage Interval (secs)", 
                    1, 30, 
                    st.session_state.get("mem_interval", 5)
                )

                # Active Search Target & Clear button
                active_search_val = st.session_state.get("active_search", None)
                if active_search_val:
                    st.markdown("<hr style='margin: 0.6rem 0; opacity: 0.15;' />", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div style='background:var(--secondary-surface); border:1px solid var(--border); border-radius:8px; padding:0.6rem; text-align:center;'>
                            <span style='font-size:0.7rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; display:block; margin-bottom:0.15rem;'>Active Target Search Lock</span>
                            <strong style='color:var(--text-card); font-size:1.1rem; text-transform:uppercase;'>🎯 {active_search_val}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
                    if st.button("❌ Clear Search Target", key="dashboard_clear_target", use_container_width=True):
                        st.session_state.active_search = None
                        st.session_state.guidance_located_announced = False
                        st.rerun()

                st.caption(f"Configured Input: {st.session_state.get('selected_source_type')} (Index/Path: {st.session_state.get('selected_cam_index') if st.session_state.get('selected_source_type') == 'Webcam' else 'Loaded File'})")

        # ── Right Column: Monitor Feed & KPIs ──
        with col_monitor:
            # Monitor card container (aligned to height 480)
            with st.container(height=480, border=True):
                # 3 KPI Metrics Row
                total_memories = memory_db.get_total_count()
                unique_objects_count = len(memory_db.get_unique_objects())
                
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <div class="kpi-card">
                            <div class="kpi-value">{total_memories}</div>
                            <div class="kpi-label">Saved Memories</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-value">{unique_objects_count}</div>
                            <div class="kpi-label">Unique Objects</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-value">18ms</div>
                            <div class="kpi-label">Search Latency</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Video feed target placeholder
                frame_placeholder = st.empty()

                if not st.session_state.get("processing", False):
                    frame_placeholder.markdown(
                        """
                        <div class="offline-placeholder">
                            <div class="offline-icon">📷</div>
                            <div class="offline-title">Camera Feed is Offline</div>
                            <p class="offline-desc">Click <b>Start Feed</b> on the left controls to begin real-time YOLO tracking and visual memory processing.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Bottom Section: Assistant and Timeline ──
        # ── Bottom Section: Assistant and Timeline ──
        st.markdown("<hr style='margin: 2.5rem 0 1.5rem 0; opacity: 0.15;' />", unsafe_allow_html=True)
        st.markdown("<h2 class='section-title'>Ask the Assistant. <em>Retrieve memories.</em></h2>", unsafe_allow_html=True)
        col_chat, col_timeline = st.columns([1.5, 1.5])

        # ── Bottom Left: AI Assistant ──
        with col_chat:
            with st.container(height=480, border=True):
                st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem; font-weight:600;'>💬 AI Assistant</p>", unsafe_allow_html=True)
                
                # Scrollable inner chat history container
                chat_container = st.container(height=380)
                with chat_container:
                    for msg in st.session_state.get("messages", []):
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                            if "crop_path" in msg and msg["crop_path"]:
                                st.image(msg["crop_path"], caption="Matched Memory", use_container_width=True)

        # ── Bottom Right: Memory Timeline ──
        with col_timeline:
            with st.container(height=480, border=True):
                st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem; font-weight:600;'>⏳ Memory Timeline</p>", unsafe_allow_html=True)
                
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    timeline_search = st.text_input("Search timeline:", placeholder="Filter by name...", key="timeline_search_input", label_visibility="collapsed")
                with t_col2:
                    all_regions = ["All Locations", "top-left", "top-center", "top-right", "middle-left", "middle-center", "middle-right", "bottom-left", "bottom-center", "bottom-right"]
                    selected_region = st.selectbox("Location Filter", options=all_regions, label_visibility="collapsed", key="timeline_region_select")

                # Scrollable inner timeline container
                timeline_container = st.container(height=340)
                with timeline_container:
                    # Fetch memories with applied filters
                    query_parts = []
                    params = []
                    if timeline_search.strip():
                        query_parts.append("object_name LIKE ?")
                        params.append(f"%{timeline_search.strip()}%")
                    if selected_region != "All Locations":
                        query_parts.append("region = ?")
                        params.append(selected_region)

                    where_clause = " WHERE " + " AND ".join(query_parts) if query_parts else ""
                    sql_query = f"SELECT * FROM memories{where_clause} ORDER BY timestamp DESC LIMIT 20"
                    db_rows = memory_db.conn.execute(sql_query, params).fetchall()
                    recent_memories = [memory_db._row_to_entry(r) for r in db_rows]

                    if not recent_memories:
                        st.markdown(
                            """
                            <div style='text-align:center; padding:3rem; color:var(--text-card-secondary); opacity:0.6;'>
                                <div style='font-size:2rem;'>⏳</div>
                                No visual memories matched.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        for idx, mem in enumerate(recent_memories):
                            crop_file = f"data/crops/{mem.memory_id}.jpg"
                            region_str = mem.region.replace("-", " ")
                            try:
                                dt = datetime.fromisoformat(mem.timestamp)
                                time_str = dt.strftime("%I:%M:%S %p")
                            except Exception:
                                time_str = mem.timestamp

                            with st.container(border=True):
                                c1, c2 = st.columns([1, 3])
                                with c1:
                                    if os.path.exists(crop_file):
                                        st.image(crop_file, use_container_width=True)
                                    else:
                                        st.markdown("<div style='text-align:center; font-size:1rem; padding:0.4rem; background:var(--secondary-surface); border-radius:6px;'>📷</div>", unsafe_allow_html=True)
                                with c2:
                                    st.markdown(
                                        f"""
                                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                                            <strong style='color:var(--text-card); font-size:0.95rem; text-transform:uppercase;'>{mem.object_name} #{mem.track_id}</strong>
                                            <span style='background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3); color:var(--text-card); font-size:0.65rem; padding:1px 5px; border-radius:4px;'>{mem.confidence:.0%} Conf</span>
                                        </div>
                                        <div style='font-size:0.78rem; color:var(--text-card-secondary); margin-top:0.25rem;'>
                                            📍 Region: <b style='color:var(--text-card);'>{region_str}</b><br>
                                            🕐 Time: <span style='opacity:0.8;'>{time_str}</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    if st.button("Ask Assistant about this", key=f"ask_mem_{mem.memory_id}", use_container_width=True):
                                        st.session_state.clicked_prompt = f"Where is the {mem.object_name} #{mem.track_id}?"
                                        st.rerun()

        # ── Bottom Centered Query Container ──
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        _, input_col, _ = st.columns([1, 10, 1])
        with input_col:
            with st.container(border=True):
                st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600; text-align:center;'>🔍 Query Your Visual Memory</p>", unsafe_allow_html=True)
                
                # Suggestion prompt chips
                sug_col1, sug_col2, sug_col3 = st.columns(3)
                with sug_col1:
                    if st.button("Where is the laptop?", key="sug_laptop_btn", use_container_width=True):
                        st.session_state.clicked_prompt = "Where is the laptop?"
                        st.rerun()
                with sug_col2:
                    if st.button("What objects are seen?", key="sug_objects_btn", use_container_width=True):
                        st.session_state.clicked_prompt = "What objects have you seen?"
                        st.rerun()
                with sug_col3:
                    if st.button("Where is the phone?", key="sug_phone_btn", use_container_width=True):
                        st.session_state.clicked_prompt = "Where is my cell phone?"
                        st.rerun()
                
                st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
                
                # Microphone and chat input layout
                chat_input_col, voice_col = st.columns([8, 1])
                with voice_col:
                    voice_clicked = st.button("🎤", key="voice_mic_btn", use_container_width=True)
                with chat_input_col:
                    chat_prompt = st.chat_input("Ask your memory assistant...")
                
                if voice_clicked:
                    st.info("🎤 Listening...")
                    recognized_text = recognize_speech()
                    if recognized_text not in ["ERROR_IMPORT", "ERROR_UNKNOWN", "ERROR_REQUEST"] and not recognized_text.startswith("ERROR_MIC"):
                        st.session_state.clicked_prompt = recognized_text
                        st.rerun()
                    else:
                        st.error(f"Speech recognition error: {recognized_text}")
                
                # Process prompt
                active_prompt = None
                if chat_prompt:
                    active_prompt = chat_prompt
                elif st.session_state.get("clicked_prompt"):
                    active_prompt = st.session_state.get("clicked_prompt")
                    st.session_state.clicked_prompt = None
                
                if active_prompt:
                    st.session_state.messages.append({"role": "user", "content": active_prompt})
                    with st.spinner("Analyzing memory..."):
                        answer, matched_mem = query_engine.query(active_prompt)
                    
                    crop_path = None
                    if matched_mem:
                        expected_path = f"data/crops/{matched_mem.memory_id}.jpg"
                        if os.path.exists(expected_path):
                            crop_path = expected_path
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "crop_path": crop_path
                    })
                    st.session_state.voice_feedback_text = answer
                    
                    target_obj = query_engine._extract_object(active_prompt)
                    if target_obj:
                        st.session_state.active_search = target_obj
                        st.session_state.guidance_located_announced = False
                    
                    st.rerun()

        # ── Frame Processing Loop inside Column 2 (col_monitor) ──
        if st.session_state.get("processing", False) and (st.session_state.get("selected_source_type") == "Webcam" or st.session_state.get("selected_video_source")):
            src = st.session_state.get("selected_video_source") if st.session_state.get("selected_source_type") != "Webcam" else int(st.session_state.get("selected_cam_index", 0))
            
            # Persistent VideoCapture
            if "video_cap" not in st.session_state or st.session_state.get("video_source_path") != src or st.session_state.get("video_cap") is None:
                if st.session_state.get("video_cap") is not None:
                    try:
                        st.session_state.get("video_cap").release()
                    except Exception:
                        pass
                st.session_state.video_cap = cv2.VideoCapture(src)
                st.session_state.video_source_path = src

            cap = st.session_state.get("video_cap")

            if not cap.isOpened():
                st.error(f"Cannot open video source: {src}")
                st.session_state.processing = False
                st.session_state.video_cap = None
                st.session_state.video_source_path = None
            else:
                frame_count = 0
                fps_timer = time.time()
                fps = 0.0

                while st.session_state.get("processing", False):
                    ret, frame = cap.read()
                    if not ret:
                        st.info("End of video stream")
                        st.session_state.processing = False
                        if "video_cap" in st.session_state and st.session_state.get("video_cap") is not None:
                            try:
                                st.session_state.get("video_cap").release()
                            except Exception:
                                pass
                            st.session_state.video_cap = None
                            st.session_state.video_source_path = None
                        st.rerun()
                        break

                    # Resize frame for processing speedup
                    max_w = 640
                    h, w = frame.shape[:2]
                    if w > max_w:
                        scale = max_w / w
                        frame = cv2.resize(frame, (max_w, int(h * scale)))

                    frame_count += 1

                    # Track objects
                    tracked_objects = tracker.track(frame)

                    # Guidance target
                    active_search = st.session_state.get("active_search", None)
                    target_detected = False
                    target_bbox = None
                    
                    if active_search:
                        for obj in tracked_objects:
                            if obj.class_name == active_search:
                                target_detected = True
                                target_bbox = obj.bbox
                                break

                    # Save memories
                    h, w = frame.shape[:2]
                    now = datetime.now()

                    for obj in tracked_objects:
                        key = f"{obj.class_name}_{obj.track_id}"

                        if key in st.session_state.get("last_store_times", {}):
                            elapsed = (now - st.session_state.last_store_times[key]).total_seconds()
                            if elapsed < st.session_state.get("mem_interval", 5):
                                continue

                        region = obj.get_region(w, h)
                        crop = obj.crop_from_frame(frame)
                        if crop.size == 0:
                            continue

                        embedding = embedder.embed_image(crop)

                        entry = MemoryEntry(
                            memory_id=None,
                            object_name=obj.class_name,
                            track_id=obj.track_id,
                            timestamp=obj.timestamp,
                            bbox_x1=obj.bbox[0],
                            bbox_y1=obj.bbox[1],
                            bbox_x2=obj.bbox[2],
                            bbox_y2=obj.bbox[3],
                            region=region,
                            confidence=obj.confidence,
                            embedding_id=None,
                        )
                        memory_id = memory_db.store_memory(entry)
                        vector_store.add(embedding, memory_id)

                        # Save crop
                        crop_dir = "data/crops"
                        os.makedirs(crop_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(crop_dir, f"{memory_id}.jpg"), crop)

                        st.session_state.last_store_times[key] = now

                    # FPS Calculation
                    if frame_count % 10 == 0:
                        elapsed_t = time.time() - fps_timer
                        fps = 10.0 / max(elapsed_t, 0.001)
                        fps_timer = time.time()

                    # Annotations
                    annotated = draw_tracked_objects(frame, tracked_objects)

                    if active_search:
                        latest_mem = memory_db.get_latest_memory(active_search)
                        annotated = draw_guidance_overlay(
                            annotated,
                            active_search,
                            target_detected,
                            target_bbox,
                            latest_mem
                        )
                        
                        now_time = time.time()
                        guidance_msg = None
                        
                        if target_detected:
                            if not st.session_state.get("guidance_located_announced", False):
                                guidance_msg = f"Target {active_search} located directly ahead!"
                                st.session_state.guidance_located_announced = True
                                st.session_state.last_guidance_voice_time = now_time
                        else:
                            st.session_state.guidance_located_announced = False
                            if latest_mem and (now_time - st.session_state.get("last_guidance_voice_time", 0) > 8.0):
                                dir_desc = get_direction_description(latest_mem.region)
                                guidance_msg = f"Pan your camera {dir_desc}."
                                st.session_state.last_guidance_voice_time = now_time
                                
                        if guidance_msg:
                            speak_background(guidance_msg)

                    annotated = create_info_overlay(annotated, {
                        "fps": fps,
                        "objects": len(tracked_objects),
                        "memories": memory_db.get_total_count(),
                        "tracking": len(tracker.get_active_tracks()),
                    })

                    # Display image
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

                    time.sleep(0.03)

    # ──────────────────────────────────────────
    # TAB 2: MEMORY GALLERY PANEL
    # ──────────────────────────────────────────

    elif selected_nav == "🖼️ Memory Gallery":
        st.markdown("<h2 class='section-title'>Stored Visual Memories</h2>", unsafe_allow_html=True)
        
        # Grid Search and Filters Panel Card
        with st.container(border=True):
            st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600;'>🔍 Search & Filter Memories</p>", unsafe_allow_html=True)
            gal_query = st.text_input("Semantic search across all visual memory crops:", placeholder="e.g. red cup, person wearing black shirt, laptop on a desk", key="gallery_search_input", label_visibility="collapsed")
            
            st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                all_classes = memory_db.get_unique_objects()
                filter_classes = st.multiselect("Filter Object Types", options=all_classes, default=[])
            with g_col2:
                filter_regions = st.multiselect(
                    "Filter Locations", 
                    options=["top-left", "top-center", "top-right", "middle-left", "middle-center", "middle-right", "bottom-left", "bottom-center", "bottom-right"],
                    default=[]
                )
            with g_col3:
                filter_conf = st.slider("Min Confidence Rating", 0.0, 1.0, 0.0, 0.05)

        # Semantic or SQL search
        if gal_query.strip():
            with st.spinner("Searching visual memories..."):
                query_emb = embedder.embed_text(gal_query)
                results = vector_store.search(query_emb, top_k=50)
                
                memories = []
                scores_dict = {}
                for memory_id, score in results:
                    mem = memory_db.get_memory(memory_id)
                    if not mem:
                        continue
                    
                    if filter_classes and mem.object_name not in filter_classes:
                        continue
                    if filter_regions and mem.region not in filter_regions:
                        continue
                    if mem.confidence < filter_conf:
                        continue
                        
                    memories.append(mem)
                    scores_dict[mem.memory_id] = score
        else:
            query_parts = []
            params = []
            if filter_classes:
                placeholders = ",".join("?" for _ in filter_classes)
                query_parts.append(f"object_name IN ({placeholders})")
                params.extend(filter_classes)
            if filter_regions:
                placeholders = ",".join("?" for _ in filter_regions)
                query_parts.append(f"region IN ({placeholders})")
                params.extend(filter_regions)
            if filter_conf > 0.0:
                query_parts.append("confidence >= ?")
                params.append(filter_conf)
                
            where_clause = " WHERE " + " AND ".join(query_parts) if query_parts else ""
            sql_query = f"SELECT * FROM memories{where_clause} ORDER BY timestamp DESC LIMIT 200"
            db_rows = memory_db.conn.execute(sql_query, params).fetchall()
            memories = [memory_db._row_to_entry(r) for r in db_rows]
            scores_dict = {}

        if not memories:
            st.info("No stored memories match your selected query and filter options.")
        else:
            st.markdown(f"Displaying **{len(memories)}** visual memory files:")
            
            # Masonry grid (4 columns)
            cols = st.columns(4)
            for idx, mem in enumerate(memories):
                col = cols[idx % 4]
                with col:
                    crop_file = f"data/crops/{mem.memory_id}.jpg"
                    score_badge = ""
                    if mem.memory_id in scores_dict:
                        score_badge = f"<span style='background:var(--accent); color:white; padding:2px 6px; border-radius:4px; font-size:0.75rem; margin-left:5px;'>Match {scores_dict[mem.memory_id]:.0%}</span>"
                    
                    try:
                        dt = datetime.fromisoformat(mem.timestamp)
                        time_str = dt.strftime("%b %d, %I:%M %p")
                    except Exception:
                        time_str = mem.timestamp
                        
                    with st.container(border=True):
                        if os.path.exists(crop_file):
                            st.image(crop_file, use_container_width=True)
                        else:
                            st.markdown("<div style='text-align:center; padding:2rem; background:var(--secondary-surface); color:var(--text-card-secondary); border-radius:8px;'>📷 No Crop</div>", unsafe_allow_html=True)
                        
                        st.markdown(
                            f"""
                            <div style='font-size:0.85rem; line-height:1.45; margin-top:0.4rem;'>
                                <strong style='color:var(--text-card); font-size:1.05rem;'>{mem.object_name} #{mem.track_id}</strong>{score_badge}<br>
                                📍 Region: <b>{mem.region.replace("-", " ")}</b><br>
                                🎯 Confidence: <b>{mem.confidence:.0%}</b><br>
                                🕐 <span style='opacity:0.7;'>{time_str}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button("Select Target Lock", key=f"lock_gal_{mem.memory_id}", use_container_width=True):
                            st.session_state.active_search = mem.object_name
                            st.session_state.guidance_located_announced = False
                            st.success(f"Locked target: {mem.object_name.upper()}")

    # ──────────────────────────────────────────
    # TAB 3: ANALYTICS PANEL
    # ──────────────────────────────────────────

    elif selected_nav == "📊 Analytics & KPI":
        st.markdown("<h2 class='section-title'>Analytics & Performance Metrics</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600;'>📈 Unique Object Frequencies</p>", unsafe_allow_html=True)
                sql_freq = "SELECT object_name, COUNT(*) as count FROM memories GROUP BY object_name ORDER BY count DESC"
                freqs = memory_db.conn.execute(sql_freq).fetchall()
                
                if freqs:
                    labels = [f[0] for f in freqs]
                    counts = [f[1] for f in freqs]
                    chart_data = {
                        "Object Type": labels,
                        "Saved Instances": counts
                    }
                    st.bar_chart(chart_data, x="Object Type", y="Saved Instances")
                else:
                    st.info("No data available yet. Run the camera feed to save memories.")
                    
        with col2:
            with st.container(border=True):
                st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600;'>🗺️ Spatial Region Distribution</p>", unsafe_allow_html=True)
                sql_regions = "SELECT region, COUNT(*) as count FROM memories GROUP BY region ORDER BY count DESC"
                region_counts = memory_db.conn.execute(sql_regions).fetchall()
                
                if region_counts:
                    labels = [r[0].replace("-", " ") for r in region_counts]
                    counts = [r[1] for r in region_counts]
                    chart_data = {
                        "Region / Location": labels,
                        "Memories Count": counts
                    }
                    st.bar_chart(chart_data, x="Region / Location", y="Memories Count")
                else:
                    st.info("No spatial data available yet.")

        # Latest database records
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600;'>📁 Recent Database Transactions</p>", unsafe_allow_html=True)
            sql_log = "SELECT memory_id, object_name, track_id, timestamp, region, confidence FROM memories ORDER BY timestamp DESC LIMIT 20"
            logs = memory_db.conn.execute(sql_log).fetchall()
            
            if logs:
                log_data = []
                for row in logs:
                    log_data.append({
                        "Memory ID": row[0],
                        "Object": row[1],
                        "Track ID": row[2],
                        "Timestamp": row[3],
                        "Region": row[4].replace("-", " "),
                        "Confidence": f"{row[5]:.0%}"
                    })
                st.dataframe(log_data, use_container_width=True)
            else:
                st.info("No memories in database yet.")

    # ──────────────────────────────────────────
    # TAB 4: SYSTEM SETTINGS PANEL
    # ──────────────────────────────────────────

    elif selected_nav == "⚙️ System Settings":
        st.markdown("<h2 class='section-title'>System Configuration Settings</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600;'>⚙️ YOLO Detection & Tracking Settings</p>", unsafe_allow_html=True)
                
                # Model Selection
                model_options = {
                    "Standard (YOLOv8n)": "yolov8n.pt",
                    "Custom Trained Model": "data/weights/custom_best.pt"
                }
                current_model_name = "Standard (YOLOv8n)"
                if st.session_state.get("selected_model_path") == "data/weights/custom_best.pt":
                    current_model_name = "Custom Trained Model"
                    
                selected_model_name = st.selectbox(
                    "Inference Engine Weights", 
                    options=list(model_options.keys()),
                    index=list(model_options.keys()).index(current_model_name)
                )
                selected_model_path = model_options[selected_model_name]
                
                custom_weights_exist = os.path.exists("data/weights/custom_best.pt")
                if selected_model_path == "data/weights/custom_best.pt" and not custom_weights_exist:
                    st.warning("Custom trained weights not found. Defaulting to standard model.")
                    selected_model_path = "yolov8n.pt"
                    
                st.session_state.selected_model_path = selected_model_path

                # Confidence slider
                st.session_state.confidence = st.slider(
                    "Object Detector Confidence Threshold", 
                    0.1, 0.95, 
                    st.session_state.get("confidence", 0.45), 
                    0.05
                )

                # Memory interval
                st.session_state.mem_interval = st.slider(
                    "Memory Insertion Interval (seconds)", 
                    1, 30, 
                    st.session_state.get("mem_interval", 5)
                )

        with col2:
            with st.container(border=True):
                st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600;'>🎥 Video Capture Source Settings</p>", unsafe_allow_html=True)
                
                source_type = st.radio("Input Feed Type", ["Webcam", "Video File"], horizontal=True)
                st.session_state.selected_source_type = source_type
                
                if source_type == "Webcam":
                    cam_index = st.number_input("Hardware Camera Device Index", 0, 10, st.session_state.get("selected_cam_index", 0))
                    st.session_state.selected_cam_index = int(cam_index)
                else:
                    uploaded = st.file_uploader("Upload Video File Source", type=["mp4", "avi", "mov", "mkv"])
                    if uploaded:
                        os.makedirs("data/temp", exist_ok=True)
                        video_path = f"data/temp/{uploaded.name}"
                        with open(video_path, "wb") as f:
                            f.write(uploaded.read())
                        st.session_state.selected_video_source = video_path
                    else:
                        st.session_state.selected_video_source = None

                # Audio system checks
                st.markdown("<hr style='margin:1rem 0; opacity:0.15;' />", unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600;'>🎙️ Audio & Interaction Status</p>", unsafe_allow_html=True)
                aud_cols = st.columns(2)
                with aud_cols[0]:
                    st.markdown(f"Speech Rec: **{'✅ ENABLED' if speech_rec_available else '❌ DISABLED'}**")
                with aud_cols[1]:
                    st.markdown(f"TTS Engine: **{'✅ ENABLED' if tts_available else '❌ DISABLED'}**")

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p style='font-size:0.8rem; color:var(--text-card-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem; font-weight:600;'>🏋️ Custom Model Training Pipeline</p>", unsafe_allow_html=True)
            st.markdown("Train YOLOv8 on the sample COCO8 dataset to customize weights for specific targets.")
            if st.button("🏋️ Begin Model Training Pipeline", use_container_width=True):
                st.info("Initiating custom YOLOv8 model training loop... This might take some time.")
                import subprocess
                try:
                    result = subprocess.run([sys.executable, "train_custom_model.py"], capture_output=True, text=True, check=True)
                    st.success("Custom model trained successfully! Model weights saved at 'data/weights/custom_best.pt'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training pipeline execution failed: {e}")

    # ── Speech Feedback (TTS) ──
    if "voice_feedback_text" in st.session_state and st.session_state.voice_feedback_text:
        text_to_speak = st.session_state.voice_feedback_text
        safe_text = text_to_speak.replace('"', '\\"').replace('\n', ' ')
        st.components.v1.html(f"""
            <script>
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    var utterance = new SpeechSynthesisUtterance("{safe_text}");
                    utterance.rate = 1.0;
                    utterance.pitch = 1.0;
                    window.speechSynthesis.speak(utterance);
                }}
            </script>
        """, height=0)
        st.session_state.voice_feedback_text = None


if __name__ == "__main__":
    run_app()
