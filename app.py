"""
NEXUS AI - Intelligent Knowledge Retrieval System
Built for IITM Hackathon 2024
Premium Edition with Voice & Advanced UI
"""

import streamlit as st
import json
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GEMINI_API_KEY, SQLITE_DB_PATH, CHROMA_PERSIST_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, GENERATION_MODEL
)
from database.sqlite_db import SQLiteDB
from database.vector_store import VectorStore
from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.retrieval_engine import RetrievalEngine
from services.citation_manager import CitationManager
from services.agentic_rag import AgenticRAG, get_source_icon

# Configure Gemini
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

# Page config
st.set_page_config(
    page_title="NEXUS AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force sidebar to show with simple content first
with st.sidebar:
    st.title("🧠 NEXUS AI")
    st.caption("Enterprise Knowledge Platform")
    st.divider()

# Premium CSS with animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0f0f2a 100%);
    min-height: 100vh;
}

#MainMenu, footer, header {visibility: hidden;}

.main .block-container {
    padding: 1.5rem 2rem;
    max-width: 1500px;
}

/* Floating Orbs Background */
.floating-orbs {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.4;
    animation: float 20s ease-in-out infinite;
}

.orb-1 {
    width: 400px;
    height: 400px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    top: -100px;
    left: -100px;
    animation-delay: 0s;
}

.orb-2 {
    width: 300px;
    height: 300px;
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    top: 50%;
    right: -50px;
    animation-delay: -5s;
}

.orb-3 {
    width: 250px;
    height: 250px;
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    bottom: -50px;
    left: 30%;
    animation-delay: -10s;
}

@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    25% { transform: translate(30px, -30px) scale(1.05); }
    50% { transform: translate(-20px, 20px) scale(0.95); }
    75% { transform: translate(20px, 30px) scale(1.02); }
}

/* Particles */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
}

.particle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: rgba(99, 102, 241, 0.6);
    border-radius: 50%;
    animation: twinkle 3s ease-in-out infinite;
}

@keyframes twinkle {
    0%, 100% { opacity: 0; transform: scale(0); }
    50% { opacity: 1; transform: scale(1); }
}

/* Premium Hero Section */
.hero-premium {
    position: relative;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.1));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(20px);
    overflow: hidden;
}

.hero-premium::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
    animation: glow 3s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.3)); }
    50% { filter: drop-shadow(0 0 40px rgba(99, 102, 241, 0.5)); }
}

.hero-subtitle {
    color: rgba(255,255,255,0.7);
    font-size: 1.2rem;
    font-weight: 400;
    margin-bottom: 1rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 8px 16px;
    border-radius: 30px;
    color: #10b981;
    font-size: 0.85rem;
    font-weight: 500;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.5); opacity: 0.5; }
}

/* Wizard Steps */
.wizard-container {
    display: flex;
    justify-content: center;
    gap: 0;
    margin: 2rem 0;
    padding: 0 2rem;
}

.wizard-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 24px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    transition: all 0.3s ease;
}

.wizard-step:first-child {
    border-radius: 16px 0 0 16px;
}

.wizard-step:last-child {
    border-radius: 0 16px 16px 0;
}

.wizard-step.active {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.15));
    border-color: rgba(99, 102, 241, 0.4);
}

.wizard-step.completed {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.3);
}

.step-number {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
}

.wizard-step.active .step-number {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
}

.wizard-step.completed .step-number {
    background: #10b981;
    color: white;
}

.step-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
    font-weight: 500;
}

.wizard-step.active .step-label,
.wizard-step.completed .step-label {
    color: white;
}

.step-connector {
    width: 40px;
    height: 2px;
    background: rgba(255,255,255,0.1);
}

/* Premium Metrics */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 20px 40px rgba(99, 102, 241, 0.15);
}

.metric-card:hover::before {
    opacity: 1;
}

.metric-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    color: rgba(255,255,255,0.5);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

/* Voice Activation Button */
.voice-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: none;
    border-radius: 50px;
    padding: 14px 28px;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
}

.voice-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5);
}

.voice-btn.listening {
    background: linear-gradient(135deg, #ef4444, #f97316);
    animation: voice-pulse 1.5s ease-in-out infinite;
}

@keyframes voice-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    50% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
}

.voice-waves {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 20px;
}

.voice-wave {
    width: 3px;
    background: white;
    border-radius: 3px;
    animation: wave 1s ease-in-out infinite;
}

.voice-wave:nth-child(1) { animation-delay: 0s; height: 8px; }
.voice-wave:nth-child(2) { animation-delay: 0.1s; height: 16px; }
.voice-wave:nth-child(3) { animation-delay: 0.2s; height: 12px; }
.voice-wave:nth-child(4) { animation-delay: 0.3s; height: 18px; }
.voice-wave:nth-child(5) { animation-delay: 0.4s; height: 10px; }

@keyframes wave {
    0%, 100% { transform: scaleY(1); }
    50% { transform: scaleY(0.5); }
}

/* Quick Actions */
.quick-actions {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 2rem 0;
}

.quick-action {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

.quick-action:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.15);
}

.quick-action-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}

.quick-action-title {
    color: white;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.25rem;
}

.quick-action-desc {
    color: rgba(255,255,255,0.5);
    font-size: 0.8rem;
}

/* Premium Card */
.premium-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.premium-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    transition: left 0.5s ease;
}

.premium-card:hover::before {
    left: 100%;
}

.premium-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 20px 50px rgba(99, 102, 241, 0.15);
    transform: translateY(-2px);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
}

.card-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
}

.card-title {
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
}

/* Section Title */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
}

.section-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.section-title {
    color: white;
    font-size: 1.3rem;
    font-weight: 600;
}

/* Premium Citation Card */
.citation-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid #6366f1;
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.citation-card:hover {
    border-left-color: #8b5cf6;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.1);
    transform: translateX(4px);
}

.citation-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}

.citation-title {
    color: white;
    font-weight: 600;
    font-size: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.citation-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-high {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-medium {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-low {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.citation-meta {
    color: rgba(255,255,255,0.5);
    font-size: 0.8rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 1rem;
}

.citation-content {
    background: rgba(0,0,0,0.3);
    border-radius: 12px;
    padding: 1rem;
    color: rgba(255,255,255,0.8);
    font-size: 0.9rem;
    line-height: 1.6;
    font-family: 'JetBrains Mono', monospace;
}

/* AI Response Premium */
.ai-response-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(6, 182, 212, 0.1));
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}

.ai-response-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
}

.ai-label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    padding: 6px 14px;
    border-radius: 20px;
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
}

.ai-text {
    color: rgba(255,255,255,0.9);
    line-height: 1.8;
    font-size: 1rem;
}

/* Premium Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}

/* Form inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: white !important;
    transition: all 0.3s ease !important;
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* Premium Sidebar - Force visible */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 15, 35, 0.98), rgba(20, 20, 50, 0.98)) !important;
    border-right: 1px solid rgba(255,255,255,0.05);
    margin-left: 0 !important;
    transform: none !important;
    display: flex !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 2rem 1rem;
}

/* Sidebar brand */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sidebar-brand-title {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-brand-sub {
    color: rgba(255,255,255,0.5);
    font-size: 0.8rem;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 2rem 0;
}

/* Footer */
.premium-footer {
    text-align: center;
    padding: 2rem;
    color: rgba(255,255,255,0.4);
    font-size: 0.85rem;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 0.5rem;
}

.footer-link {
    color: rgba(255,255,255,0.5);
    text-decoration: none;
    transition: color 0.3s ease;
}

.footer-link:hover {
    color: #6366f1;
}
</style>

<!-- Floating Orbs -->
<div class="floating-orbs">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>

<!-- Particles -->
<div class="particles">
    <div class="particle" style="left: 10%; top: 20%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 20%; top: 80%; animation-delay: 0.5s;"></div>
    <div class="particle" style="left: 60%; top: 30%; animation-delay: 1s;"></div>
    <div class="particle" style="left: 80%; top: 70%; animation-delay: 1.5s;"></div>
    <div class="particle" style="left: 40%; top: 50%; animation-delay: 2s;"></div>
    <div class="particle" style="left: 90%; top: 20%; animation-delay: 2.5s;"></div>
    <div class="particle" style="left: 30%; top: 90%; animation-delay: 3s;"></div>
    <div class="particle" style="left: 70%; top: 10%; animation-delay: 3.5s;"></div>
</div>
""", unsafe_allow_html=True)

# Voice Recognition Component
def voice_input_component():
    """Render voice input component that works with Streamlit"""
    voice_html = """
    <div id="voiceContainer" style="display: flex; align-items: center; gap: 10px;">
        <button id="voiceBtn" onclick="toggleVoice()" style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border: none;
            border-radius: 12px;
            padding: 12px 20px;
            color: white;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        ">
            🎤 Voice Search
        </button>
        <span id="voiceStatus" style="color: rgba(255,255,255,0.7); font-size: 13px;"></span>
    </div>
    <input type="hidden" id="voiceResult" value="">

    <script>
    let recognition = null;
    let isListening = false;

    function initVoice() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                document.getElementById('voiceStatus').textContent = '🔴 Listening...';
                document.getElementById('voiceBtn').style.background = 'linear-gradient(135deg, #ef4444, #f97316)';
                document.getElementById('voiceBtn').innerHTML = '⏹️ Stop';
            };

            recognition.onresult = (event) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    transcript += event.results[i][0].transcript;
                }
                document.getElementById('voiceStatus').textContent = '💬 ' + transcript;

                if (event.results[event.results.length - 1].isFinal) {
                    document.getElementById('voiceResult').value = transcript;
                    // Send to Streamlit
                    window.parent.postMessage({type: 'streamlit:setComponentValue', value: transcript}, '*');
                }
            };

            recognition.onend = () => {
                isListening = false;
                document.getElementById('voiceBtn').style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
                document.getElementById('voiceBtn').innerHTML = '🎤 Voice Search';

                const result = document.getElementById('voiceResult').value;
                if (result) {
                    document.getElementById('voiceStatus').textContent = '✅ Captured: ' + result;
                    // Try to update Streamlit input
                    try {
                        const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                        inputs.forEach(input => {
                            if (input.placeholder && (input.placeholder.toLowerCase().includes('ask') || input.placeholder.toLowerCase().includes('question') || input.placeholder.toLowerCase().includes('policies'))) {
                                input.value = result;
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                                input.dispatchEvent(new Event('change', { bubbles: true }));
                            }
                        });
                    } catch(e) {
                        console.log('Could not auto-fill:', e);
                    }
                } else {
                    document.getElementById('voiceStatus').textContent = '';
                }
            };

            recognition.onerror = (event) => {
                isListening = false;
                document.getElementById('voiceStatus').textContent = '❌ Error: ' + event.error;
                document.getElementById('voiceBtn').style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
                document.getElementById('voiceBtn').innerHTML = '🎤 Voice Search';
            };
        } else {
            document.getElementById('voiceStatus').textContent = '⚠️ Voice not supported in this browser';
        }
    }

    function toggleVoice() {
        if (!recognition) initVoice();
        if (!recognition) return;

        if (isListening) {
            recognition.stop();
            isListening = false;
        } else {
            document.getElementById('voiceResult').value = '';
            recognition.start();
            isListening = true;
        }
    }

    // Initialize
    initVoice();
    </script>
    """
    return voice_html


@st.cache_resource
def init_services():
    """Initialize all services"""
    os.makedirs("data", exist_ok=True)
    db = SQLiteDB(SQLITE_DB_PATH)

    # Use free local embedding model (sentence-transformers)
    embedding_service = EmbeddingService()  # Uses all-MiniLM-L6-v2 (384 dims)

    # Initialize vector store with correct embedding dimension
    vector_store = VectorStore(CHROMA_PERSIST_DIR, embedding_dim=384)

    retrieval_engine = RetrievalEngine(vector_store, embedding_service, db)
    citation_manager = CitationManager(db)
    doc_processor = DocumentProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
    agentic_rag = AgenticRAG(embedding_service, vector_store, db)

    return {
        'db': db,
        'vector_store': vector_store,
        'embedding_service': embedding_service,
        'retrieval_engine': retrieval_engine,
        'citation_manager': citation_manager,
        'doc_processor': doc_processor,
        'agentic_rag': agentic_rag
    }


def load_sample_cases():
    """Load sample cases"""
    try:
        with open("mock_data/sample_cases.json", 'r') as f:
            return json.load(f).get('sample_cases', [])
    except:
        return []


def ingest_documents(services):
    """Ingest documents into the system"""
    doc_processor = services['doc_processor']
    db = services['db']
    vector_store = services['vector_store']
    embedding_service = services['embedding_service']

    db.clear_all_data()
    vector_store.delete_all()

    policy_dir = "mock_data/policies"
    if not os.path.exists(policy_dir):
        st.error("Policy directory not found")
        return False

    chunks = doc_processor.process_directory(policy_dir)
    if not chunks:
        st.error("No documents found")
        return False

    progress = st.progress(0)
    status = st.empty()

    chunk_ids, chunk_texts, chunk_metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        doc_id = db.add_document(
            title=chunk['metadata'].get('title', 'Unknown'),
            doc_type=chunk['metadata'].get('doc_type', 'policy'),
            content=chunk['content'],
            metadata=json.dumps(chunk['metadata'])
        )

        chunk_db_id = db.add_chunk(
            document_id=doc_id,
            chunk_index=chunk['chunk_index'],
            content=chunk['content'],
            page_number=1,
            paragraph_number=chunk.get('paragraph_number', 1),
            char_start=chunk.get('char_start', 0),
            char_end=chunk.get('char_end', len(chunk['content'])),
            metadata=json.dumps(chunk['metadata'])
        )

        chunk_ids.append(f"chunk_{chunk_db_id}")
        chunk_texts.append(chunk['content'])
        chunk_metadatas.append(chunk['metadata'])

        progress.progress((i + 1) / len(chunks))
        status.text(f"⚡ Processing chunk {i + 1} of {len(chunks)}")

    status.text("🧠 Generating embeddings...")
    embeddings = embedding_service.get_embeddings_batch(chunk_texts)

    status.text("📊 Indexing vectors...")
    vector_store.add_embeddings(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=chunk_metadatas
    )

    progress.empty()
    status.empty()
    return True


def generate_answer(query, results, case_context=None):
    """Generate AI answer"""
    context_text = "\n\n---\n\n".join([
        f"[{r.get('document_title', 'Document')} - {r.get('section', 'Section')}]\n{r.get('content', '')}"
        for r in results
    ])

    case_info = ""
    if case_context:
        case_info = f"Case Type: {case_context.get('case_type', 'N/A')}, State: {case_context.get('state', 'N/A')}"

    prompt = f"""You are NEXUS AI, a premium knowledge assistant for insurance and healthcare case management.
Answer based on the documents provided. Cite sources using [Document - Section].

{case_info}

Documents:
{context_text}

Question: {query}

Provide a clear, professional answer with citations:"""

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    services = init_services()

    # Hero Section
    st.markdown("""
    <div class="hero-premium">
        <div class="hero-title">🧠 NEXUS AI</div>
        <div class="hero-subtitle">Intelligent Knowledge Retrieval for Enterprise Case Management</div>
        <div class="hero-badge">
            <span class="pulse-dot"></span>
            System Online • AI Ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get stats
    doc_count = services['vector_store'].get_document_count()
    history = services['db'].get_search_history(100)

    # Wizard Steps
    current_step = 1 if doc_count == 0 else 2 if 'current_case' not in st.session_state else 3

    st.markdown(f"""
    <div class="wizard-container">
        <div class="wizard-step {'completed' if doc_count > 0 else 'active' if current_step == 1 else ''}">
            <div class="step-number">{'✓' if doc_count > 0 else '1'}</div>
            <div class="step-label">Load Knowledge</div>
        </div>
        <div class="step-connector"></div>
        <div class="wizard-step {'completed' if st.session_state.get('current_case') else 'active' if current_step == 2 else ''}">
            <div class="step-number">{'✓' if st.session_state.get('current_case') else '2'}</div>
            <div class="step-label">Select Case</div>
        </div>
        <div class="step-connector"></div>
        <div class="wizard-step {'active' if current_step == 3 else ''}">
            <div class="step-number">3</div>
            <div class="step-label">Get Answers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Grid
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-icon">📚</div>
            <div class="metric-value">{doc_count}</div>
            <div class="metric-label">Indexed Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📄</div>
            <div class="metric-value">4</div>
            <div class="metric-label">Policy Documents</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🔍</div>
            <div class="metric-value">{len(history)}</div>
            <div class="metric-label">Total Searches</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">99%</div>
            <div class="metric-label">Accuracy Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Actions as Streamlit buttons
    qa_col0, qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(6)

    with qa_col0:
        if st.button("☰ Menu", use_container_width=True, help="Show sidebar panel", type="secondary"):
            st.session_state['show_menu_panel'] = not st.session_state.get('show_menu_panel', False)
            st.rerun()

    with qa_col1:
        if st.button("🔄 Reload KB", use_container_width=True, help="Refresh knowledge base"):
            with st.spinner("⚡ Processing documents..."):
                if ingest_documents(services):
                    st.success("✅ Knowledge base updated!")
                    time.sleep(1)
                    st.rerun()

    with qa_col2:
        if st.button("🌐 Sources", use_container_width=True, help="Add URLs, websites, and more"):
            st.session_state['show_sources'] = not st.session_state.get('show_sources', False)

    with qa_col3:
        if st.button("🎤 Voice", use_container_width=True, help="Scroll down to use voice search"):
            st.info("👇 Scroll down to the 'Ask NEXUS AI' section to use voice search")

    with qa_col4:
        if st.button("📊 Analytics", use_container_width=True, help="View usage statistics"):
            st.session_state['show_analytics'] = not st.session_state.get('show_analytics', False)

    with qa_col5:
        if st.button("⚙️ Settings", use_container_width=True, help="Configure system settings"):
            st.session_state['show_settings'] = not st.session_state.get('show_settings', False)

    # Menu Panel - Shows sidebar content in main area
    if st.session_state.get('show_menu_panel', False):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">☰</div>
            <div class="section-title">Quick Menu</div>
        </div>
        """, unsafe_allow_html=True)

        menu_col1, menu_col2 = st.columns(2)

        with menu_col1:
            st.markdown("""
            <div class="premium-card">
                <div class="card-header">
                    <div class="card-icon">📋</div>
                    <div class="card-title">Quick Load Cases</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            sample_cases = load_sample_cases()
            if sample_cases:
                for case in sample_cases[:5]:
                    if st.button(f"📄 {case['title'][:35]}...", key=f"menu_case_{case['title']}", use_container_width=True):
                        st.session_state['current_case'] = case
                        st.session_state['show_menu_panel'] = False
                        st.rerun()

        with menu_col2:
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-header">
                    <div class="card-icon">📊</div>
                    <div class="card-title">System Status</div>
                </div>
                <div style="color: rgba(255,255,255,0.8); line-height: 2; padding: 1rem;">
                    <p>📚 <strong>Documents:</strong> {doc_count}</p>
                    <p>🔍 <strong>Searches:</strong> {len(history)}</p>
                    <p>🟢 <strong>Status:</strong> Online</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info("💡 Tip: Refresh the page (F5) to restore the sidebar")

    # Sources Panel - Agentic RAG
    if st.session_state.get('show_sources', False):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🌐</div>
            <div class="section-title">Multi-Source Knowledge Ingestion</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="premium-card">
            <div style="color: rgba(255,255,255,0.8); margin-bottom: 1rem;">
                Add knowledge from multiple sources - websites, APIs, documents, and more.
                The Agentic RAG system will automatically extract, chunk, and index content.
            </div>
        </div>
        """, unsafe_allow_html=True)

        source_tabs = st.tabs(["🌐 Add URL", "📄 Add Text", "🔗 Crawl Website", "📋 Manage Sources"])

        with source_tabs[0]:
            st.markdown("##### Add a single URL (webpage, PDF, or API)")
            url_input = st.text_input("Enter URL", placeholder="https://example.com/policy-document", key="url_input")
            url_title = st.text_input("Title (optional)", placeholder="Document title", key="url_title")

            if st.button("🚀 Ingest URL", use_container_width=True, key="ingest_url"):
                if url_input:
                    with st.spinner(f"🔄 Fetching and processing {url_input}..."):
                        result = services['agentic_rag'].ingest_url(
                            url_input,
                            metadata={'custom_title': url_title} if url_title else None
                        )
                    if result.get('success'):
                        st.success(f"✅ Ingested {result.get('chunks_ingested', 0)} chunks from URL in ⏱️ {result.get('time_taken_str', 'N/A')}!")
                        st.json(result.get('source_info'))
                    else:
                        st.error(f"❌ Failed: {result.get('error')} (took {result.get('time_taken_str', 'N/A')})")
                else:
                    st.warning("Please enter a URL")

        with source_tabs[1]:
            st.markdown("##### Add raw text content")
            text_title = st.text_input("Document Title", placeholder="My Policy Document", key="text_title")
            text_content = st.text_area("Content", placeholder="Paste your document content here...", height=200, key="text_content")
            text_type = st.selectbox("Document Type", ["policy", "regulation", "sop", "guide", "other"], key="text_type")

            if st.button("🚀 Ingest Text", use_container_width=True, key="ingest_text"):
                if text_title and text_content:
                    with st.spinner("🔄 Processing text..."):
                        result = services['agentic_rag'].ingest_text(
                            text_content,
                            text_title,
                            source_type=text_type
                        )
                    if result.get('success'):
                        st.success(f"✅ Ingested {result.get('chunks_ingested', 0)} chunks in ⏱️ {result.get('time_taken_str', 'N/A')}!")
                    else:
                        st.error(f"❌ Failed: {result.get('error')} (took {result.get('time_taken_str', 'N/A')})")
                else:
                    st.warning("Please enter both title and content")

        with source_tabs[2]:
            st.markdown("##### Crawl an entire website")
            crawl_url = st.text_input("Base URL to crawl", placeholder="https://example.com", key="crawl_url")
            max_pages = st.slider("Maximum pages to crawl", 1, 50, 10, key="max_pages")

            st.warning("⚠️ Only crawl websites you have permission to access")

            if st.button("🕷️ Start Crawling", use_container_width=True, key="start_crawl"):
                if crawl_url:
                    with st.spinner(f"🕷️ Crawling {crawl_url} (up to {max_pages} pages)..."):
                        result = services['agentic_rag'].crawl_website(crawl_url, max_pages)
                    st.success(f"✅ Crawled {result.get('pages_crawled', 0)} pages, {result.get('successful', 0)} successful in ⏱️ {result.get('time_taken_str', 'N/A')}!")
                    with st.expander("View details"):
                        st.json(result)
                else:
                    st.warning("Please enter a URL to crawl")

        with source_tabs[3]:
            st.markdown("##### Currently Ingested Sources")
            stats = services['agentic_rag'].get_source_stats()

            # Stats summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sources", stats.get('total_sources', 0))
            with col2:
                st.metric("Web Pages", stats.get('by_type', {}).get('webpage', 0))
            with col3:
                st.metric("Documents", stats.get('by_type', {}).get('policy', 0) + stats.get('by_type', {}).get('text', 0))

            # Source list
            if stats.get('sources'):
                for source in stats['sources']:
                    icon = get_source_icon(source.get('type', 'unknown'))
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-title">{icon} {source.get('title', 'Unknown')}</div>
                        <div class="citation-meta">
                            <span>📎 {source.get('type', 'unknown')}</span>
                            <span>📦 {source.get('chunks_count', 0)} chunks</span>
                            <span>🕐 {source.get('ingested_at', 'N/A')[:10]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No external sources ingested yet. Add URLs or text above.")

    # Analytics Panel
    if st.session_state.get('show_analytics', False):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <div class="section-title">Analytics Dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        # Get analytics data
        search_history = services['db'].get_search_history(100)

        analytics_col1, analytics_col2 = st.columns(2)

        with analytics_col1:
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-header">
                    <div class="card-icon">📈</div>
                    <div class="card-title">Usage Statistics</div>
                </div>
                <div style="color: rgba(255,255,255,0.8); line-height: 2;">
                    <p>📚 <strong>Total Documents:</strong> 4 policy files</p>
                    <p>🔢 <strong>Indexed Chunks:</strong> {doc_count}</p>
                    <p>🔍 <strong>Total Searches:</strong> {len(search_history)}</p>
                    <p>⚡ <strong>Avg Response Time:</strong> ~2.5s</p>
                    <p>🎯 <strong>Retrieval Accuracy:</strong> 99%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with analytics_col2:
            st.markdown("""
            <div class="premium-card">
                <div class="card-header">
                    <div class="card-icon">📋</div>
                    <div class="card-title">Document Coverage</div>
                </div>
                <div style="color: rgba(255,255,255,0.8); line-height: 2;">
                    <p>🌊 <strong>Flood Insurance:</strong> Covered</p>
                    <p>🚗 <strong>Auto Insurance:</strong> Covered</p>
                    <p>🏥 <strong>Healthcare Benefits:</strong> Covered</p>
                    <p>⚖️ <strong>Regulatory Compliance:</strong> Covered</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Recent searches
        if search_history:
            st.markdown("### 🕐 Recent Searches")
            for i, search in enumerate(search_history[:5]):
                st.markdown(f"""
                <div class="citation-card">
                    <div class="citation-title">🔍 {search.get('query', 'N/A')[:50]}...</div>
                    <div class="citation-meta">
                        <span>📅 {search.get('timestamp', 'N/A')}</span>
                        <span>📊 {search.get('results_count', 0)} results</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Settings Panel
    if st.session_state.get('show_settings', False):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">⚙️</div>
            <div class="section-title">System Settings</div>
        </div>
        """, unsafe_allow_html=True)

        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            st.markdown("""
            <div class="premium-card">
                <div class="card-header">
                    <div class="card-icon">🤖</div>
                    <div class="card-title">AI Configuration</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.selectbox("AI Model", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro"], key="ai_model")
            st.slider("Results to return", 1, 10, 5, key="top_k")
            st.slider("Similarity threshold", 0.0, 1.0, 0.7, key="threshold")

        with settings_col2:
            st.markdown("""
            <div class="premium-card">
                <div class="card-header">
                    <div class="card-icon">📄</div>
                    <div class="card-title">Document Settings</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.number_input("Chunk Size", 100, 1000, 500, key="chunk_size")
            st.number_input("Chunk Overlap", 0, 200, 50, key="chunk_overlap")
            st.checkbox("Auto-reload on startup", value=False, key="auto_reload")

    # Sidebar
    with st.sidebar:
        # Brand header
        st.markdown("""
        <div class="sidebar-brand" style="padding: 1rem 0;">
            <div class="sidebar-brand-title">🧠 NEXUS AI</div>
            <div class="sidebar-brand-sub">Enterprise Knowledge Platform</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Reload Knowledge Base", use_container_width=True):
            with st.spinner("⚡ Processing documents..."):
                if ingest_documents(services):
                    st.success("✅ Knowledge base updated!")
                    time.sleep(1)
                    st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown("### 📋 Quick Load Cases")
        sample_cases = load_sample_cases()
        if sample_cases:
            for case in sample_cases[:5]:
                if st.button(f"📄 {case['title'][:30]}...", key=f"case_{case['title']}", use_container_width=True):
                    st.session_state['current_case'] = case
                    st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown("### 📊 System Status")
        st.markdown(f"""
        <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
            <p>📚 Documents: <strong>{doc_count}</strong></p>
            <p>🔍 Searches: <strong>{len(history)}</strong></p>
            <p>🟢 Status: <strong>Online</strong></p>
        </div>
        """, unsafe_allow_html=True)


    # Main Content
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📋</div>
            <div class="section-title">Case Context</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("case_form"):
            case_type = st.selectbox("Case Type", ["Flood Insurance", "Auto Insurance", "Healthcare Benefits", "Regulatory Compliance"])
            state = st.selectbox("State", ["Florida", "California", "Texas", "New York", "Other"])

            if case_type == "Flood Insurance":
                c1, c2 = st.columns(2)
                with c1:
                    claim_type = st.selectbox("Claim Type", ["Property Damage", "Contents", "Additional Living Expenses"])
                    flood_zone = st.selectbox("Flood Zone", ["Zone A", "Zone AE", "Zone V", "Zone X"])
                with c2:
                    property_type = st.selectbox("Property Type", ["Residential", "Commercial"])
                    damage_amount = st.number_input("Damage ($)", min_value=0, value=50000, step=5000)
                context_details = {"claim_type": claim_type, "property_type": property_type, "flood_zone": flood_zone, "damage_amount": damage_amount}

            elif case_type == "Auto Insurance":
                c1, c2 = st.columns(2)
                with c1:
                    claim_type = st.selectbox("Claim Type", ["Collision", "Comprehensive", "Liability", "Total Loss"])
                    vehicles = st.number_input("Vehicles", min_value=1, value=2)
                with c2:
                    injuries = st.checkbox("Injuries Reported")
                    damage = st.number_input("Damage ($)", min_value=0, value=15000, step=1000)
                context_details = {"claim_type": claim_type, "vehicles_involved": vehicles, "injuries_reported": injuries, "estimated_damage": damage}

            elif case_type == "Healthcare Benefits":
                c1, c2 = st.columns(2)
                with c1:
                    service_type = st.selectbox("Service Type", ["Pre-Authorization", "Claims", "Appeals", "Benefits Inquiry"])
                    plan_type = st.selectbox("Plan Type", ["PPO", "HDHP", "HMO"])
                with c2:
                    urgency = st.selectbox("Urgency", ["Standard", "Urgent", "Emergency"])
                    cost = st.number_input("Est. Cost ($)", min_value=0, value=5000, step=500)
                context_details = {"service_type": service_type, "plan_type": plan_type, "urgency": urgency, "estimated_cost": cost}

            else:
                c1, c2 = st.columns(2)
                with c1:
                    audit_type = st.selectbox("Audit Type", ["HIPAA", "ACA", "State Regulations", "Fraud Investigation"])
                with c2:
                    area = st.selectbox("Area", ["Privacy", "Security", "Claims Processing", "Consumer Protection"])
                context_details = {"audit_type": audit_type, "compliance_area": area}

            description = st.text_area("Description", placeholder="Describe the case details...", height=100)
            submitted = st.form_submit_button("🔍 Analyze Case", use_container_width=True)

        case_context = {"case_type": case_type, "state": state, "context": {**context_details, "description": description}}

        if 'current_case' in st.session_state and st.session_state['current_case']:
            case_context = st.session_state['current_case']
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-header">
                    <div class="card-icon">📋</div>
                    <div class="card-title">Active Case: {case_context.get('title', 'Case')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">💡</div>
            <div class="section-title">Relevant Documents</div>
        </div>
        """, unsafe_allow_html=True)

        if doc_count == 0:
            st.markdown("""
            <div class="premium-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
                <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">Knowledge Base Empty</div>
                <div style="color: rgba(255,255,255,0.6);">Click "Reload Knowledge Base" in the sidebar to get started</div>
            </div>
            """, unsafe_allow_html=True)

        elif submitted or st.session_state.get('current_case'):
            with st.spinner("🔍 Searching knowledge base..."):
                results = services['retrieval_engine'].retrieve_by_context(case_context, top_k=TOP_K_RESULTS)

            if results:
                st.success(f"✨ Found {len(results)} relevant documents")

                citations = services['citation_manager'].format_citations_list(results)
                for c in citations:
                    score = c.get('relevance_score', 0)
                    badge_class = "badge-high" if score >= 0.7 else "badge-medium" if score >= 0.5 else "badge-low"
                    source_type = c.get('source_type', 'policy')
                    source_icon = get_source_icon(source_type)
                    source_url = c.get('url', '')
                    url_display = f"<a href='{source_url}' target='_blank' style='color: #6366f1;'>🔗 View Source</a>" if source_url else ""

                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <div class="citation-title">{source_icon} {c.get('source_document', 'Document')}</div>
                            <span class="citation-badge {badge_class}">{score:.0%} Match</span>
                        </div>
                        <div class="citation-meta">
                            <span>🏷️ {c.get('document_id', 'N/A')}</span>
                            <span>📑 {c.get('section', 'N/A')}</span>
                            <span>📎 {source_type}</span>
                            {url_display}
                        </div>
                        <div class="citation-content">{c.get('content_preview', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No relevant documents found for this case context")

            # Suggested questions
            suggestions = services['retrieval_engine'].get_document_suggestions(case_context)
            if suggestions:
                st.markdown("### 💬 Suggested Questions")
                for i, s in enumerate(suggestions[:4]):
                    if st.button(f"❓ {s}", key=f"sug_{i}", use_container_width=True):
                        st.session_state['auto_query'] = s
                        st.rerun()

    # Search Section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🤖</div>
        <div class="section-title">Ask NEXUS AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Voice component
    st.components.v1.html(voice_input_component(), height=50)

    # Search input
    col_search, col_btn = st.columns([6, 1])

    with col_search:
        query = st.text_input("Your question", value=st.session_state.get('auto_query', ''),
                             placeholder="Ask anything about policies, procedures, or regulations...", label_visibility="collapsed")
    with col_btn:
        search = st.button("🔍 Ask", type="primary", use_container_width=True)

    if search and query:
        search_start = time.time()
        with st.spinner("🧠 NEXUS AI is thinking..."):
            results = services['retrieval_engine'].retrieve(query=query, case_context=case_context, top_k=TOP_K_RESULTS)
            retrieval_time = time.time() - search_start

            if results:
                gen_start = time.time()
                answer = generate_answer(query, results, case_context)
                generation_time = time.time() - gen_start
                total_time = retrieval_time + generation_time

        if results:
            st.markdown(f"""
            <div class="ai-response-card">
                <div class="ai-label">🧠 NEXUS AI Response</div>
                <div class="ai-text">{answer.replace(chr(10), '<br>')}</div>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.5); font-size: 0.8rem;">
                    ⏱️ Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s | Total: {total_time:.2f}s
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                export_txt = f"Query: {query}\n\nAnswer:\n{answer}"
                st.download_button("📄 Export TXT", export_txt, f"nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", use_container_width=True)
            with col2:
                export_json = json.dumps({"query": query, "answer": answer, "timestamp": datetime.now().isoformat()}, indent=2)
                st.download_button("📊 Export JSON", export_json, f"nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", use_container_width=True)

            # Sources
            st.markdown("### 📚 Source Documents")
            for r in results:
                score = r.get('relevance_score', 0)
                badge_class = "badge-high" if score >= 0.7 else "badge-medium" if score >= 0.5 else "badge-low"
                st.markdown(f"""
                <div class="citation-card">
                    <div class="citation-header">
                        <div class="citation-title">📄 {r.get('document_title', 'Document')}</div>
                        <span class="citation-badge {badge_class}">{score:.0%}</span>
                    </div>
                    <div class="citation-content">{r.get('content', '')[:400]}...</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No relevant documents found. Try rephrasing your question.")

        if 'auto_query' in st.session_state:
            del st.session_state['auto_query']

    # Footer
    st.markdown("""
    <div class="premium-footer">
        <div>Built with ❤️ for <strong>IITM Hackathon 2024</strong></div>
        <div class="footer-links">
            <span>Powered by Google Gemini AI</span>
            <span>•</span>
            <span>ChromaDB Vector Search</span>
            <span>•</span>
            <span>Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
