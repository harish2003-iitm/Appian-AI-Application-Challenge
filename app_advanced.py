"""
NEXUS AI - Advanced Intelligent Knowledge Retrieval System
Built for IITM Hackathon 2024
Enterprise Edition with 4-Layer Architecture
"""

import streamlit as st
import json
import os
import sys
import time
import re
from datetime import datetime

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

from services.intelligent_ingestion import IntelligentIngestionPipeline
from services.hybrid_search import HybridSearchEngine, BM25Index
from services.ontology import OntologyStore, Policy, Claim, Provider, PolicyType, ClaimStatus
from services.query_engine import QueryExpander, CitationAwareGenerator
from services.guardrails import GuardrailsPipeline
from services.audit_logger import ImmutableAuditLogger

# Graph RAG and Agentic RAG imports
from services.graph_rag import GraphRAGEngine, KnowledgeGraph
from services.agentic_rag_langchain import AgenticRAGEngine, WebContentIndexer, LANGCHAIN_AVAILABLE
from services.query_analyzer import IntelligentQueryAnalyzer
from services.report_formatter import ConsultingReportFormatter, ReportExporter

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)


def extract_page_from_content(content: str, metadata_page: int = 1) -> int:
    """Extract actual page number from content text (e.g., '--- Page 20 ---')
    Falls back to metadata page if no marker found."""
    match = re.search(r'---\s*Page\s+(\d+)\s*---', content)
    if match:
        return int(match.group(1))
    return metadata_page


def content_similarity(text1: str, text2: str) -> float:
    """Calculate simple Jaccard similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def deduplicate_results(results: list, similarity_threshold: float = 0.85) -> list:
    """Remove near-duplicate results based on content similarity."""
    if not results:
        return results

    unique = []
    for r in results:
        content = r.get('content', '')
        is_duplicate = False

        for existing in unique:
            existing_content = existing.get('content', '')
            if content_similarity(content, existing_content) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(r)

    return unique


st.set_page_config(
    page_title="NEXUS AI - Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS with proper star rating and bounding boxes
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0f0f2a 100%);
    min-height: 100vh;
}

#MainMenu, footer, header {visibility: hidden;}

.main .block-container {
    padding: 1.5rem 2rem;
    max-width: 1600px;
}

/* Hero Section */
.hero-premium {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, #a5b4fc, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: rgba(255,255,255,0.7);
    font-size: 1.1rem;
}

/* Layer Badge */
.layer-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 4px;
}

.layer-1 { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
.layer-2 { background: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
.layer-3 { background: rgba(168, 85, 247, 0.2); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); }
.layer-4 { background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.3); }

/* Loading Timer */
.loading-timer {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.timer-value {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #fff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.timer-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    margin-top: 4px;
}

/* Document Viewer with Real Bounding Boxes */
.doc-viewer-container {
    background: rgba(15, 15, 35, 0.98);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.doc-viewer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    color: white;
}

.doc-content-wrapper {
    position: relative;
    background: #ffffff;
    border-radius: 8px;
    padding: 20px;
    min-height: 200px;
}

.doc-text {
    color: #1a1a2e;
    font-family: 'Georgia', serif;
    font-size: 14px;
    line-height: 1.8;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* Bounding Box Highlight - Yellow box around matched text */
.bbox-match {
    background: linear-gradient(135deg, rgba(250, 204, 21, 0.5), rgba(251, 191, 36, 0.4));
    border: 2px solid #f59e0b;
    border-radius: 3px;
    padding: 1px 4px;
    margin: 0 1px;
    display: inline;
    box-shadow: 0 0 8px rgba(250, 204, 21, 0.4);
}

.bbox-info {
    background: rgba(250, 204, 21, 0.15);
    border: 1px solid rgba(250, 204, 21, 0.3);
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #fbbf24;
}

.match-count-badge {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Premium Cards */
.premium-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.premium-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

/* Metrics Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: rgba(255,255,255,0.5);
    font-size: 0.75rem;
    text-transform: uppercase;
}

/* AI Response */
.ai-response-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(6, 182, 212, 0.1));
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    position: relative;
}

/* Query Expansion Panel */
.query-expansion {
    background: rgba(168, 85, 247, 0.1);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.sub-query-chip {
    display: inline-block;
    background: rgba(168, 85, 247, 0.2);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 20px;
    padding: 4px 12px;
    margin: 4px;
    font-size: 0.8rem;
    color: #a855f7;
}

/* Guardrails Panel */
.guardrails-panel {
    background: rgba(249, 115, 22, 0.1);
    border: 1px solid rgba(249, 115, 22, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.guardrail-pass { color: #22c55e; }
.guardrail-fail { color: #ef4444; }

/* Audit Log */
.audit-entry {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.8);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 15, 35, 0.98), rgba(20, 20, 50, 0.98)) !important;
}

/* Star Rating - Single Bar */
.star-rating-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1.5rem;
}

.star-rating-title {
    color: white;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.star-bar {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 0.5rem;
}

.star-item {
    font-size: 2rem;
    cursor: pointer;
    transition: all 0.2s ease;
    opacity: 0.3;
}

.star-item.filled {
    opacity: 1;
    text-shadow: 0 0 10px rgba(250, 204, 21, 0.5);
}

.star-item:hover {
    transform: scale(1.2);
}

.rating-text {
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


def find_query_matches(content: str, query: str) -> list:
    """Find positions of query terms in content"""
    matches = []
    query_terms = [t.lower() for t in query.split() if len(t) > 3]

    for term in query_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for match in pattern.finditer(content):
            matches.append({
                'term': term,
                'start': match.start(),
                'end': match.end(),
                'matched_text': match.group()
            })

    # Remove overlapping matches
    matches.sort(key=lambda x: x['start'])
    filtered = []
    last_end = -1
    for m in matches:
        if m['start'] >= last_end:
            filtered.append(m)
            last_end = m['end']

    return filtered


def highlight_content_with_bbox(content: str, query: str) -> tuple:
    """Highlight content and return bbox info"""
    matches = find_query_matches(content, query)

    if not matches:
        return content, []

    # Build highlighted content (from end to start to preserve positions)
    highlighted = content
    bbox_list = []

    for match in reversed(matches):
        start, end = match['start'], match['end']
        original = highlighted[start:end]

        # Calculate approximate bbox coordinates
        lines_before = highlighted[:start].count('\n')
        char_in_line = start - highlighted[:start].rfind('\n') - 1

        bbox = {
            'term': match['term'],
            'matched_text': original,
            'x': max(0, char_in_line * 8),  # ~8px per char
            'y': lines_before * 24,  # ~24px per line
            'width': len(original) * 8,
            'height': 20,
            'char_start': start,
            'char_end': end
        }
        bbox_list.append(bbox)

        # Wrap with highlight span
        highlighted = (
            highlighted[:start] +
            f'<span class="bbox-match">{original}</span>' +
            highlighted[end:]
        )

    bbox_list.reverse()
    return highlighted, bbox_list


@st.cache_resource
def init_services():
    """Initialize all services"""
    os.makedirs("data", exist_ok=True)

    db = SQLiteDB(SQLITE_DB_PATH)

    # Use free local embedding model (sentence-transformers)
    # No API key needed - runs completely locally
    # Note: First load may take 30-60 seconds to download the model
    embedding_service = EmbeddingService()  # Uses all-MiniLM-L6-v2 (384 dims)

    # Initialize vector store with correct embedding dimension
    vector_store = VectorStore(CHROMA_PERSIST_DIR, embedding_dim=384)

    retrieval_engine = RetrievalEngine(vector_store, embedding_service, db)
    citation_manager = CitationManager(db)
    doc_processor = DocumentProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
    agentic_rag = AgenticRAG(embedding_service, vector_store, db)

    ingestion_pipeline = IntelligentIngestionPipeline(GEMINI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP)
    hybrid_search = HybridSearchEngine(vector_store, embedding_service, alpha=0.6)
    ontology = OntologyStore()
    query_expander = QueryExpander(GEMINI_API_KEY)
    citation_generator = CitationAwareGenerator(GEMINI_API_KEY, GENERATION_MODEL)
    guardrails = GuardrailsPipeline(GEMINI_API_KEY, enable_hallucination_check=False)
    audit_logger = ImmutableAuditLogger()

    # Graph RAG with fallback models
    from config import GENERATION_MODELS
    graph_rag = GraphRAGEngine(GEMINI_API_KEY, vector_store, embedding_service, model_list=GENERATION_MODELS)

    # Agentic RAG with LangChain and fallback models
    agentic_rag_lc = AgenticRAGEngine(
        api_key=GEMINI_API_KEY,
        vector_store=vector_store,
        embedding_service=embedding_service,
        retrieval_engine=retrieval_engine,
        graph_rag=graph_rag,
        model_list=GENERATION_MODELS
    )

    # Web Content Indexer
    web_indexer = WebContentIndexer(vector_store, embedding_service)

    # Query Analyzer for intelligent RAG mode selection
    query_analyzer = IntelligentQueryAnalyzer()

    # Report Formatter for professional consulting reports
    report_formatter = ConsultingReportFormatter()

    return {
        'db': db,
        'vector_store': vector_store,
        'embedding_service': embedding_service,
        'retrieval_engine': retrieval_engine,
        'citation_manager': citation_manager,
        'doc_processor': doc_processor,
        'agentic_rag': agentic_rag,
        'ingestion_pipeline': ingestion_pipeline,
        'hybrid_search': hybrid_search,
        'ontology': ontology,
        'query_expander': query_expander,
        'citation_generator': citation_generator,
        'guardrails': guardrails,
        'audit_logger': audit_logger,
        'graph_rag': graph_rag,
        'agentic_rag_lc': agentic_rag_lc,
        'web_indexer': web_indexer,
        'query_analyzer': query_analyzer,
        'report_formatter': report_formatter
    }


def render_star_rating(current_rating: int, key_prefix: str):
    """Render a single 5-star rating bar"""
    stars_html = ""
    for i in range(1, 6):
        filled_class = "filled" if i <= current_rating else ""
        stars_html += f'<span class="star-item {filled_class}">⭐</span>'

    rating_labels = {0: "Click a star to rate", 1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"}

    st.markdown(f"""
    <div class="star-rating-container">
        <div class="star-rating-title">Rate this Response</div>
        <div class="star-bar">{stars_html}</div>
        <div class="rating-text">{rating_labels.get(current_rating, "")}</div>
    </div>
    """, unsafe_allow_html=True)

    # Use columns for clickable buttons (hidden styling)
    cols = st.columns(5)
    new_rating = current_rating
    for i, col in enumerate(cols):
        with col:
            if st.button(f"{'⭐' * (i+1)}", key=f"{key_prefix}_{i+1}", help=f"{i+1} star(s)"):
                new_rating = i + 1

    return new_rating


def main():
    services = init_services()

    # Initialize session state
    if 'star_rating' not in st.session_state:
        st.session_state.star_rating = 0
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'current_audit_id' not in st.session_state:
        st.session_state.current_audit_id = None
    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Hero Section
    st.markdown("""
    <div class="hero-premium">
        <div class="hero-title">🧠 NEXUS AI Enterprise</div>
        <div class="hero-subtitle">4-Layer Intelligent Knowledge Retrieval Architecture</div>
        <div style="margin-top: 1rem;">
            <span class="layer-badge layer-1">📥 Layer 1: IDP</span>
            <span class="layer-badge layer-2">🔍 Layer 2: Hybrid Search</span>
            <span class="layer-badge layer-3">🎯 Layer 3: Query Expansion</span>
            <span class="layer-badge layer-4">🛡️ Layer 4: Guardrails</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    doc_count = services['vector_store'].get_document_count()
    audit_stats = services['audit_logger'].get_stats()
    ontology_stats = services['ontology'].get_stats()

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{doc_count}</div>
            <div class="metric-label">Indexed Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{ontology_stats.get('total_objects', 0)}</div>
            <div class="metric-label">Ontology Objects</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{audit_stats.get('total_entries', 0)}</div>
            <div class="metric-label">Audit Entries</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{audit_stats.get('total_sessions', 0)}</div>
            <div class="metric-label">Sessions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🧠 NEXUS AI Enterprise")
        st.caption("4-Layer Architecture")
        st.divider()

        # === Conversation Threads Section ===
        st.markdown("#### 💬 Conversations")

        # New thread button
        if st.button("➕ New Conversation", use_container_width=True, type="primary"):
            st.session_state.current_thread_id = None
            st.session_state.messages = []
            st.session_state.star_rating = 0
            st.rerun()

        # Get existing threads
        threads = services['db'].get_all_threads(limit=10)

        if threads:
            for thread in threads:
                thread_id = thread['id']
                title = thread['title'][:30] + "..." if len(thread['title']) > 30 else thread['title']
                msg_count = thread.get('message_count', 0)
                is_active = st.session_state.current_thread_id == thread_id

                # Thread button with delete option
                col1, col2 = st.columns([5, 1])
                with col1:
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(f"💬 {title}", key=f"thread_{thread_id}", use_container_width=True,
                                 help=f"{msg_count} messages"):
                        st.session_state.current_thread_id = thread_id
                        # Load messages for this thread
                        st.session_state.messages = services['db'].get_thread_messages(thread_id)
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{thread_id}", help="Delete thread"):
                        services['db'].delete_thread(thread_id)
                        if st.session_state.current_thread_id == thread_id:
                            st.session_state.current_thread_id = None
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.caption("No conversations yet")

        st.divider()

        st.markdown("#### 🤖 RAG Mode")
        # Initialize RAG mode in session state
        if 'selected_rag_mode' not in st.session_state:
            st.session_state.selected_rag_mode = "Auto"

        rag_mode = st.radio(
            "Select RAG Mode",
            ["Auto", "Standard", "Graph RAG", "Agentic RAG"],
            index=["Auto", "Standard", "Graph RAG", "Agentic RAG"].index(st.session_state.selected_rag_mode),
            horizontal=False,
            help="Auto: Intelligent selection based on query | Standard: Basic retrieval | Graph RAG: Knowledge graph enhanced | Agentic RAG: Multi-tool agents"
        )

        # Update session state when radio changes
        st.session_state.selected_rag_mode = rag_mode

        st.divider()

        st.markdown("#### 📊 Output Format")
        # Initialize report format in session state
        if 'report_format_enabled' not in st.session_state:
            st.session_state.report_format_enabled = True

        report_format = st.toggle(
            "Professional Report Format",
            value=st.session_state.report_format_enabled,
            help="Format responses as professional consulting reports with executive summary, analysis, and citations"
        )
        st.session_state.report_format_enabled = report_format

        st.divider()

        st.markdown("#### ⚙️ Active Layers")
        layer1_active = st.toggle("Layer 1: IDP", value=True)
        layer2_active = st.toggle("Layer 2: Hybrid Search", value=True)
        layer3_active = st.toggle("Layer 3: Query Expansion", value=True)
        layer4_active = st.toggle("Layer 4: Guardrails", value=True)

        st.divider()

        if st.button("🔄 Reload Knowledge Base", use_container_width=True):
            with st.spinner("Processing documents..."):
                policy_dir = "mock_data/policies"
                if os.path.exists(policy_dir):
                    for filename in os.listdir(policy_dir):
                        if filename.endswith('.txt'):
                            filepath = os.path.join(policy_dir, filename)
                            with open(filepath, 'r') as f:
                                content = f.read()
                            result = services['ingestion_pipeline'].process_document(content, filename)
                            st.write(f"📄 {filename}: {result['chunk_count']} chunks")
                    st.success("✅ Knowledge base reloaded!")
                    st.rerun()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Query", "📥 Ingest", "🌐 Web Ingest", "🏗️ Ontology", "📊 Audit"])

    with tab1:
        st.markdown("### Ask NEXUS AI")

        # Display conversation history if there are messages
        if st.session_state.messages:
            st.markdown("#### 💬 Conversation History")
            for msg in st.session_state.messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'user':
                    st.markdown(f"""
                    <div style="background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6;
                                padding: 12px 16px; border-radius: 8px; margin: 8px 0;">
                        <strong>👤 You:</strong><br>{content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(139, 92, 246, 0.1); border-left: 3px solid #8b5cf6;
                                padding: 12px 16px; border-radius: 8px; margin: 8px 0;">
                        <strong>🧠 NEXUS AI:</strong><br>{content[:500]}{'...' if len(content) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()

        col1, col2 = st.columns(2)
        with col1:
            case_type = st.selectbox("Case Type", ["Flood Insurance", "Auto Insurance", "Healthcare Benefits", "Regulatory"])
        with col2:
            state = st.selectbox("State", ["Florida", "California", "Texas", "New York"])

        case_context = {"case_type": case_type, "state": state}
        query = st.text_input("Your question", placeholder="Ask anything about policies...")

        if st.button("🔍 Search", type="primary") and query:
            # Reset rating for new search
            st.session_state.star_rating = 0

            # Start total timer
            total_start = time.time()

            # Create timer placeholder
            timer_placeholder = st.empty()

            # Layer 4: Input guardrails
            if layer4_active:
                input_check = services['guardrails'].check_input(query)
                if not input_check['passed']:
                    st.error(f"⚠️ Query blocked: {input_check['violations']}")
                    st.stop()
                if input_check.get('pii_redacted'):
                    st.warning(f"🔒 PII redacted: {input_check['pii_redacted']}")
                query = input_check['redacted_query']

            audit_id = services['audit_logger'].log_query(raw_prompt=query, case_context=case_context)
            st.session_state.current_audit_id = audit_id

            # Show RAG mode indicator
            # Automatic mode selection if needed
            actual_rag_mode = rag_mode
            analysis_info = None

            if rag_mode == "Auto":
                # Analyze query to determine best mode
                analysis = services['query_analyzer'].analyze_query(query, case_context)
                actual_rag_mode = analysis.recommended_mode
                analysis_info = analysis

                # Show analysis result
                st.markdown(f"""
                <div style="background: rgba(99, 102, 241, 0.15); padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6366f1;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.2rem;">🧠</span>
                        <div>
                            <div style="font-weight: 600; color: #6366f1; margin-bottom: 4px;">
                                Auto-Selected: {actual_rag_mode}
                            </div>
                            <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7);">
                                Complexity: {analysis.complexity.title()} | Confidence: {analysis.confidence:.0%}
                            </div>
                            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6); margin-top: 4px;">
                                {analysis.reasoning.split(' - ')[1] if ' - ' in analysis.reasoning else analysis.reasoning}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 8px 16px; border-radius: 8px; margin: 10px 0; display: inline-block;">
                <span style="color: white; font-weight: 500;">🤖 RAG Mode: {actual_rag_mode}</span>
            </div>
            """, unsafe_allow_html=True)

            # Handle different RAG modes
            if actual_rag_mode == "Agentic RAG":
                # Use Agentic RAG with LangChain
                timer_placeholder.markdown(f"""
                <div class="loading-timer">
                    <div class="timer-value">⏳ {time.time() - total_start:.1f}s</div>
                    <div class="timer-label">🤖 Agentic RAG processing with tools...</div>
                </div>
                """, unsafe_allow_html=True)

                agentic_result = services['agentic_rag_lc'].query(query, case_context)

                total_time = time.time() - total_start
                timer_placeholder.empty()

                st.markdown(f"""
                <div class="loading-timer">
                    <div class="timer-value">✅ {total_time:.2f}s</div>
                    <div class="timer-label">Agentic RAG Complete | Mode: {agentic_result.get('mode', 'unknown')}</div>
                </div>
                """, unsafe_allow_html=True)

                # Show tools used
                if agentic_result.get('tools_used'):
                    st.markdown(f"""
                    <div style="background: rgba(139, 92, 246, 0.1); padding: 10px; border-radius: 8px; margin: 10px 0;">
                        <strong>🔧 Tools Used:</strong> {', '.join(agentic_result['tools_used'])}
                    </div>
                    """, unsafe_allow_html=True)

                # Display answer
                answer = agentic_result.get('answer', 'No answer generated')

                # Format as professional report if enabled
                if st.session_state.report_format_enabled:
                    formatted_report = services['report_formatter'].format_response(
                        query=query,
                        answer=answer,
                        sources=[],  # Agentic RAG doesn't return sources directly
                        rag_mode="Agentic RAG",
                        case_context=case_context,
                        metadata={
                            'tools_used': agentic_result.get('tools_used', []),
                            'model_used': agentic_result.get('model_used', ''),
                            'complexity': analysis_info.complexity if analysis_info else 'medium',
                            'confidence': analysis_info.confidence if analysis_info else 0.8
                        }
                    )
                    st.markdown(formatted_report, unsafe_allow_html=True)

                    # Add download button for report
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.download_button(
                            label="📄 Download as Markdown",
                            data=formatted_report,
                            file_name=f"nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    with col2:
                        html_report = ReportExporter.to_html(formatted_report)
                        st.download_button(
                            label="🌐 Download as HTML",
                            data=html_report,
                            file_name=f"nexus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                else:
                    # Standard answer display
                    st.markdown(f"""
                    <div class="ai-response-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                            <span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; color: white;">
                                🤖 Agentic RAG Response
                            </span>
                        </div>
                        <div style="color: rgba(255,255,255,0.9); line-height: 1.8;">
                            {answer.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Save to thread
                if st.session_state.current_thread_id is None:
                    thread_title = query[:50] + "..." if len(query) > 50 else query
                    st.session_state.current_thread_id = services['db'].create_thread(
                        title=thread_title, case_type=case_type, state=state
                    )
                services['db'].add_message(st.session_state.current_thread_id, 'user', query)
                services['db'].add_message(st.session_state.current_thread_id, 'assistant', answer)
                st.session_state.messages = services['db'].get_thread_messages(st.session_state.current_thread_id)

            elif actual_rag_mode == "Graph RAG":
                # Use Graph RAG
                timer_placeholder.markdown(f"""
                <div class="loading-timer">
                    <div class="timer-value">⏳ {time.time() - total_start:.1f}s</div>
                    <div class="timer-label">📊 Graph RAG: Searching with knowledge graph...</div>
                </div>
                """, unsafe_allow_html=True)

                # Graph enhanced search
                graph_results = services['graph_rag'].graph_enhanced_search(query, top_k=TOP_K_RESULTS)

                # Generate graph-aware answer
                timer_placeholder.markdown(f"""
                <div class="loading-timer">
                    <div class="timer-value">⏳ {time.time() - total_start:.1f}s</div>
                    <div class="timer-label">📊 Generating graph-enhanced response...</div>
                </div>
                """, unsafe_allow_html=True)

                answer = services['graph_rag'].generate_graph_aware_answer(query, graph_results, case_context)

                total_time = time.time() - total_start
                timer_placeholder.empty()

                st.markdown(f"""
                <div class="loading-timer">
                    <div class="timer-value">✅ {total_time:.2f}s</div>
                    <div class="timer-label">Graph RAG Complete</div>
                </div>
                """, unsafe_allow_html=True)

                # Show graph context
                all_entities = []
                for r in graph_results[:3]:
                    all_entities.extend(r.get('mentioned_entities', []))
                if all_entities:
                    st.markdown(f"""
                    <div style="background: rgba(34, 197, 94, 0.1); padding: 10px; border-radius: 8px; margin: 10px 0;">
                        <strong>🔗 Entities Found:</strong> {', '.join(set(all_entities[:10]))}
                    </div>
                    """, unsafe_allow_html=True)

                # Display answer
                if st.session_state.report_format_enabled:
                    # Convert graph results to sources format
                    sources_for_report = []
                    for gr in graph_results[:5]:
                        sources_for_report.append({
                            'content': gr.get('content', ''),
                            'metadata': gr.get('metadata', {}),
                            'score': gr.get('score', 0)
                        })

                    formatted_report = services['report_formatter'].format_response(
                        query=query,
                        answer=answer,
                        sources=sources_for_report,
                        rag_mode="Graph RAG",
                        case_context=case_context,
                        metadata={
                            'entities_found': ', '.join(set(all_entities[:10])) if all_entities else '',
                            'complexity': analysis_info.complexity if analysis_info else 'medium',
                            'confidence': analysis_info.confidence if analysis_info else 0.8
                        }
                    )
                    st.markdown(formatted_report, unsafe_allow_html=True)

                    # Add download buttons
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.download_button(
                            label="📄 Download as Markdown",
                            data=formatted_report,
                            file_name=f"nexus_graph_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    with col2:
                        html_report = ReportExporter.to_html(formatted_report)
                        st.download_button(
                            label="🌐 Download as HTML",
                            data=html_report,
                            file_name=f"nexus_graph_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                else:
                    # Standard display
                    st.markdown(f"""
                    <div class="ai-response-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                            <span style="background: linear-gradient(135deg, #22c55e, #16a34a); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; color: white;">
                                📊 Graph RAG Response
                            </span>
                        </div>
                        <div style="color: rgba(255,255,255,0.9); line-height: 1.8;">
                            {answer.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Save to thread
                if st.session_state.current_thread_id is None:
                    thread_title = query[:50] + "..." if len(query) > 50 else query
                    st.session_state.current_thread_id = services['db'].create_thread(
                        title=thread_title, case_type=case_type, state=state
                    )
                services['db'].add_message(st.session_state.current_thread_id, 'user', query)
                services['db'].add_message(st.session_state.current_thread_id, 'assistant', answer)
                st.session_state.messages = services['db'].get_thread_messages(st.session_state.current_thread_id)

            else:
                # Standard RAG mode - existing logic continues below
                pass

            # Only run standard RAG logic if actual mode is Standard
            if actual_rag_mode not in ["Standard"]:
                st.stop()  # Stop here for Agentic/Graph modes

            # Layer 3: Query expansion with timer
            expanded_query = None
            sub_queries = []

            expansion_start = time.time()
            timer_placeholder.markdown(f"""
            <div class="loading-timer">
                <div class="timer-value">⏳ {time.time() - total_start:.1f}s</div>
                <div class="timer-label">Expanding query...</div>
            </div>
            """, unsafe_allow_html=True)

            if layer3_active:
                expanded = services['query_expander'].expand(query, case_context)
                expanded_query = expanded.transformed_query
                sub_queries = expanded.sub_queries
            expansion_time = time.time() - expansion_start

            # Layer 2: Search with timer
            retrieval_start = time.time()
            timer_placeholder.markdown(f"""
            <div class="loading-timer">
                <div class="timer-value">⏳ {time.time() - total_start:.1f}s</div>
                <div class="timer-label">Searching knowledge base...</div>
            </div>
            """, unsafe_allow_html=True)

            search_query = expanded_query or query
            results = services['retrieval_engine'].retrieve(
                query=search_query, case_context=case_context, top_k=TOP_K_RESULTS
            )

            if layer3_active and sub_queries:
                for sq in sub_queries[:2]:
                    sub_results = services['retrieval_engine'].retrieve(
                        query=sq, case_context=case_context, top_k=2
                    )
                    results.extend(sub_results)

            # Content-based deduplication (removes near-duplicates)
            results = deduplicate_results(results, similarity_threshold=0.80)

            # Add bbox highlighting and fix page numbers from content
            unique_results = []
            for r in results:
                content = r.get('content', '')
                highlighted, bboxes = highlight_content_with_bbox(content, query)
                r['highlighted_content'] = highlighted
                r['bboxes'] = bboxes

                # Fix page number: extract from content if metadata is wrong
                metadata = r.get('metadata', {})
                metadata_page = metadata.get('page', r.get('page', 1))
                actual_page = extract_page_from_content(content, metadata_page)
                r['page'] = actual_page
                if 'metadata' in r:
                    r['metadata']['page'] = actual_page

                unique_results.append(r)

            retrieval_time = time.time() - retrieval_start

            services['audit_logger'].log_retrieval(
                entry_id=audit_id,
                documents_accessed=[r.get('document_id', 'unknown') for r in unique_results],
                retrieval_scores=[r.get('relevance_score', 0) for r in unique_results],
                chunks_retrieved=len(unique_results)
            )

            if unique_results:
                # Generation with timer
                generation_start = time.time()
                timer_placeholder.markdown(f"""
                <div class="loading-timer">
                    <div class="timer-value">⏳ {time.time() - total_start:.1f}s</div>
                    <div class="timer-label">Generating response with AI...</div>
                </div>
                """, unsafe_allow_html=True)

                context_parts = []
                for i, r in enumerate(unique_results[:5]):
                    doc_id = r.get('document_id', f'DOC{i}')
                    page = r.get('page', 1)
                    para = r.get('paragraph', i + 1)
                    citation_tag = f"[{doc_id}:p{page}:¶{para}]"
                    context_parts.append(f"SOURCE {citation_tag}:\n{r.get('content', '')}")

                context_text = "\n\n---\n\n".join(context_parts)

                prompt = f"""You are NEXUS AI. Answer based ONLY on the provided sources.
CRITICAL: Cite sources using [DOC_ID:pPAGE:¶PARA] after EVERY claim.

Case Context: {case_type}, {state}

SOURCES:
{context_text}

QUESTION: {query}

Provide a detailed answer with inline citations:"""

                model = genai.GenerativeModel(GENERATION_MODEL)
                response = model.generate_content(prompt)
                answer = response.text
                generation_time = time.time() - generation_start

                # Total time
                total_time = time.time() - total_start

                # Clear timer and show final timing
                timer_placeholder.empty()

                # Show timing breakdown
                st.markdown(f"""
                <div class="loading-timer">
                    <div class="timer-value">✅ {total_time:.2f}s</div>
                    <div class="timer-label">
                        Query Expansion: {expansion_time:.2f}s |
                        Retrieval: {retrieval_time:.2f}s |
                        Generation: {generation_time:.2f}s
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Guardrails check
                if layer4_active:
                    output_check = services['guardrails'].check_output(
                        answer, sources=[r.get('content', '') for r in unique_results]
                    )
                    if output_check.get('pii_redacted'):
                        st.warning("🔒 PII redacted from response")
                        answer = output_check['redacted_answer']

                    st.markdown(f"""
                    <div class="guardrails-panel">
                        <strong>🛡️ Guardrails (Layer 4)</strong>
                        <span class="{'guardrail-pass' if output_check['passed'] else 'guardrail-fail'}">
                            {'✅ Passed' if output_check['passed'] else '⚠️ Warnings'}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                # Query expansion info
                if layer3_active and sub_queries:
                    st.markdown(f"""
                    <div class="query-expansion">
                        <strong>🎯 Query Expansion (Layer 3)</strong><br>
                        <small>Intent: {expanded.intent}</small><br>
                        {''.join([f'<span class="sub-query-chip">{sq}</span>' for sq in sub_queries[:4]])}
                    </div>
                    """, unsafe_allow_html=True)

                services['audit_logger'].log_generation(
                    entry_id=audit_id,
                    model_used=GENERATION_MODEL,
                    raw_model_output=response.text,
                    final_answer=answer,
                    citations_used=re.findall(r'\[([^\]]+)\]', answer)
                )

                # Save to conversation thread
                if st.session_state.current_thread_id is None:
                    # Create new thread with first query as title
                    thread_title = query[:50] + "..." if len(query) > 50 else query
                    st.session_state.current_thread_id = services['db'].create_thread(
                        title=thread_title,
                        case_type=case_type,
                        state=state
                    )

                # Save user message
                sources_json = json.dumps([{
                    'doc_id': r.get('document_id', ''),
                    'page': r.get('page', 1),
                    'score': r.get('relevance_score', 0)
                } for r in unique_results[:5]])

                services['db'].add_message(
                    thread_id=st.session_state.current_thread_id,
                    role='user',
                    content=query
                )
                # Save assistant message
                services['db'].add_message(
                    thread_id=st.session_state.current_thread_id,
                    role='assistant',
                    content=answer,
                    sources=sources_json
                )

                # Update session messages
                st.session_state.messages = services['db'].get_thread_messages(st.session_state.current_thread_id)

                # AI Response
                st.markdown(f"""
                <div class="ai-response-card">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                        <span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; color: white;">
                            🧠 NEXUS AI Response
                        </span>
                    </div>
                    <div style="color: rgba(255,255,255,0.9); line-height: 1.8;">
                        {answer.replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Sources with bounding boxes
                st.markdown("### 📚 Sources (Click to View with Highlighted Matches)")

                for i, r in enumerate(unique_results[:5]):
                    doc_id = r.get('document_id', f'DOC{i}')
                    doc_title = r.get('document_title', 'Document')
                    score = r.get('relevance_score', 0)
                    bboxes = r.get('bboxes', [])
                    match_count = len(bboxes)

                    # Get page and section info from metadata
                    metadata = r.get('metadata', {})
                    page_num = metadata.get('page', r.get('page', 1))
                    paragraph_num = metadata.get('paragraph', r.get('paragraph', i + 1))
                    section_header = metadata.get('section_header', '')

                    # Filter out false section headers (years like "2012.", page markers)
                    if section_header and re.match(r'^\d{4}\.$', section_header.strip()):
                        section_header = ""  # Remove year-like headers
                    if section_header and "Page" in section_header:
                        section_header = ""  # Remove page markers

                    # Build page info string
                    page_info = f"Page {page_num}"
                    if section_header:
                        page_info += f" • {section_header}"

                    with st.expander(f"📄 {doc_title} — **{page_info}** — {score:.0%} match"):
                        # Document viewer with highlighted content
                        st.markdown(f"""
                        <div class="doc-viewer-container">
                            <div class="doc-viewer-header">
                                <span><strong>{doc_title}</strong></span>
                                <span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem;">
                                    📄 Page {page_num} | ¶ {paragraph_num}
                                </span>
                                <span class="match-count-badge">🎯 {match_count} matches</span>
                            </div>
                            <div class="doc-content-wrapper">
                                <div class="doc-text">{r.get('highlighted_content', r.get('content', ''))}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Bounding box details
                        if bboxes:
                            st.markdown("**📍 Bounding Boxes (for UI highlighting):**")
                            for bbox in bboxes[:5]:
                                st.markdown(f"""
                                <div class="bbox-info">
                                    <strong>"{bbox['matched_text']}"</strong> →
                                    Page {page_num} | x:{bbox['x']}px y:{bbox['y']}px
                                    w:{bbox['width']}px h:{bbox['height']}px
                                </div>
                                """, unsafe_allow_html=True)

                        # Citation with full reference
                        st.markdown(f"""
                        <div style="background: rgba(99, 102, 241, 0.1); padding: 8px 12px; border-radius: 8px; margin-top: 10px;">
                            <strong>📎 Citation:</strong> <code>[{doc_id}:p{page_num}:¶{paragraph_num}]</code>
                        </div>
                        """, unsafe_allow_html=True)

                # Star Rating - Single bar
                st.markdown("---")

                # Display current rating
                current_rating = st.session_state.star_rating
                stars_display = "".join(["⭐" if i < current_rating else "☆" for i in range(5)])
                rating_labels = {0: "Click to rate", 1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"}

                st.markdown(f"""
                <div class="star-rating-container">
                    <div class="star-rating-title">⭐ Rate this Response</div>
                    <div style="font-size: 2.5rem; letter-spacing: 8px;">{stars_display}</div>
                    <div class="rating-text">{rating_labels.get(current_rating, "")}</div>
                </div>
                """, unsafe_allow_html=True)

                # Rating buttons in a row
                rating_cols = st.columns(5)
                for i, col in enumerate(rating_cols):
                    with col:
                        if st.button(f"{i+1}⭐", key=f"rate_{i+1}", use_container_width=True):
                            st.session_state.star_rating = i + 1
                            if st.session_state.current_audit_id:
                                services['audit_logger'].log_feedback(st.session_state.current_audit_id, i + 1)
                            st.success(f"Thank you! Rated {i + 1} star{'s' if i > 0 else ''}")
                            st.rerun()

            else:
                timer_placeholder.empty()
                st.warning("No relevant documents found.")

    with tab2:
        st.markdown("### 📥 Document Ingestion (Layer 1)")

        # Show current index stats
        doc_count = services['vector_store'].get_document_count()
        st.info(f"📊 Currently indexed: **{doc_count}** chunks in vector database")

        # Re-index button for fixing metadata issues
        col_reindex, col_clear = st.columns(2)
        with col_reindex:
            if st.button("🔄 Re-index All Documents", use_container_width=True, help="Re-process all documents with updated chunking logic"):
                with st.spinner("Re-indexing all documents..."):
                    policy_dir = "mock_data/policies"
                    if os.path.exists(policy_dir):
                        # Clear existing index
                        services['vector_store'].reset_collection()
                        services['hybrid_search'].bm25_index = BM25Index()

                        total_chunks = 0
                        for filename in os.listdir(policy_dir):
                            if filename.endswith('.txt') or filename.endswith('.md'):
                                filepath = os.path.join(policy_dir, filename)
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    content = f.read()

                                # Process with updated chunking logic
                                result = services['ingestion_pipeline'].process_document(content, filename)
                                chunks = result['chunks']

                                if chunks:
                                    ids = [f"{result['doc_id']}_chunk_{i}" for i in range(len(chunks))]
                                    documents = [chunk.text for chunk in chunks]
                                    metadatas = [chunk.to_metadata() for chunk in chunks]

                                    for meta in metadatas:
                                        meta['title'] = filename
                                        meta['document_id'] = result['doc_id']

                                    embeddings = services['embedding_service'].get_embeddings_batch(documents)
                                    services['vector_store'].add_embeddings(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                                    services['hybrid_search'].bm25_index.add_documents(ids, documents)
                                    total_chunks += len(chunks)

                        st.success(f"✅ Re-indexed {total_chunks} chunks with updated metadata!")
                        st.rerun()
                    else:
                        st.warning("No documents found in mock_data/policies")

        with col_clear:
            if st.button("🗑️ Clear Index", use_container_width=True, type="secondary"):
                services['vector_store'].reset_collection()
                services['hybrid_search'].bm25_index = BM25Index()
                st.success("Index cleared!")
                st.rerun()

        st.divider()
        uploaded = st.file_uploader("Upload document", type=['txt', 'md', 'pdf'])
        if uploaded and st.button("🚀 Process & Index", type="primary"):
            with st.spinner("Processing document..."):
                # Handle different file types
                if uploaded.name.lower().endswith('.pdf'):
                    # Extract text from PDF
                    try:
                        import PyPDF2
                        pdf_reader = PyPDF2.PdfReader(uploaded)
                        content = ""
                        for page_num, page in enumerate(pdf_reader.pages):
                            page_text = page.extract_text()
                            if page_text:
                                # Try to extract printed page number from the page text
                                # Look for standalone number at start of lines (common footer pattern)
                                lines = page_text.strip().split('\n')
                                printed_page = None
                                for line in lines[:3] + lines[-3:]:  # Check first/last 3 lines
                                    line = line.strip()
                                    if re.match(r'^\d{1,3}$', line):  # Standalone 1-3 digit number
                                        printed_page = int(line)
                                        break
                                # Use printed page number if found, otherwise use PDF page
                                display_page = printed_page if printed_page else page_num + 1
                                content += f"\n--- Page {display_page} ---\n{page_text}"
                        if not content.strip():
                            st.error("Could not extract text from PDF. It may be scanned/image-based.")
                            content = None
                    except ImportError:
                        st.error("PyPDF2 not installed. Run: pip install PyPDF2")
                        content = None
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")
                        content = None
                else:
                    # Text/Markdown files
                    content = uploaded.read().decode('utf-8')

                if content:
                    # Step 1: Process document (classify, chunk)
                    result = services['ingestion_pipeline'].process_document(content, uploaded.name)
                    st.success(f"✅ Processed: {result['chunk_count']} chunks from {uploaded.name}")

                    with st.expander("Document Classification"):
                        st.json(result['classification'])

                    # Step 2: Generate embeddings and index to vector store
                    progress_bar = st.progress(0, text="Generating embeddings...")
                    chunks = result['chunks']

                    if chunks:
                        # Prepare data for vector store
                        ids = [f"{result['doc_id']}_chunk_{i}" for i in range(len(chunks))]
                        documents = [chunk.text for chunk in chunks]
                        metadatas = [chunk.to_metadata() for chunk in chunks]

                        # Add document title to metadata
                        for meta in metadatas:
                            meta['title'] = uploaded.name
                            meta['document_id'] = result['doc_id']

                        progress_bar.progress(30, text="Generating embeddings...")

                        # Generate embeddings using local model
                        embeddings = services['embedding_service'].get_embeddings_batch(documents)

                        progress_bar.progress(60, text="Storing in vector database...")

                        # Store in vector database
                        services['vector_store'].add_embeddings(
                            ids=ids,
                            embeddings=embeddings,
                            documents=documents,
                            metadatas=metadatas
                        )

                        progress_bar.progress(80, text="Updating search index...")

                        # Also update BM25 index for hybrid search
                        services['hybrid_search'].bm25_index.add_documents(ids, documents)

                        # Store in SQLite for reference
                        sqlite_doc_id = services['db'].add_document(
                            title=uploaded.name,
                            doc_type=result['classification']['doc_type'],
                            content=content[:500],  # Store summary
                            metadata=json.dumps({"vector_doc_id": result['doc_id'], "chunk_count": len(chunks)})
                        )

                        progress_bar.progress(100, text="Done!")

                        st.success(f"🎉 Successfully indexed {len(chunks)} chunks!")
                        st.metric("Total Documents in DB", services['vector_store'].get_document_count())

    with tab3:
        st.markdown("### 🌐 Web Content Ingestion")
        st.caption("Add web pages to your knowledge base")

        # URL input
        url_input = st.text_area(
            "Enter URLs (one per line)",
            placeholder="https://example.com/policy-document\nhttps://another-site.com/regulations",
            height=100
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌐 Ingest URLs", type="primary", use_container_width=True):
                if url_input.strip():
                    urls = [u.strip() for u in url_input.strip().split('\n') if u.strip()]
                    with st.spinner(f"Ingesting {len(urls)} URLs..."):
                        results = services['web_indexer'].index_multiple_urls(urls)

                        success_count = sum(1 for r in results if r.get('success'))
                        st.success(f"✅ Successfully indexed {success_count}/{len(urls)} URLs")

                        for result in results:
                            if result.get('success'):
                                st.markdown(f"""
                                <div style="background: rgba(34, 197, 94, 0.1); padding: 10px; border-radius: 8px; margin: 5px 0;">
                                    ✅ <strong>{result['title']}</strong><br>
                                    <small>{result['url']} - {result['chunks_indexed']} chunks</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                error_msg = result.get('error', 'Unknown error')
                                st.markdown(f"""
                                <div style="background: rgba(239, 68, 68, 0.1); padding: 10px; border-radius: 8px; margin: 5px 0;">
                                    ❌ <strong>Failed:</strong> {result.get('url', 'Unknown')}<br>
                                    <small>{error_msg}</small>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter at least one URL")

        with col2:
            if st.button("🔄 Clear Web Content", use_container_width=True):
                st.info("Web content clearing would remove web-sourced chunks")

        st.divider()
        st.markdown("#### 📊 Knowledge Graph Stats")
        graph_stats = services['graph_rag'].knowledge_graph.get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Entities", graph_stats['total_entities'])
        c2.metric("Relationships", graph_stats['total_relationships'])
        c3.metric("Components", graph_stats['connected_components'])

        if graph_stats['entity_types']:
            st.markdown("**Entity Types:**")
            for etype, count in graph_stats['entity_types'].items():
                st.caption(f"  • {etype}: {count}")

    with tab4:
        st.markdown("### 🏗️ Ontology (Layer 2)")
        stats = services['ontology'].get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Policies", stats['policies'])
        c2.metric("Claims", stats['claims'])
        c3.metric("Providers", stats['providers'])

    with tab5:
        st.markdown("### 📊 Audit Log (Layer 4)")
        history = services['audit_logger'].get_session_history()
        for entry in history[-10:]:
            rating = entry.get('feedback_rating') or 0
            st.markdown(f"""
            <div class="audit-entry">
                {entry.get('timestamp', '')[:19]} |
                {entry.get('action_type', '')} |
                Docs: {entry.get('chunks_retrieved', 0)} |
                Rating: {'⭐' * rating if rating else '—'}
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
