"""Configuration settings for the Knowledge Retrieval System"""

import os

# Google Gemini API Configuration
GEMINI_API_KEY = "AIzaSyAwR5-DG6Oa__WxCuiVdvcITOJsuoVeLzA"

# Database Configuration
SQLITE_DB_PATH = "data/knowledge_base.db"
CHROMA_PERSIST_DIR = "data/chroma_db_v2"  # New folder for free embeddings (384-dim)

# Document Processing Configuration
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks

# Retrieval Configuration
TOP_K_RESULTS = 5  # number of results to return
SIMILARITY_THRESHOLD = 0.7  # minimum similarity score

# Model Configuration
EMBEDDING_MODEL = "models/embedding-001"

# Fallback model list - tries in order when quota is exceeded
GENERATION_MODELS = [
    "gemini-3-flash-preview",         # Gemini 3 Flash - latest model with frontier intelligence
    "gemini-3-pro-preview",           # Gemini 3 Pro - reasoning-first model for complex workflows
    "gemini-2.0-flash-exp",           # Gemini 2.0 experimental
    "gemini-1.5-flash-8b",            # Fast 8B parameter model
    "gemini-1.5-flash",               # Stable 1.5 version
    "gemini-1.5-pro",                 # Pro version as last resort
]
GENERATION_MODEL = GENERATION_MODELS[0]  # Default to first model
