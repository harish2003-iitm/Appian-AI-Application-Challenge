# Intelligent Knowledge Retrieval System

A "Just-in-Time" knowledge retrieval system for complex case management - built for the IITM Hackathon.

## Problem Statement

Organizations employing thousands of support agents handling high-stakes casework need instant access to relevant policy documents, regulations, and SOPs. This system automatically surfaces relevant information based on the active case context, with verifiable citations.

## Features

- **Context-Aware Suggestions**: Automatically retrieves relevant documents based on case type, state, and details
- **Semantic Search**: Uses Google Gemini embeddings for intelligent document matching
- **Verifiable Citations**: Every result includes document ID, section, and direct quotes
- **AI-Powered Answers**: Generates natural language answers with source citations
- **Multiple Case Types**: Supports Flood Insurance, Auto Insurance, Healthcare Benefits, and Regulatory Compliance

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM & Embeddings | Google Gemini API |
| Vector Database | ChromaDB |
| Metadata Storage | SQLite |
| Language | Python 3.10+ |

## Quick Start

### 1. Install Dependencies

```bash
cd intelligent-knowledge-retrieval
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Initialize Documents

1. Click "Reload Documents" in the sidebar to ingest the mock policy documents
2. Wait for the embeddings to be generated
3. Start exploring!

## Project Structure

```
intelligent-knowledge-retrieval/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── database/
│   ├── sqlite_db.py          # SQLite operations
│   └── vector_store.py       # ChromaDB operations
├── services/
│   ├── document_processor.py # Document ingestion & chunking
│   ├── embedding_service.py  # Gemini embedding generation
│   ├── retrieval_engine.py   # Context-aware retrieval
│   └── citation_manager.py   # Citation formatting
├── mock_data/
│   ├── policies/             # Sample policy documents
│   └── sample_cases.json     # Pre-configured test cases
└── utils/
    └── helpers.py            # Utility functions
```

## How It Works

1. **Document Ingestion**: Policy documents are chunked with metadata (section, paragraph, character positions)
2. **Embedding Generation**: Each chunk is converted to a vector using Gemini's embedding model
3. **Context Parsing**: When a case is loaded, the system extracts relevant fields (case type, state, etc.)
4. **Semantic Retrieval**: Query embedding is matched against document embeddings using cosine similarity
5. **Citation Generation**: Results include precise provenance (document, section, paragraph)
6. **AI Answer**: Gemini generates a natural language answer citing the retrieved documents

## Sample Cases

The system includes 5 pre-configured test cases:
1. Flood Damage Claim - Florida Residential
2. Multi-Vehicle Auto Accident
3. Healthcare Pre-Authorization Request
4. Compliance Audit - HIPAA Breach
5. Total Loss Vehicle Claim

## API Key

The application uses Google Gemini API. The key is configured in `config.py`.

## License

MIT License - Built for IITM Hackathon 2024
