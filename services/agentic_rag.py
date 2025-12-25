"""
Agentic RAG Service
Supports multiple sources: Web URLs, PDFs, Text files, APIs
With intelligent routing and source verification
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
from functools import wraps


def timed_operation(func):
    """Decorator to track operation time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        # Add timing info to result if it's a dict
        if isinstance(result, dict):
            result['time_taken'] = round(elapsed_time, 2)
            result['time_taken_str'] = f"{elapsed_time:.2f}s"

        return result
    return wrapper


class AgenticRAG:
    """
    Agentic RAG system that can:
    1. Ingest content from multiple sources (URLs, PDFs, text)
    2. Intelligently route queries to relevant sources
    3. Verify and cite sources with full provenance
    4. Handle dynamic web content
    """

    def __init__(self, embedding_service, vector_store, db):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.db = db
        self.source_registry = {}  # Track all ingested sources
        self.web_cache = {}  # Cache web content

    @timed_operation
    def ingest_url(self, url: str, metadata: Dict = None) -> Dict:
        """
        Ingest content from a URL (webpage, PDF link, API endpoint)
        """
        try:
            # Parse URL
            parsed = urlparse(url)
            domain = parsed.netloc

            # Fetch content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')

            # Handle different content types
            if 'application/pdf' in content_type:
                return self._ingest_pdf_url(url, response.content, metadata)
            elif 'application/json' in content_type:
                return self._ingest_api_response(url, response.json(), metadata)
            else:
                return self._ingest_webpage(url, response.text, metadata)

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    def _ingest_webpage(self, url: str, html: str, metadata: Dict = None) -> Dict:
        """Extract and ingest content from HTML webpage"""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script, style, nav, footer elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Extract title
        title = soup.title.string if soup.title else urlparse(url).netloc

        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')

        if not main_content:
            return {'success': False, 'error': 'No content found', 'url': url}

        # Extract text with structure
        sections = []
        current_section = {'title': 'Introduction', 'content': '', 'elements': []}

        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'td']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                if current_section['content'].strip():
                    sections.append(current_section)
                current_section = {
                    'title': element.get_text(strip=True),
                    'content': '',
                    'elements': []
                }
            else:
                text = element.get_text(strip=True)
                if text:
                    current_section['content'] += text + '\n'
                    current_section['elements'].append({
                        'tag': element.name,
                        'text': text
                    })

        if current_section['content'].strip():
            sections.append(current_section)

        # Create chunks
        chunks = []
        for i, section in enumerate(sections):
            if len(section['content']) > 50:  # Skip very short sections
                chunk_metadata = {
                    'source_type': 'webpage',
                    'url': url,
                    'domain': urlparse(url).netloc,
                    'title': title,
                    'section': section['title'],
                    'section_index': i,
                    'ingested_at': datetime.now().isoformat(),
                    **(metadata or {})
                }
                chunks.append({
                    'content': section['content'],
                    'metadata': chunk_metadata
                })

        # Ingest chunks
        return self._ingest_chunks(chunks, f"web_{hashlib.md5(url.encode()).hexdigest()[:8]}")

    def _ingest_pdf_url(self, url: str, content: bytes, metadata: Dict = None) -> Dict:
        """Ingest PDF from URL"""
        try:
            import io
            # Try to use PyPDF2 or pdfplumber if available
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except ImportError:
                return {'success': False, 'error': 'PDF parsing not available', 'url': url}

            # Create chunks from PDF text
            chunks = self._chunk_text(text, {
                'source_type': 'pdf_url',
                'url': url,
                'title': url.split('/')[-1],
                'ingested_at': datetime.now().isoformat(),
                **(metadata or {})
            })

            return self._ingest_chunks(chunks, f"pdf_{hashlib.md5(url.encode()).hexdigest()[:8]}")

        except Exception as e:
            return {'success': False, 'error': str(e), 'url': url}

    def _ingest_api_response(self, url: str, data: Dict, metadata: Dict = None) -> Dict:
        """Ingest JSON API response"""
        try:
            # Flatten JSON to text
            text = self._flatten_json(data)

            chunks = self._chunk_text(text, {
                'source_type': 'api',
                'url': url,
                'title': f"API: {urlparse(url).netloc}",
                'ingested_at': datetime.now().isoformat(),
                **(metadata or {})
            })

            return self._ingest_chunks(chunks, f"api_{hashlib.md5(url.encode()).hexdigest()[:8]}")

        except Exception as e:
            return {'success': False, 'error': str(e), 'url': url}

    def _flatten_json(self, data: Any, prefix: str = '') -> str:
        """Convert nested JSON to readable text"""
        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                lines.append(self._flatten_json(value, new_prefix))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                lines.append(self._flatten_json(item, f"{prefix}[{i}]"))
        else:
            lines.append(f"{prefix}: {data}")

        return '\n'.join(filter(None, lines))

    def _chunk_text(self, text: str, metadata: Dict, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Split text into chunks"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': {
                        **metadata,
                        'chunk_index': chunk_index,
                        'char_start': len(text) - len(current_chunk),
                        'char_end': len(text)
                    }
                })
                # Keep overlap
                words = current_chunk.split()
                current_chunk = ' '.join(words[-overlap//10:]) if len(words) > overlap//10 else ""
                chunk_index += 1

            current_chunk += " " + sentence

        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    **metadata,
                    'chunk_index': chunk_index
                }
            })

        return chunks

    def _ingest_chunks(self, chunks: List[Dict], source_id: str) -> Dict:
        """Ingest chunks into vector store and database"""
        if not chunks:
            return {'success': False, 'error': 'No chunks to ingest'}

        try:
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for i, chunk in enumerate(chunks):
                # Add to database
                doc_id = self.db.add_document(
                    title=chunk['metadata'].get('title', 'Unknown'),
                    doc_type=chunk['metadata'].get('source_type', 'unknown'),
                    content=chunk['content'],
                    metadata=json.dumps(chunk['metadata'])
                )

                chunk_db_id = self.db.add_chunk(
                    document_id=doc_id,
                    chunk_index=chunk['metadata'].get('chunk_index', i),
                    content=chunk['content'],
                    page_number=chunk['metadata'].get('page', 1),
                    paragraph_number=chunk['metadata'].get('section_index', 1),
                    char_start=chunk['metadata'].get('char_start', 0),
                    char_end=chunk['metadata'].get('char_end', len(chunk['content'])),
                    metadata=json.dumps(chunk['metadata'])
                )

                chunk_ids.append(f"{source_id}_chunk_{chunk_db_id}")
                chunk_texts.append(chunk['content'])
                chunk_metadatas.append(chunk['metadata'])

            # Generate embeddings
            embeddings = self.embedding_service.get_embeddings_batch(chunk_texts)

            # Add to vector store
            self.vector_store.add_embeddings(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )

            # Register source
            source_info = {
                'source_id': source_id,
                'type': chunks[0]['metadata'].get('source_type'),
                'url': chunks[0]['metadata'].get('url'),
                'title': chunks[0]['metadata'].get('title'),
                'chunks_count': len(chunks),
                'ingested_at': datetime.now().isoformat()
            }
            self.source_registry[source_id] = source_info

            return {
                'success': True,
                'source_id': source_id,
                'chunks_ingested': len(chunks),
                'source_info': source_info
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    @timed_operation
    def ingest_text(self, text: str, title: str, source_type: str = 'text', metadata: Dict = None) -> Dict:
        """Ingest raw text content"""
        chunks = self._chunk_text(text, {
            'source_type': source_type,
            'title': title,
            'ingested_at': datetime.now().isoformat(),
            **(metadata or {})
        })

        return self._ingest_chunks(chunks, f"text_{hashlib.md5(title.encode()).hexdigest()[:8]}")

    @timed_operation
    def search_with_sources(self, query: str, top_k: int = 5, source_filter: List[str] = None) -> Dict:
        """
        Search across all sources with intelligent routing
        Returns results with full source information
        """
        # Get query embedding
        query_embedding = self.embedding_service.get_embedding(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k * 2  # Get more for filtering
        )

        # Process and enrich results
        enriched_results = []
        for result in results:
            metadata = result.get('metadata', {})

            # Apply source filter if specified
            if source_filter:
                source_type = metadata.get('source_type', '')
                if source_type not in source_filter:
                    continue

            enriched_result = {
                'content': result.get('content', result.get('document', '')),
                'relevance_score': result.get('score', 0),
                'source_type': metadata.get('source_type', 'unknown'),
                'source_url': metadata.get('url', ''),
                'source_title': metadata.get('title', 'Unknown'),
                'section': metadata.get('section', ''),
                'domain': metadata.get('domain', ''),
                'ingested_at': metadata.get('ingested_at', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'provenance': {
                    'char_start': metadata.get('char_start'),
                    'char_end': metadata.get('char_end'),
                    'page': metadata.get('page'),
                    'paragraph': metadata.get('section_index')
                }
            }
            enriched_results.append(enriched_result)

            if len(enriched_results) >= top_k:
                break

        return {
            'results': enriched_results,
            'count': len(enriched_results),
            'query': query
        }

    def get_source_stats(self) -> Dict:
        """Get statistics about all ingested sources"""
        stats = {
            'total_sources': len(self.source_registry),
            'by_type': {},
            'sources': list(self.source_registry.values())
        }

        for source in self.source_registry.values():
            source_type = source.get('type', 'unknown')
            if source_type not in stats['by_type']:
                stats['by_type'][source_type] = 0
            stats['by_type'][source_type] += 1

        return stats

    @timed_operation
    def crawl_website(self, base_url: str, max_pages: int = 10, metadata: Dict = None) -> Dict:
        """
        Crawl a website and ingest multiple pages
        """
        visited = set()
        to_visit = [base_url]
        results = []
        domain = urlparse(base_url).netloc

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue

            visited.add(url)

            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=15)

                if response.status_code == 200:
                    # Ingest the page
                    result = self._ingest_webpage(url, response.text, metadata)
                    results.append(result)

                    # Find more links on same domain
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(url, href)
                        parsed = urlparse(full_url)

                        # Only follow links on same domain
                        if parsed.netloc == domain and full_url not in visited:
                            to_visit.append(full_url)

                time.sleep(0.5)  # Be polite

            except Exception as e:
                results.append({'success': False, 'error': str(e), 'url': url})

        return {
            'pages_crawled': len(visited),
            'successful': sum(1 for r in results if r.get('success')),
            'results': results
        }


# Source type icons for UI
SOURCE_ICONS = {
    'webpage': '🌐',
    'pdf': '📄',
    'pdf_url': '📄',
    'api': '🔌',
    'text': '📝',
    'policy': '📋',
    'unknown': '📎'
}


def get_source_icon(source_type: str) -> str:
    """Get icon for source type"""
    return SOURCE_ICONS.get(source_type, '📎')
