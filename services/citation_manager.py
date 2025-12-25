"""Citation and provenance management service"""

from typing import List, Dict, Optional
import json

class CitationManager:
    def __init__(self, sqlite_db):
        self.db = sqlite_db

    def create_citation(self, chunk_id: int, search_id: int = None,
                        relevance_score: float = 0.0) -> int:
        """Create a citation record"""
        return self.db.add_citation(search_id, chunk_id, relevance_score)

    def format_citation_for_display(self, result: Dict) -> Dict:
        """Format a search result into a displayable citation"""
        metadata = result.get('metadata', {})

        citation = {
            'display_text': self._build_display_text(result),
            'source_document': metadata.get('title', 'Unknown Document'),
            'document_id': metadata.get('document_id', 'N/A'),
            'section': metadata.get('section', 'N/A'),
            'category': metadata.get('category', 'Document'),
            'relevance_score': result.get('relevance_score', 0),
            'content_preview': self._get_content_preview(result.get('content', '')),
            'full_content': result.get('content', ''),
            'metadata': metadata
        }

        return citation

    def _build_display_text(self, result: Dict) -> str:
        """Build the display text for a citation"""
        metadata = result.get('metadata', {})

        parts = []

        # Document title
        title = metadata.get('title', 'Document')
        parts.append(f"📄 {title}")

        # Document ID if available
        doc_id = metadata.get('document_id')
        if doc_id:
            parts.append(f"[{doc_id}]")

        # Section
        section = metadata.get('section')
        if section:
            parts.append(f"§ {section}")

        return " | ".join(parts)

    def _get_content_preview(self, content: str, max_length: int = 200) -> str:
        """Get a preview of the content"""
        if len(content) <= max_length:
            return content

        # Try to break at a sentence or word boundary
        preview = content[:max_length]
        last_period = preview.rfind('.')
        last_space = preview.rfind(' ')

        if last_period > max_length * 0.7:
            return preview[:last_period + 1]
        elif last_space > max_length * 0.7:
            return preview[:last_space] + "..."
        else:
            return preview + "..."

    def format_citations_list(self, results: List[Dict]) -> List[Dict]:
        """Format a list of search results into citations"""
        return [self.format_citation_for_display(r) for r in results]

    def generate_answer_with_citations(self, answer: str, citations: List[Dict]) -> str:
        """Add inline citation references to an answer"""
        # This would typically be done by the LLM, but here's a simple version
        formatted_answer = answer

        # Add citation references
        if citations:
            formatted_answer += "\n\n**Sources:**\n"
            for i, citation in enumerate(citations, 1):
                formatted_answer += f"{i}. {citation['display_text']}\n"

        return formatted_answer

    def get_citation_by_id(self, citation_id: int) -> Optional[Dict]:
        """Get full citation details by ID"""
        chunk = self.db.get_chunk(citation_id)
        if chunk:
            return {
                'chunk_id': chunk['id'],
                'content': chunk['content'],
                'document_title': chunk.get('doc_title', 'Unknown'),
                'document_type': chunk.get('doc_type', 'Unknown'),
                'page_number': chunk.get('page_number'),
                'paragraph_number': chunk.get('paragraph_number'),
                'char_start': chunk.get('char_start'),
                'char_end': chunk.get('char_end')
            }
        return None

    def export_citations(self, citations: List[Dict], format: str = 'json') -> str:
        """Export citations in various formats"""
        if format == 'json':
            return json.dumps(citations, indent=2)

        elif format == 'text':
            lines = []
            for i, c in enumerate(citations, 1):
                lines.append(f"{i}. {c['source_document']}")
                if c.get('section'):
                    lines.append(f"   Section: {c['section']}")
                if c.get('document_id'):
                    lines.append(f"   ID: {c['document_id']}")
                lines.append(f"   Relevance: {c['relevance_score']:.1%}")
                lines.append("")
            return "\n".join(lines)

        elif format == 'markdown':
            lines = ["## Citations\n"]
            for i, c in enumerate(citations, 1):
                lines.append(f"### {i}. {c['source_document']}\n")
                lines.append(f"- **Document ID:** {c.get('document_id', 'N/A')}")
                lines.append(f"- **Section:** {c.get('section', 'N/A')}")
                lines.append(f"- **Relevance:** {c['relevance_score']:.1%}")
                lines.append(f"\n> {c['content_preview']}\n")
            return "\n".join(lines)

        return str(citations)

    def verify_citation(self, citation: Dict, original_document: str) -> bool:
        """Verify that a citation exists in the original document"""
        content = citation.get('full_content', '')
        if not content or not original_document:
            return False

        # Check if the cited content appears in the original
        # Using a fuzzy match to account for minor differences
        return content[:100] in original_document
