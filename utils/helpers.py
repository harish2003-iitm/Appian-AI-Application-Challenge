"""Utility helper functions"""

from typing import Dict, List, Optional
import json
import re

def format_citation(doc_title: str, page_num: int = None, paragraph_num: int = None,
                    section: str = None) -> str:
    """Format a citation string for display"""
    citation_parts = [f"[{doc_title}]"]

    if section:
        citation_parts.append(f"Section: {section}")
    if page_num:
        citation_parts.append(f"Page {page_num}")
    if paragraph_num:
        citation_parts.append(f"Para {paragraph_num}")

    return " - ".join(citation_parts)

def extract_case_context(case_data: Dict) -> str:
    """Extract relevant context from case data for query generation"""
    context_parts = []

    if 'case_type' in case_data:
        context_parts.append(f"Case Type: {case_data['case_type']}")

    if 'state' in case_data:
        context_parts.append(f"State: {case_data['state']}")

    context = case_data.get('context', {})

    if 'claim_type' in context:
        context_parts.append(f"Claim Type: {context['claim_type']}")

    if 'property_type' in context:
        context_parts.append(f"Property Type: {context['property_type']}")

    if 'damage_amount' in context:
        context_parts.append(f"Damage Amount: ${context['damage_amount']:,}")

    if 'service_type' in context:
        context_parts.append(f"Service: {context['service_type']}")

    if 'urgency' in context:
        context_parts.append(f"Urgency: {context['urgency']}")

    return " | ".join(context_parts)

def highlight_text(full_text: str, highlight_phrase: str,
                   before_chars: int = 100, after_chars: int = 100) -> str:
    """Extract and highlight a phrase within its surrounding context"""
    if not highlight_phrase or highlight_phrase not in full_text:
        return full_text[:300] + "..." if len(full_text) > 300 else full_text

    # Find the phrase
    start_idx = full_text.lower().find(highlight_phrase.lower())
    if start_idx == -1:
        return full_text[:300] + "..."

    # Calculate context boundaries
    context_start = max(0, start_idx - before_chars)
    context_end = min(len(full_text), start_idx + len(highlight_phrase) + after_chars)

    # Extract context
    prefix = "..." if context_start > 0 else ""
    suffix = "..." if context_end < len(full_text) else ""

    context = full_text[context_start:context_end]

    return f"{prefix}{context}{suffix}"

def build_context_query(case_context: Dict) -> str:
    """Build a semantic search query from case context"""
    query_parts = []

    context = case_context.get('context', case_context)

    # Add claim/case type
    if 'claim_type' in context:
        query_parts.append(context['claim_type'])
    if 'case_type' in case_context:
        query_parts.append(case_context['case_type'])

    # Add location context
    if 'state' in case_context:
        query_parts.append(case_context['state'])
    if 'location' in context:
        query_parts.append(context['location'])

    # Add specific context elements
    if 'flood_zone' in context:
        query_parts.append(f"flood zone {context['flood_zone']}")
    if 'property_type' in context:
        query_parts.append(context['property_type'])
    if 'service_type' in context:
        query_parts.append(context['service_type'])
    if 'audit_type' in context:
        query_parts.append(context['audit_type'])

    # Add description if available
    if 'description' in context:
        query_parts.append(context['description'])

    return " ".join(query_parts)

def parse_document_sections(content: str) -> List[Dict]:
    """Parse document content into sections"""
    sections = []

    # Split by section headers (lines with === or all caps)
    section_pattern = r'(SECTION \d+[:\s].+|[A-Z][A-Z\s]+(?:\n|$))'

    current_section = {"title": "Introduction", "content": "", "start": 0}

    lines = content.split('\n')
    current_content = []

    for i, line in enumerate(lines):
        # Check if this is a section header
        if re.match(r'^={3,}$', line.strip()):
            continue
        elif re.match(r'^SECTION \d+', line.strip()) or (line.isupper() and len(line.strip()) > 5):
            # Save previous section
            if current_content:
                current_section['content'] = '\n'.join(current_content)
                sections.append(current_section)

            # Start new section
            current_section = {
                "title": line.strip(),
                "content": "",
                "start": i
            }
            current_content = []
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_content:
        current_section['content'] = '\n'.join(current_content)
        sections.append(current_section)

    return sections

def calculate_relevance_display(score: float) -> str:
    """Convert relevance score to display format"""
    percentage = int(score * 100)
    if percentage >= 90:
        return f"🟢 {percentage}% - Highly Relevant"
    elif percentage >= 70:
        return f"🟡 {percentage}% - Relevant"
    elif percentage >= 50:
        return f"🟠 {percentage}% - Somewhat Relevant"
    else:
        return f"🔴 {percentage}% - Low Relevance"

def clean_text_for_embedding(text: str) -> str:
    """Clean text before creating embeddings"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    # Trim
    text = text.strip()
    return text
