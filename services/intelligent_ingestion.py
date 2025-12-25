"""
Layer 1: Intelligent Ingestion and Multi-Modal Processing
- OCR and Layout Analysis
- Document Classification
- Legacy Data Handling
"""

import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import google.generativeai as genai

@dataclass
class DocumentChunk:
    """Rich chunk with full metadata for citation"""
    text: str
    source_id: str
    doc_type: str
    page: int
    paragraph: int
    bbox: Optional[Dict] = None  # Bounding box for UI highlighting
    section_header: str = ""
    char_start: int = 0
    char_end: int = 0
    version_date: str = ""
    confidence_score: float = 1.0

    def to_metadata(self) -> Dict:
        return {
            "source_id": self.source_id,
            "doc_type": self.doc_type,
            "page": self.page,
            "paragraph": self.paragraph,
            "bbox": json.dumps(self.bbox) if self.bbox else None,
            "section_header": self.section_header,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "version_date": self.version_date,
            "confidence_score": self.confidence_score
        }


class DocumentClassifier:
    """
    Classifies documents into categories before processing.
    Routes to correct extraction pipeline.
    """

    DOCUMENT_TYPES = {
        "policy": ["policy", "coverage", "insurance", "premium", "deductible", "endorsement"],
        "claim": ["claim", "loss", "damage", "incident", "accident", "occurrence"],
        "medical": ["diagnosis", "treatment", "patient", "provider", "cpt", "icd", "cms-1500"],
        "legal": ["attorney", "litigation", "subpoena", "settlement", "court"],
        "regulatory": ["regulation", "compliance", "hipaa", "aca", "cms", "state law"],
        "correspondence": ["dear", "sincerely", "regards", "letter", "memo"],
        "invoice": ["invoice", "bill", "payment", "amount due", "total"],
        "police_report": ["police", "officer", "incident report", "badge", "jurisdiction"]
    }

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def classify_by_keywords(self, text: str) -> Tuple[str, float]:
        """Fast keyword-based classification"""
        text_lower = text.lower()
        scores = {}

        for doc_type, keywords in self.DOCUMENT_TYPES.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[doc_type] = score

        if max(scores.values()) == 0:
            return "unknown", 0.0

        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / len(self.DOCUMENT_TYPES[best_type])
        return best_type, min(confidence, 1.0)

    def classify_with_llm(self, text: str) -> Tuple[str, float, str]:
        """LLM-based classification for complex documents"""
        prompt = f"""Classify this document into one of these categories:
- policy (insurance policies, coverage documents)
- claim (claim forms, loss reports)
- medical (medical records, invoices, CMS forms)
- legal (legal correspondence, court documents)
- regulatory (compliance documents, regulations)
- correspondence (letters, memos)
- invoice (bills, payment requests)
- police_report (incident reports)

Document excerpt:
{text[:2000]}

Respond in JSON format:
{{"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
"""
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
            return result.get("category", "unknown"), result.get("confidence", 0.5), result.get("reasoning", "")
        except:
            # Fallback to keyword classification
            doc_type, conf = self.classify_by_keywords(text)
            return doc_type, conf, "Keyword-based classification"

    def classify(self, text: str, use_llm: bool = False) -> Dict:
        """Main classification method"""
        kw_type, kw_conf = self.classify_by_keywords(text)

        if use_llm and kw_conf < 0.5:
            llm_type, llm_conf, reasoning = self.classify_with_llm(text)
            return {
                "doc_type": llm_type,
                "confidence": llm_conf,
                "method": "llm",
                "reasoning": reasoning
            }

        return {
            "doc_type": kw_type,
            "confidence": kw_conf,
            "method": "keyword",
            "reasoning": f"Matched keywords for {kw_type}"
        }


class LayoutAnalyzer:
    """
    Analyzes document layout to understand spatial relationships.
    Identifies tables, headers, sections, etc.
    """

    def __init__(self):
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]{3,}):?\s*$',  # ALL CAPS headers (min 4 chars)
            r'^(\d{1,2}\.\d+[\d\.]*)\s+(.+)$',  # Numbered sections like 1.1, 2.3.1 (NOT years like 2012.)
            r'^(Section|Article|Part|Chapter)\s+[\dIVX]+',  # Legal sections
            r'^\*\*(.+)\*\*$',  # Bold text as headers
        ]
        # Patterns to skip (false positives)
        self.skip_patterns = [
            r'^\d{4}\.$',  # Years like "2012."
            r'^---\s*Page',  # Page markers
        ]

    def extract_sections(self, text: str) -> List[Dict]:
        """Extract document sections with hierarchy"""
        lines = text.split('\n')
        sections = []
        current_section = {"header": "", "content": [], "level": 0, "start_line": 0}

        for i, line in enumerate(lines):
            is_header = False
            header_text = ""
            level = 0
            stripped_line = line.strip()

            # First check if this is a false positive to skip
            should_skip = False
            for skip_pattern in self.skip_patterns:
                if re.match(skip_pattern, stripped_line):
                    should_skip = True
                    break

            if not should_skip:
                for pattern in self.section_patterns:
                    match = re.match(pattern, stripped_line, re.MULTILINE)
                    if match:
                        is_header = True
                        header_text = match.group(1) if match.groups() else stripped_line
                        # Determine level based on pattern
                        if line.startswith('#'):
                            level = len(line) - len(line.lstrip('#'))
                        elif re.match(r'^\d+\.\d+', line):
                            level = line.count('.')
                        else:
                            level = 1
                        break

            if is_header:
                # Save previous section
                if current_section["content"]:
                    current_section["end_line"] = i - 1
                    sections.append(current_section)

                current_section = {
                    "header": header_text,
                    "content": [],
                    "level": level,
                    "start_line": i
                }
            else:
                current_section["content"].append(line)

        # Save last section
        if current_section["content"]:
            current_section["end_line"] = len(lines) - 1
            sections.append(current_section)

        return sections

    def detect_tables(self, text: str) -> List[Dict]:
        """Detect table structures in text"""
        tables = []
        lines = text.split('\n')

        in_table = False
        table_start = 0
        table_lines = []

        for i, line in enumerate(lines):
            # Detect markdown tables or pipe-separated content
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                    table_start = i
                table_lines.append(line)
            elif in_table:
                # End of table
                tables.append({
                    "start_line": table_start,
                    "end_line": i - 1,
                    "content": '\n'.join(table_lines),
                    "rows": len(table_lines),
                    "cols": table_lines[0].count('|') - 1 if table_lines else 0
                })
                in_table = False
                table_lines = []

        return tables

    def analyze(self, text: str) -> Dict:
        """Full layout analysis"""
        return {
            "sections": self.extract_sections(text),
            "tables": self.detect_tables(text),
            "line_count": len(text.split('\n')),
            "char_count": len(text),
            "has_structured_content": bool(self.detect_tables(text))
        }


class SemanticChunker:
    """
    Intelligent chunking that respects document structure.
    Keeps headers with their content, preserves tables, etc.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.layout_analyzer = LayoutAnalyzer()

    def _detect_page_number(self, text: str) -> int:
        """Detect page number from text containing page markers like '--- Page X ---'"""
        match = re.search(r'---\s*Page\s+(\d+)\s*---', text)
        if match:
            return int(match.group(1))
        return 0  # Return 0 if no page marker found (so we don't reset to 1)

    def _find_all_page_markers(self, text: str) -> List[Tuple[int, int]]:
        """Find all page markers in text and return (char_position, page_number) pairs"""
        markers = []
        for match in re.finditer(r'---\s*Page\s+(\d+)\s*---', text):
            markers.append((match.start(), int(match.group(1))))
        return markers

    def _get_page_for_position(self, char_pos: int, page_markers: List[Tuple[int, int]]) -> int:
        """Get page number for a given character position based on page markers"""
        current_page = 1
        for marker_pos, page_num in page_markers:
            if marker_pos <= char_pos:
                current_page = page_num
            else:
                break
        return current_page

    def chunk_by_sections(self, text: str, doc_id: str, doc_type: str) -> List[DocumentChunk]:
        """Chunk based on document sections with page number detection"""
        layout = self.layout_analyzer.analyze(text)
        chunks = []

        # Find all page markers in the full text upfront
        page_markers = self._find_all_page_markers(text)

        # Build a map of line number to character position
        lines = text.split('\n')
        line_to_char = [0]
        for line in lines:
            line_to_char.append(line_to_char[-1] + len(line) + 1)

        for section in layout["sections"]:
            section_text = '\n'.join(section["content"])
            section_header = section["header"]

            # Get page number based on section's start position in the document
            start_line = section.get("start_line", 0)
            char_pos = line_to_char[start_line] if start_line < len(line_to_char) else 0
            current_page = self._get_page_for_position(char_pos, page_markers)

            if len(section_text) <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(DocumentChunk(
                    text=f"{section_header}\n\n{section_text}".strip(),
                    source_id=doc_id,
                    doc_type=doc_type,
                    page=current_page,
                    paragraph=len(chunks) + 1,
                    section_header=section_header,
                    char_start=section.get("start_line", 0),
                    char_end=section.get("end_line", 0),
                    version_date=datetime.now().isoformat()
                ))
            else:
                # Split large sections while keeping context
                words = section_text.split()
                current_chunk = []
                current_len = 0
                para_in_section = 0
                chunk_char_offset = 0  # Track character offset within section

                for word in words:
                    current_chunk.append(word)
                    current_len += len(word) + 1
                    chunk_char_offset += len(word) + 1

                    if current_len >= self.chunk_size:
                        chunk_text = f"{section_header}\n\n" + ' '.join(current_chunk)
                        para_in_section += 1

                        # Calculate page for this chunk based on its position
                        chunk_pos = char_pos + chunk_char_offset
                        chunk_page = self._get_page_for_position(chunk_pos, page_markers)

                        chunks.append(DocumentChunk(
                            text=chunk_text,
                            source_id=doc_id,
                            doc_type=doc_type,
                            page=chunk_page,
                            paragraph=len(chunks) + 1,
                            section_header=section_header,
                            version_date=datetime.now().isoformat()
                        ))

                        # Keep overlap
                        overlap_words = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_words
                        current_len = sum(len(w) + 1 for w in overlap_words)

                # Remaining content
                if current_chunk:
                    chunk_text = f"{section_header}\n\n" + ' '.join(current_chunk)
                    chunk_pos = char_pos + chunk_char_offset
                    chunk_page = self._get_page_for_position(chunk_pos, page_markers)

                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        source_id=doc_id,
                        doc_type=doc_type,
                        page=chunk_page,
                        paragraph=len(chunks) + 1,
                        section_header=section_header,
                        version_date=datetime.now().isoformat()
                    ))

        return chunks

    def chunk_with_bbox(self, text: str, doc_id: str, doc_type: str) -> List[DocumentChunk]:
        """Chunk with bounding box metadata for UI highlighting"""
        chunks = self.chunk_by_sections(text, doc_id, doc_type)

        # Add approximate bounding boxes based on character positions
        total_chars = len(text)
        for i, chunk in enumerate(chunks):
            # Simulate bbox based on position (in real system, OCR provides this)
            start_ratio = chunk.char_start / max(total_chars, 1)
            end_ratio = chunk.char_end / max(total_chars, 1)

            chunk.bbox = {
                "x": 50,  # Left margin
                "y": int(start_ratio * 800),  # Approximate Y position
                "width": 500,
                "height": int((end_ratio - start_ratio) * 800),
                "page": chunk.page
            }

        return chunks


class LegacyCodeAnalyzer:
    """
    Analyzes legacy code (COBOL, etc.) and extracts business rules
    as natural language for indexing.
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def analyze_cobol(self, code: str) -> Dict:
        """Extract business rules from COBOL code"""
        prompt = f"""Analyze this COBOL code and extract the business rules as natural language.
For each rule, provide:
1. A plain English description
2. The conditions that trigger it
3. The outcome/action

COBOL Code:
{code[:3000]}

Respond in JSON format:
{{
    "business_rules": [
        {{
            "id": "RULE001",
            "description": "Plain English description",
            "conditions": ["condition1", "condition2"],
            "action": "What happens when conditions are met",
            "variables": ["var1", "var2"]
        }}
    ],
    "summary": "Overall summary of what this code does"
}}
"""
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
            return result
        except Exception as e:
            return {"error": str(e), "business_rules": [], "summary": "Failed to analyze"}

    def convert_to_searchable(self, rules: Dict) -> List[str]:
        """Convert extracted rules to searchable text chunks"""
        searchable = []

        for rule in rules.get("business_rules", []):
            text = f"""Business Rule: {rule.get('id', 'Unknown')}

{rule.get('description', '')}

Conditions:
{chr(10).join('- ' + c for c in rule.get('conditions', []))}

Action: {rule.get('action', '')}

Related Variables: {', '.join(rule.get('variables', []))}
"""
            searchable.append(text)

        return searchable


class IntelligentIngestionPipeline:
    """
    Main pipeline that orchestrates all Layer 1 components.
    """

    def __init__(self, api_key: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.classifier = DocumentClassifier(api_key)
        self.chunker = SemanticChunker(chunk_size, chunk_overlap)
        self.layout_analyzer = LayoutAnalyzer()
        self.legacy_analyzer = LegacyCodeAnalyzer(api_key)

    def generate_doc_id(self, content: str, filename: str) -> str:
        """Generate unique document ID"""
        hash_input = f"{filename}:{content[:1000]}:{datetime.now().date()}"
        return f"DOC_{hashlib.md5(hash_input.encode()).hexdigest()[:12].upper()}"

    def process_document(self, content: str, filename: str, use_llm_classification: bool = False) -> Dict:
        """Full document processing pipeline"""
        # Step 1: Generate document ID
        doc_id = self.generate_doc_id(content, filename)

        # Step 2: Classify document
        classification = self.classifier.classify(content, use_llm=use_llm_classification)

        # Step 3: Analyze layout
        layout = self.layout_analyzer.analyze(content)

        # Step 4: Semantic chunking with metadata
        chunks = self.chunker.chunk_with_bbox(
            content,
            doc_id,
            classification["doc_type"]
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "classification": classification,
            "layout": {
                "section_count": len(layout["sections"]),
                "table_count": len(layout["tables"]),
                "has_structured_content": layout["has_structured_content"]
            },
            "chunks": chunks,
            "chunk_count": len(chunks),
            "processed_at": datetime.now().isoformat()
        }

    def process_legacy_code(self, code: str, language: str = "cobol") -> Dict:
        """Process legacy code and extract business rules"""
        if language.lower() == "cobol":
            rules = self.legacy_analyzer.analyze_cobol(code)
            searchable_texts = self.legacy_analyzer.convert_to_searchable(rules)

            return {
                "language": language,
                "rules_extracted": len(rules.get("business_rules", [])),
                "summary": rules.get("summary", ""),
                "searchable_chunks": searchable_texts,
                "raw_rules": rules
            }

        return {"error": f"Language {language} not supported"}
