"""Document processing and chunking service"""

import os
import re
from typing import List, Dict, Tuple
from pathlib import Path

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> Tuple[str, Dict]:
        """Load a document and extract metadata"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract metadata from content
        metadata = self._extract_metadata(content, path.name)
        metadata['file_path'] = str(file_path)
        metadata['file_name'] = path.name

        return content, metadata

    def _extract_metadata(self, content: str, filename: str) -> Dict:
        """Extract metadata from document content"""
        metadata = {
            'title': filename.replace('_', ' ').replace('.txt', '').title(),
            'doc_type': 'policy'
        }

        # Try to extract document ID
        doc_id_match = re.search(r'Document ID:\s*([A-Z0-9-]+)', content)
        if doc_id_match:
            metadata['document_id'] = doc_id_match.group(1)

        # Try to extract effective date
        date_match = re.search(r'Effective Date:\s*([A-Za-z]+ \d+, \d{4})', content)
        if date_match:
            metadata['effective_date'] = date_match.group(1)

        # Determine document type from filename
        if 'flood' in filename.lower():
            metadata['doc_type'] = 'flood_insurance'
            metadata['category'] = 'Insurance Policy'
        elif 'auto' in filename.lower():
            metadata['doc_type'] = 'auto_insurance'
            metadata['category'] = 'Standard Operating Procedure'
        elif 'healthcare' in filename.lower():
            metadata['doc_type'] = 'healthcare'
            metadata['category'] = 'Benefits Guide'
        elif 'compliance' in filename.lower() or 'government' in filename.lower():
            metadata['doc_type'] = 'compliance'
            metadata['category'] = 'Regulatory Requirements'

        return metadata

    def chunk_document(self, content: str, metadata: Dict) -> List[Dict]:
        """Split document into chunks with provenance tracking"""
        chunks = []

        # First, try to split by sections
        sections = self._split_into_sections(content)

        chunk_index = 0
        for section in sections:
            section_chunks = self._chunk_section(
                section['content'],
                section['title'],
                section['section_num']
            )

            for chunk_content, chunk_meta in section_chunks:
                chunk = {
                    'content': chunk_content,
                    'chunk_index': chunk_index,
                    'section_title': section['title'],
                    'section_number': section['section_num'],
                    'paragraph_number': chunk_meta.get('paragraph', 1),
                    'char_start': chunk_meta.get('char_start', 0),
                    'char_end': chunk_meta.get('char_end', len(chunk_content)),
                    'metadata': {
                        **metadata,
                        'section': section['title']
                    }
                }
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _split_into_sections(self, content: str) -> List[Dict]:
        """Split content into logical sections"""
        sections = []

        # Pattern to match section headers
        section_pattern = r'(SECTION \d+[:\s][^\n]+)'
        parts = re.split(section_pattern, content)

        current_section = {
            'title': 'Introduction',
            'section_num': 0,
            'content': ''
        }

        for i, part in enumerate(parts):
            if re.match(r'SECTION \d+', part):
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section)

                # Extract section number
                section_num_match = re.search(r'SECTION (\d+)', part)
                section_num = int(section_num_match.group(1)) if section_num_match else i

                current_section = {
                    'title': part.strip(),
                    'section_num': section_num,
                    'content': ''
                }
            else:
                current_section['content'] += part

        # Don't forget the last section
        if current_section['content'].strip():
            sections.append(current_section)

        # If no sections found, treat whole document as one section
        if not sections:
            sections.append({
                'title': 'Document Content',
                'section_num': 1,
                'content': content
            })

        return sections

    def _chunk_section(self, content: str, section_title: str,
                       section_num: int) -> List[Tuple[str, Dict]]:
        """Chunk a section into smaller pieces"""
        chunks = []

        # Clean content
        content = self._clean_content(content)

        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk = ""
        current_meta = {'paragraph': 1, 'char_start': 0}
        char_position = 0
        para_num = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    current_meta['char_end'] = char_position
                    chunks.append((current_chunk.strip(), current_meta.copy()))

                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                current_meta = {
                    'paragraph': para_num,
                    'char_start': char_position
                }
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

            char_position += len(para) + 2  # +2 for \n\n
            para_num += 1

        # Don't forget the last chunk
        if current_chunk.strip():
            current_meta['char_end'] = char_position
            chunks.append((current_chunk.strip(), current_meta))

        return chunks

    def _clean_content(self, content: str) -> str:
        """Clean content for processing"""
        # Remove excessive equals signs (section dividers)
        content = re.sub(r'={3,}', '', content)
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)
        return content.strip()

    def process_directory(self, directory_path: str) -> List[Dict]:
        """Process all documents in a directory"""
        all_chunks = []
        dir_path = Path(directory_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Process all text files
        for file_path in dir_path.glob('*.txt'):
            try:
                content, metadata = self.load_document(str(file_path))
                chunks = self.chunk_document(content, metadata)
                all_chunks.extend(chunks)
                print(f"Processed: {file_path.name} - {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        return all_chunks
