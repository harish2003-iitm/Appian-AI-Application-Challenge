"""
Layer 2: Semantic Core - Hybrid Search Engine
- Combines Vector Search (semantic) with BM25 (keyword exact match)
- Re-ranking with cross-encoder scoring
- Prevents fuzzy matches on exact identifiers (policy numbers, codes)
"""

import re
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    """Search result with combined scoring"""
    doc_id: str
    content: str
    metadata: Dict
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    match_type: str = "hybrid"  # vector, keyword, hybrid


class BM25Index:
    """
    BM25 (Best Match 25) implementation for keyword search.
    Ensures exact matches on policy numbers, codes, etc.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_ids = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.term_freqs = []  # Term frequency per document
        self.doc_freqs = Counter()  # Document frequency per term
        self.idf_cache = {}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text, preserving important identifiers"""
        # Preserve policy numbers, codes, etc.
        text = text.lower()

        # Find and protect identifiers (alphanumeric codes)
        identifiers = re.findall(r'\b[A-Za-z0-9]+-[A-Za-z0-9]+\b|\b[A-Z]{2,}\d+\b|\b\d+[A-Z]+\b', text, re.IGNORECASE)

        # Standard tokenization
        tokens = re.findall(r'\b\w+\b', text)

        # Add protected identifiers back
        tokens.extend([id.lower() for id in identifiers])

        return tokens

    def add_documents(self, documents: List[str], doc_ids: List[str], metadatas: List[Dict] = None):
        """Index documents for BM25 search"""
        self.documents = documents
        self.doc_ids = doc_ids
        self.metadatas = metadatas or [{} for _ in documents]

        # Compute term frequencies and document frequencies
        self.term_freqs = []
        self.doc_freqs = Counter()

        for doc in documents:
            tokens = self.tokenize(doc)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))

            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # Pre-compute IDF values
        N = len(documents)
        for term, df in self.doc_freqs.items():
            self.idf_cache[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_idf(self, term: str) -> float:
        """Get IDF score for a term"""
        return self.idf_cache.get(term, 0)

    def score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a single document"""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.term_freqs[doc_idx]

        for term in query_tokens:
            if term not in tf_dict:
                continue

            tf = tf_dict[term]
            idf = self.get_idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search and return top-k results with scores"""
        query_tokens = self.tokenize(query)

        scores = []
        for idx in range(len(self.documents)):
            score = self.score_document(query_tokens, idx)
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def exact_match_boost(self, query: str, doc_idx: int) -> float:
        """Boost score for exact phrase matches"""
        doc = self.documents[doc_idx].lower()
        query_lower = query.lower()

        boost = 0.0

        # Exact phrase match
        if query_lower in doc:
            boost += 2.0

        # Check for identifier patterns (policy numbers, codes)
        identifiers = re.findall(r'\b[A-Z0-9]+-[A-Z0-9]+\b|\b[A-Z]{2,}\d+\b', query, re.IGNORECASE)
        for identifier in identifiers:
            if identifier.lower() in doc:
                boost += 3.0  # Strong boost for exact identifier match

        return boost


class HybridSearchEngine:
    """
    Combines Vector Search with BM25 for optimal retrieval.
    Uses re-ranking to merge results.
    """

    def __init__(self, vector_store, embedding_service, alpha: float = 0.5):
        """
        Args:
            vector_store: ChromaDB vector store
            embedding_service: Embedding service for query embedding
            alpha: Weight for vector search (1-alpha = BM25 weight)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.alpha = alpha
        self.bm25_index = BM25Index()
        self.indexed = False

    def build_bm25_index(self, documents: List[str], doc_ids: List[str], metadatas: List[Dict] = None):
        """Build BM25 index from documents"""
        self.bm25_index.add_documents(documents, doc_ids, metadatas)
        self.indexed = True

    def detect_exact_match_query(self, query: str) -> bool:
        """Detect if query contains identifiers requiring exact match"""
        # Patterns that need exact matching
        patterns = [
            r'\b[A-Z]{2,}\d+\b',  # Code like "CPT90834"
            r'\b\d+-[A-Z]+\b',  # ID like "123-ABC"
            r'\bpolicy\s*#?\s*\d+\b',  # Policy number
            r'\bclaim\s*#?\s*\d+\b',  # Claim number
            r'\b[A-Z]\d{2}\.\d+\b',  # ICD codes like "J18.9"
        ]

        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def search(self, query: str, top_k: int = 10, filter_dict: Dict = None) -> List[SearchResult]:
        """
        Hybrid search combining vector and BM25.
        Automatically adjusts weights based on query type.
        """
        results = {}

        # Detect if query needs exact matching
        needs_exact = self.detect_exact_match_query(query)

        # Adjust alpha based on query type
        effective_alpha = 0.3 if needs_exact else self.alpha

        # Vector search
        query_embedding = self.embedding_service.get_embedding(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more for re-ranking
            filter_dict=filter_dict
        )

        # Process vector results
        vector_scores = [r.get('score', 0) for r in vector_results]
        normalized_vector = self.normalize_scores(vector_scores)

        for i, result in enumerate(vector_results):
            doc_id = result.get('id', f"doc_{i}")
            results[doc_id] = SearchResult(
                doc_id=doc_id,
                content=result.get('content', ''),
                metadata=result.get('metadata', {}),
                vector_score=normalized_vector[i] if i < len(normalized_vector) else 0,
                match_type="vector"
            )

        # BM25 search (if index is built)
        if self.indexed:
            bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

            bm25_scores = [score for _, score in bm25_results]
            normalized_bm25 = self.normalize_scores(bm25_scores)

            for i, (doc_idx, _) in enumerate(bm25_results):
                doc_id = self.bm25_index.doc_ids[doc_idx]
                content = self.bm25_index.documents[doc_idx]
                metadata = self.bm25_index.metadatas[doc_idx]

                # Add exact match boost
                exact_boost = self.bm25_index.exact_match_boost(query, doc_idx)
                bm25_score = normalized_bm25[i] + exact_boost

                if doc_id in results:
                    # Merge scores
                    results[doc_id].bm25_score = bm25_score
                    results[doc_id].match_type = "hybrid"
                else:
                    results[doc_id] = SearchResult(
                        doc_id=doc_id,
                        content=content,
                        metadata=metadata,
                        bm25_score=bm25_score,
                        match_type="keyword"
                    )

        # Calculate combined scores
        for result in results.values():
            result.combined_score = (
                effective_alpha * result.vector_score +
                (1 - effective_alpha) * result.bm25_score
            )

        # Sort by combined score
        sorted_results = sorted(results.values(), key=lambda x: x.combined_score, reverse=True)

        return sorted_results[:top_k]

    def search_with_reranking(self, query: str, top_k: int = 10, filter_dict: Dict = None) -> List[SearchResult]:
        """
        Search with cross-encoder re-ranking for better precision.
        Uses LLM to re-score top candidates.
        """
        # Get initial candidates
        candidates = self.search(query, top_k=top_k * 2, filter_dict=filter_dict)

        # For now, use combined score as final ranking
        # In production, would use a cross-encoder model here

        return candidates[:top_k]


class IdentifierMatcher:
    """
    Specialized matcher for exact identifiers.
    Ensures policy numbers, claim IDs, codes don't get fuzzy-matched.
    """

    IDENTIFIER_PATTERNS = {
        "policy_number": r'\bPOL[-_]?\d{6,10}\b',
        "claim_number": r'\bCLM[-_]?\d{6,10}\b',
        "icd_code": r'\b[A-Z]\d{2}\.?\d{0,2}\b',
        "cpt_code": r'\b\d{5}\b',
        "npi": r'\b\d{10}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "date": r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    }

    def extract_identifiers(self, text: str) -> Dict[str, List[str]]:
        """Extract all identifiers from text"""
        found = {}

        for id_type, pattern in self.IDENTIFIER_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found[id_type] = matches

        return found

    def must_match_exactly(self, query: str) -> List[str]:
        """Get identifiers that must match exactly"""
        exact_matches = []

        for id_type, pattern in self.IDENTIFIER_PATTERNS.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            exact_matches.extend(matches)

        return exact_matches

    def filter_by_identifiers(self, results: List[SearchResult], required_ids: List[str]) -> List[SearchResult]:
        """Filter results to only include those with exact identifier matches"""
        if not required_ids:
            return results

        filtered = []
        for result in results:
            content = result.content.lower()
            if all(rid.lower() in content for rid in required_ids):
                # Boost score for exact matches
                result.combined_score *= 1.5
                filtered.append(result)

        return filtered
