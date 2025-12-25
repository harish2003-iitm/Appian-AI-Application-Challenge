"""Context-aware retrieval engine"""

from typing import List, Dict, Optional
import json

class RetrievalEngine:
    def __init__(self, vector_store, embedding_service, sqlite_db):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.db = sqlite_db

    def retrieve(self, query: str, case_context: Dict = None,
                 top_k: int = 5, min_score: float = 0.1) -> List[Dict]:
        """Retrieve relevant documents based on query and case context"""

        # Build enhanced query from case context
        enhanced_query = self._build_enhanced_query(query, case_context)

        # Get query embedding
        query_embedding = self.embedding_service.get_query_embedding(enhanced_query)

        # Build metadata filter from case context
        metadata_filter = self._build_metadata_filter(case_context)

        # Search vector store
        results = self.vector_store.search_by_text(
            query_text=enhanced_query,
            query_embedding=query_embedding,
            n_results=top_k * 2,  # Get more results for filtering
            filter_metadata=metadata_filter
        )

        # Filter by minimum score and get top_k
        filtered_results = [
            r for r in results
            if r.get('relevance_score', 0) >= min_score
        ][:top_k]

        # Enrich results with full document info
        enriched_results = self._enrich_results(filtered_results)

        # Log the search
        self._log_search(query, case_context, len(enriched_results))

        return enriched_results

    def retrieve_by_context(self, case_context: Dict, top_k: int = 5) -> List[Dict]:
        """Automatically retrieve relevant documents based on case context only"""

        # Generate automatic query from case context
        auto_query = self._generate_context_query(case_context)

        return self.retrieve(auto_query, case_context, top_k)

    def _build_enhanced_query(self, query: str, case_context: Dict = None) -> str:
        """Enhance query with case context information"""
        if not case_context:
            return query

        context_parts = [query]

        # Add relevant context fields to query
        context = case_context.get('context', case_context)

        if 'claim_type' in context:
            context_parts.append(context['claim_type'])

        if 'state' in case_context:
            context_parts.append(case_context['state'])

        if 'case_type' in case_context:
            context_parts.append(case_context['case_type'])

        return " ".join(context_parts)

    def _generate_context_query(self, case_context: Dict) -> str:
        """Generate a search query from case context"""
        query_parts = []

        # Extract key fields
        if 'case_type' in case_context:
            query_parts.append(case_context['case_type'])

        context = case_context.get('context', {})

        if 'claim_type' in context:
            query_parts.append(f"{context['claim_type']} claim")

        if 'state' in case_context:
            query_parts.append(f"{case_context['state']} regulations")

        if 'property_type' in context:
            query_parts.append(context['property_type'])

        if 'service_type' in context:
            query_parts.append(f"{context['service_type']} requirements")

        if 'damage_amount' in context:
            amount = context['damage_amount']
            if amount > 50000:
                query_parts.append("large claim procedures")
            else:
                query_parts.append("standard claim procedures")

        if 'urgency' in context:
            if context['urgency'].lower() == 'urgent':
                query_parts.append("urgent priority expedited")

        if 'audit_type' in context:
            query_parts.append(context['audit_type'])

        return " ".join(query_parts) if query_parts else "policy procedures"

    def _build_metadata_filter(self, case_context: Dict) -> Optional[Dict]:
        """Build ChromaDB metadata filter from case context"""
        # Disable strict filtering for now - let semantic search handle relevance
        # The metadata filter was causing "No documents found" because
        # indexed documents don't always have matching doc_type metadata
        return None

    def _enrich_results(self, results: List[Dict]) -> List[Dict]:
        """Enrich results with additional information"""
        enriched = []

        for result in results:
            enriched_result = {
                'id': result.get('id'),
                'content': result.get('content', ''),
                'relevance_score': result.get('relevance_score', 0),
                'metadata': result.get('metadata', {}),
                'citation': self._format_citation(result)
            }

            # Extract section info from metadata
            metadata = result.get('metadata', {})
            enriched_result['document_title'] = metadata.get('title', 'Unknown Document')
            enriched_result['section'] = metadata.get('section', '')
            enriched_result['doc_type'] = metadata.get('doc_type', '')
            enriched_result['category'] = metadata.get('category', '')

            enriched.append(enriched_result)

        return enriched

    def _format_citation(self, result: Dict) -> str:
        """Format citation string for a result"""
        metadata = result.get('metadata', {})

        title = metadata.get('title', 'Document')
        section = metadata.get('section', '')
        doc_id = metadata.get('document_id', '')

        citation_parts = [f"[{title}]"]

        if doc_id:
            citation_parts.append(f"ID: {doc_id}")
        if section:
            citation_parts.append(f"Section: {section}")

        return " | ".join(citation_parts)

    def _log_search(self, query: str, case_context: Dict, results_count: int):
        """Log search to database"""
        try:
            context_str = json.dumps(case_context) if case_context else "{}"
            self.db.log_search(context_str, query, results_count)
        except Exception as e:
            print(f"Error logging search: {e}")

    def get_document_suggestions(self, case_context: Dict) -> List[str]:
        """Get suggested queries based on case context"""
        suggestions = []

        case_type = case_context.get('case_type', '').lower()
        context = case_context.get('context', {})

        if 'flood' in case_type:
            suggestions.extend([
                "What are the coverage limits?",
                "What documentation is required for claims?",
                "What are the waiting periods?",
                "What is excluded from coverage?"
            ])
        elif 'auto' in case_type:
            suggestions.extend([
                "How is liability determined?",
                "What is the claims priority level?",
                "What are the payment authorization levels?",
                "What is the rental car coverage?"
            ])
        elif 'healthcare' in case_type:
            suggestions.extend([
                "Is pre-authorization required?",
                "What is the appeal process?",
                "What are the coverage limits?",
                "What is the claims timeline?"
            ])
        elif 'compliance' in case_type:
            suggestions.extend([
                "What are the notification requirements?",
                "What are the penalties for non-compliance?",
                "What documentation is required?",
                "What are the audit requirements?"
            ])

        return suggestions
