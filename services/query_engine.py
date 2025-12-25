"""
Layer 3: Query Transformation and Expansion Engine
- Transforms vague user queries into precise retrieval queries
- Generates sub-queries for comprehensive coverage
- Citation-aware generation with provenance tracking
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import google.generativeai as genai


@dataclass
class ExpandedQuery:
    """Expanded query with sub-queries"""
    original_query: str
    transformed_query: str
    sub_queries: List[str]
    intent: str
    entities: Dict[str, str]
    filters: Dict[str, any]


@dataclass
class CitedChunk:
    """Chunk with full citation metadata"""
    chunk_id: str
    content: str
    source_doc_id: str
    source_title: str
    page_number: int
    paragraph_id: int
    section_header: str
    bbox: Optional[Dict] = None  # Bounding box for UI highlighting
    version_date: str = ""
    relevance_score: float = 0.0
    char_start: int = 0
    char_end: int = 0

    def to_citation_tag(self) -> str:
        """Generate citation tag for LLM output"""
        return f"[{self.source_doc_id}:p{self.page_number}:¶{self.paragraph_id}]"

    def to_ui_metadata(self) -> Dict:
        """Generate metadata for UI click-to-verify"""
        return {
            "citation_id": self.chunk_id,
            "source_doc_id": self.source_doc_id,
            "source_title": self.source_title,
            "page": self.page_number,
            "paragraph": self.paragraph_id,
            "section": self.section_header,
            "bbox": self.bbox,
            "char_range": [self.char_start, self.char_end]
        }


class QueryExpander:
    """
    Expands user queries into multiple retrieval queries.
    Handles vague questions like "Why was this denied?"
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()

        if any(w in query_lower for w in ["why", "reason", "cause"]):
            return "explanation"
        elif any(w in query_lower for w in ["how", "process", "steps"]):
            return "procedure"
        elif any(w in query_lower for w in ["what", "define", "meaning"]):
            return "definition"
        elif any(w in query_lower for w in ["when", "deadline", "date"]):
            return "temporal"
        elif any(w in query_lower for w in ["who", "responsible", "contact"]):
            return "person"
        elif any(w in query_lower for w in ["how much", "cost", "price", "limit"]):
            return "quantitative"
        elif any(w in query_lower for w in ["can", "allowed", "eligible"]):
            return "eligibility"
        else:
            return "general"

    def extract_entities(self, query: str) -> Dict[str, str]:
        """Extract named entities from query"""
        entities = {}

        # Claim numbers
        claim_match = re.search(r'claim\s*#?\s*(\w+)', query, re.IGNORECASE)
        if claim_match:
            entities["claim_number"] = claim_match.group(1)

        # Policy numbers
        policy_match = re.search(r'policy\s*#?\s*(\w+)', query, re.IGNORECASE)
        if policy_match:
            entities["policy_number"] = policy_match.group(1)

        # Dates
        date_match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', query)
        if date_match:
            entities["date"] = date_match.group(0)

        # Dollar amounts
        amount_match = re.search(r'\$[\d,]+(?:\.\d{2})?', query)
        if amount_match:
            entities["amount"] = amount_match.group(0)

        # Medical codes
        icd_match = re.search(r'\b[A-Z]\d{2}\.?\d{0,2}\b', query)
        if icd_match:
            entities["icd_code"] = icd_match.group(0)

        cpt_match = re.search(r'\b\d{5}\b', query)
        if cpt_match:
            entities["cpt_code"] = cpt_match.group(0)

        return entities

    def generate_sub_queries(self, query: str, intent: str, entities: Dict) -> List[str]:
        """Generate sub-queries based on intent and entities"""
        sub_queries = []

        if intent == "explanation":
            # "Why was this denied?" -> multiple sub-queries
            sub_queries.append(f"denial codes and reasons")
            sub_queries.append(f"eligibility requirements")
            sub_queries.append(f"coverage exclusions")
            if entities.get("claim_number"):
                sub_queries.append(f"claim {entities['claim_number']} status history")

        elif intent == "procedure":
            sub_queries.append(f"step by step process")
            sub_queries.append(f"required documentation")
            sub_queries.append(f"timelines and deadlines")

        elif intent == "eligibility":
            sub_queries.append(f"eligibility criteria")
            sub_queries.append(f"qualifying conditions")
            sub_queries.append(f"exclusions and limitations")

        elif intent == "quantitative":
            sub_queries.append(f"coverage limits")
            sub_queries.append(f"deductible amounts")
            sub_queries.append(f"maximum benefits")

        # Add entity-specific queries
        if entities.get("icd_code"):
            sub_queries.append(f"medical necessity for {entities['icd_code']}")
        if entities.get("cpt_code"):
            sub_queries.append(f"procedure code {entities['cpt_code']} coverage")

        return sub_queries

    def expand_with_llm(self, query: str, context: Dict = None) -> ExpandedQuery:
        """Use LLM to expand query comprehensively"""
        context_str = json.dumps(context) if context else "{}"

        prompt = f"""Analyze this user query and expand it for comprehensive retrieval.

User Query: "{query}"
Context: {context_str}

Generate:
1. A transformed, more precise version of the query
2. 3-5 sub-queries to search for related information
3. Identified intent (explanation/procedure/definition/eligibility/quantitative/general)
4. Key entities (claim numbers, policy numbers, codes, dates, amounts)
5. Suggested filters (document type, date range, etc.)

Respond in JSON:
{{
    "transformed_query": "More precise version of the query",
    "sub_queries": ["sub-query 1", "sub-query 2", ...],
    "intent": "explanation",
    "entities": {{"claim_number": "123", ...}},
    "filters": {{"doc_type": "policy", ...}}
}}
"""
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))

            return ExpandedQuery(
                original_query=query,
                transformed_query=result.get("transformed_query", query),
                sub_queries=result.get("sub_queries", []),
                intent=result.get("intent", "general"),
                entities=result.get("entities", {}),
                filters=result.get("filters", {})
            )
        except Exception as e:
            # Fallback to rule-based expansion
            intent = self.detect_intent(query)
            entities = self.extract_entities(query)
            sub_queries = self.generate_sub_queries(query, intent, entities)

            return ExpandedQuery(
                original_query=query,
                transformed_query=query,
                sub_queries=sub_queries,
                intent=intent,
                entities=entities,
                filters={}
            )

    def expand(self, query: str, context: Dict = None, use_llm: bool = True) -> ExpandedQuery:
        """Main expansion method"""
        if use_llm:
            return self.expand_with_llm(query, context)
        else:
            intent = self.detect_intent(query)
            entities = self.extract_entities(query)
            sub_queries = self.generate_sub_queries(query, intent, entities)

            return ExpandedQuery(
                original_query=query,
                transformed_query=query,
                sub_queries=sub_queries,
                intent=intent,
                entities=entities,
                filters={}
            )


class CitationAwareGenerator:
    """
    Generates answers with inline citations.
    Enforces citation requirements in prompts.
    """

    CITATION_SYSTEM_PROMPT = """You are NEXUS AI, an expert knowledge assistant for insurance and healthcare case management.

CRITICAL CITATION RULES:
1. You MUST cite sources using the format [SOURCE_ID:pPAGE:¶PARAGRAPH] after EVERY factual claim
2. If no source supports a claim, you MUST state "I don't have information about this in the provided documents"
3. NEVER make claims without citations
4. If sources conflict, cite both and note the discrepancy

Example format:
"The deductible for flood damage is $1,000 [DOC123:p4:¶2]. However, for Zone V properties, an additional $500 applies [DOC123:p5:¶1]."

Available source documents are provided below. Answer ONLY using these sources."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def format_chunks_for_prompt(self, chunks: List[CitedChunk]) -> str:
        """Format chunks with citation markers for the prompt"""
        formatted = []

        for chunk in chunks:
            citation_tag = chunk.to_citation_tag()
            formatted.append(f"""
---
SOURCE: {citation_tag}
Document: {chunk.source_title}
Section: {chunk.section_header}
Page: {chunk.page_number}, Paragraph: {chunk.paragraph_id}

{chunk.content}
---
""")

        return "\n".join(formatted)

    def generate_with_citations(self, query: str, chunks: List[CitedChunk], case_context: Dict = None) -> Dict:
        """Generate answer with inline citations"""
        context_str = ""
        if case_context:
            context_str = f"\nCase Context: {json.dumps(case_context)}"

        sources_text = self.format_chunks_for_prompt(chunks)

        prompt = f"""{self.CITATION_SYSTEM_PROMPT}
{context_str}

SOURCE DOCUMENTS:
{sources_text}

USER QUESTION: {query}

Provide a comprehensive answer with citations to the specific sources. Remember to cite EVERY factual claim."""

        try:
            response = self.model.generate_content(prompt)
            answer = response.text

            # Extract citations used
            citations_used = re.findall(r'\[([^\]]+)\]', answer)

            return {
                "answer": answer,
                "citations_used": list(set(citations_used)),
                "sources_provided": len(chunks),
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "citations_used": [],
                "sources_provided": len(chunks),
                "error": str(e)
            }

    def extract_citation_metadata(self, answer: str, chunks: List[CitedChunk]) -> List[Dict]:
        """Extract citation metadata for UI rendering"""
        citations = re.findall(r'\[([^\]]+)\]', answer)
        metadata = []

        chunk_map = {chunk.to_citation_tag().strip('[]'): chunk for chunk in chunks}

        for citation in citations:
            if citation in chunk_map:
                chunk = chunk_map[citation]
                metadata.append(chunk.to_ui_metadata())

        return metadata


class RetrievalOrchestrator:
    """
    Orchestrates the full retrieval and generation pipeline.
    """

    def __init__(self, api_key: str, retrieval_engine, model_name: str = "gemini-2.0-flash"):
        self.query_expander = QueryExpander(api_key)
        self.generator = CitationAwareGenerator(api_key, model_name)
        self.retrieval_engine = retrieval_engine

    def retrieve_and_generate(self, query: str, case_context: Dict = None, top_k: int = 5) -> Dict:
        """Full pipeline: expand -> retrieve -> generate with citations"""
        # Step 1: Expand query
        expanded = self.query_expander.expand(query, case_context)

        # Step 2: Retrieve for main query
        all_results = []
        main_results = self.retrieval_engine.retrieve(
            query=expanded.transformed_query,
            case_context=case_context,
            top_k=top_k
        )
        all_results.extend(main_results)

        # Step 3: Retrieve for sub-queries
        for sub_query in expanded.sub_queries[:3]:  # Limit sub-queries
            sub_results = self.retrieval_engine.retrieve(
                query=sub_query,
                case_context=case_context,
                top_k=2
            )
            all_results.extend(sub_results)

        # Deduplicate results
        seen_ids = set()
        unique_results = []
        for r in all_results:
            rid = r.get('id', r.get('chunk_id', ''))
            if rid not in seen_ids:
                seen_ids.add(rid)
                unique_results.append(r)

        # Step 4: Convert to CitedChunks
        cited_chunks = []
        for i, r in enumerate(unique_results[:top_k * 2]):
            cited_chunks.append(CitedChunk(
                chunk_id=r.get('id', f'chunk_{i}'),
                content=r.get('content', ''),
                source_doc_id=r.get('document_id', r.get('source_id', f'DOC{i}')),
                source_title=r.get('document_title', r.get('title', 'Document')),
                page_number=r.get('page', 1),
                paragraph_id=r.get('paragraph', i + 1),
                section_header=r.get('section', r.get('section_header', '')),
                bbox=r.get('bbox'),
                relevance_score=r.get('relevance_score', r.get('score', 0))
            ))

        # Step 5: Generate with citations
        generation_result = self.generator.generate_with_citations(
            query=query,
            chunks=cited_chunks,
            case_context=case_context
        )

        # Step 6: Extract UI metadata
        ui_citations = self.generator.extract_citation_metadata(
            generation_result["answer"],
            cited_chunks
        )

        return {
            "query": query,
            "expanded_query": expanded.transformed_query,
            "sub_queries": expanded.sub_queries,
            "intent": expanded.intent,
            "entities": expanded.entities,
            "answer": generation_result["answer"],
            "citations": ui_citations,
            "citations_used": generation_result["citations_used"],
            "sources_count": len(cited_chunks),
            "retrieval_results": unique_results[:top_k],
            "generated_at": generation_result.get("generated_at")
        }
