"""
Graph RAG - Knowledge Graph Enhanced Retrieval
- Extracts entities and relationships from documents
- Builds knowledge graph for better context understanding
- Uses graph traversal for related information retrieval
"""

import re
import json
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import google.generativeai as genai
from .model_manager import ModelManager


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    name: str
    entity_type: str  # PERSON, ORGANIZATION, POLICY, REGULATION, LOCATION, etc.
    properties: Dict = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source_id: str
    target_id: str
    relation_type: str  # COVERS, REGULATES, APPLIES_TO, REQUIRES, etc.
    properties: Dict = field(default_factory=dict)
    confidence: float = 1.0


class KnowledgeGraphExtractor:
    """
    Extracts entities and relationships from text using LLM.
    """

    ENTITY_TYPES = [
        "POLICY", "REGULATION", "ORGANIZATION", "PERSON", "LOCATION",
        "COVERAGE_TYPE", "CLAIM_TYPE", "AMOUNT", "DATE", "PROCEDURE",
        "REQUIREMENT", "EXCLUSION", "CONDITION", "BENEFIT"
    ]

    RELATION_TYPES = [
        "COVERS", "EXCLUDES", "REQUIRES", "APPLIES_TO", "REGULATES",
        "ISSUED_BY", "LOCATED_IN", "HAS_LIMIT", "EFFECTIVE_FROM",
        "SUPERSEDES", "REFERENCES", "DEFINES", "MANDATES"
    ]

    def __init__(self, api_key: str, model_list: List[str] = None):
        self.api_key = api_key
        if model_list is None:
            from config import GENERATION_MODELS
            model_list = GENERATION_MODELS
        self.model_manager = ModelManager(api_key, model_list)

    def extract_entities_and_relations(self, text: str, chunk_id: str = "") -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text using LLM"""

        prompt = f"""Analyze this insurance/policy document text and extract entities and relationships.

TEXT:
{text[:3000]}

Extract entities of these types: {', '.join(self.ENTITY_TYPES)}
Extract relationships of these types: {', '.join(self.RELATION_TYPES)}

Respond in JSON format:
{{
    "entities": [
        {{"name": "entity name", "type": "ENTITY_TYPE", "properties": {{"key": "value"}}}}
    ],
    "relationships": [
        {{"source": "entity1 name", "target": "entity2 name", "type": "RELATION_TYPE", "properties": {{}}}}
    ]
}}

Only include clearly identifiable entities and relationships. Be precise."""

        try:
            result_text = self.model_manager.generate_with_fallback(prompt)
            # Clean up markdown code blocks if present
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(result_text)

            entities = []
            for e in result.get("entities", []):
                entity_id = f"{e['type']}_{e['name'].lower().replace(' ', '_')}"
                entities.append(Entity(
                    id=entity_id,
                    name=e['name'],
                    entity_type=e['type'],
                    properties=e.get('properties', {}),
                    source_chunks=[chunk_id] if chunk_id else []
                ))

            relationships = []
            for r in result.get("relationships", []):
                source_id = f"{self._guess_type(r['source'])}_{r['source'].lower().replace(' ', '_')}"
                target_id = f"{self._guess_type(r['target'])}_{r['target'].lower().replace(' ', '_')}"
                relationships.append(Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=r['type'],
                    properties=r.get('properties', {})
                ))

            return entities, relationships

        except Exception as e:
            print(f"Error extracting entities: {e}")
            return [], []

    def _guess_type(self, name: str) -> str:
        """Guess entity type from name for ID generation"""
        name_lower = name.lower()
        if any(word in name_lower for word in ['insurance', 'policy', 'coverage']):
            return "POLICY"
        if any(word in name_lower for word in ['act', 'regulation', 'law', 'code']):
            return "REGULATION"
        if any(word in name_lower for word in ['inc', 'corp', 'company', 'department', 'agency']):
            return "ORGANIZATION"
        if any(word in name_lower for word in ['florida', 'california', 'texas', 'new york']):
            return "LOCATION"
        return "ENTITY"


class KnowledgeGraph:
    """
    In-memory knowledge graph using NetworkX.
    Supports entity storage, relationship traversal, and graph-based retrieval.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids

    def add_entity(self, entity: Entity):
        """Add an entity to the graph"""
        self.entities[entity.id] = entity
        self.entity_index[entity.entity_type].add(entity.id)

        # Add node to graph
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            properties=entity.properties
        )

    def add_relationship(self, relationship: Relationship):
        """Add a relationship to the graph"""
        # Ensure both entities exist
        if relationship.source_id not in self.graph:
            self.graph.add_node(relationship.source_id)
        if relationship.target_id not in self.graph:
            self.graph.add_node(relationship.target_id)

        # Add edge
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            relation_type=relationship.relation_type,
            properties=relationship.properties,
            confidence=relationship.confidence
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        return [self.entities[eid] for eid in self.entity_index.get(entity_type, set())
                if eid in self.entities]

    def get_neighbors(self, entity_id: str, relation_type: str = None, depth: int = 1) -> List[Tuple[str, str, Dict]]:
        """Get neighboring entities with optional relation type filter"""
        neighbors = []

        if entity_id not in self.graph:
            return neighbors

        # BFS traversal up to depth
        visited = {entity_id}
        current_level = [entity_id]

        for _ in range(depth):
            next_level = []
            for node in current_level:
                # Outgoing edges
                for _, target, data in self.graph.out_edges(node, data=True):
                    if target not in visited:
                        if relation_type is None or data.get('relation_type') == relation_type:
                            neighbors.append((target, data.get('relation_type', ''), data))
                            visited.add(target)
                            next_level.append(target)

                # Incoming edges
                for source, _, data in self.graph.in_edges(node, data=True):
                    if source not in visited:
                        if relation_type is None or data.get('relation_type') == relation_type:
                            neighbors.append((source, data.get('relation_type', ''), data))
                            visited.add(source)
                            next_level.append(source)

            current_level = next_level

        return neighbors

    def find_path(self, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between two entities"""
        try:
            return nx.shortest_path(self.graph.to_undirected(), source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def search_entities(self, query: str) -> List[Entity]:
        """Search entities by name (fuzzy match)"""
        query_lower = query.lower()
        matches = []

        for entity in self.entities.values():
            if query_lower in entity.name.lower():
                matches.append(entity)

        return matches

    def get_subgraph(self, entity_ids: List[str], depth: int = 1) -> 'KnowledgeGraph':
        """Extract a subgraph around specified entities"""
        subgraph = KnowledgeGraph()

        all_nodes = set(entity_ids)
        for eid in entity_ids:
            neighbors = self.get_neighbors(eid, depth=depth)
            all_nodes.update([n[0] for n in neighbors])

        # Add entities
        for node_id in all_nodes:
            if node_id in self.entities:
                subgraph.add_entity(self.entities[node_id])

        # Add edges between nodes in subgraph
        for u, v, data in self.graph.edges(data=True):
            if u in all_nodes and v in all_nodes:
                subgraph.graph.add_edge(u, v, **data)

        return subgraph

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        return {
            "total_entities": len(self.entities),
            "total_relationships": self.graph.number_of_edges(),
            "entity_types": {t: len(ids) for t, ids in self.entity_index.items()},
            "connected_components": nx.number_weakly_connected_components(self.graph)
        }

    def to_dict(self) -> Dict:
        """Export graph to dictionary format"""
        return {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.entity_type,
                    "properties": e.properties
                }
                for e in self.entities.values()
            ],
            "relationships": [
                {
                    "source": u,
                    "target": v,
                    "type": data.get('relation_type', ''),
                    "properties": data.get('properties', {})
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }


class GraphRAGEngine:
    """
    Graph-enhanced RAG that combines vector search with knowledge graph traversal.
    """

    def __init__(self, api_key: str, vector_store, embedding_service, model_list: List[str] = None):
        if model_list is None:
            from config import GENERATION_MODELS
            model_list = GENERATION_MODELS
        self.extractor = KnowledgeGraphExtractor(api_key, model_list)
        self.knowledge_graph = KnowledgeGraph()
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.api_key = api_key
        self.model_manager = ModelManager(api_key, model_list)

    def index_document(self, chunks: List[Dict], doc_id: str):
        """Index document chunks and extract knowledge graph"""
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            text = chunk.get('text', chunk.get('content', ''))

            # Extract entities and relationships
            entities, relationships = self.extractor.extract_entities_and_relations(text, chunk_id)

            # Add to knowledge graph
            for entity in entities:
                self.knowledge_graph.add_entity(entity)

            for rel in relationships:
                self.knowledge_graph.add_relationship(rel)

        print(f"Indexed {len(chunks)} chunks, extracted {len(self.knowledge_graph.entities)} entities")

    def graph_enhanced_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform graph-enhanced search:
        1. Vector search for initial results
        2. Extract entities from query
        3. Find related entities in graph
        4. Expand context with graph neighbors
        """

        # Step 1: Standard vector search
        query_embedding = self.embedding_service.get_embedding(query)
        vector_results = self.vector_store.search_by_text(
            query_text=query,
            query_embedding=query_embedding,
            n_results=top_k
        )

        # Step 2: Extract entities from query
        query_entities, _ = self.extractor.extract_entities_and_relations(query)

        # Step 3: Find related entities in graph
        related_entities = []
        for qe in query_entities:
            # Search for matching entities
            matches = self.knowledge_graph.search_entities(qe.name)
            for match in matches:
                # Get neighbors
                neighbors = self.knowledge_graph.get_neighbors(match.id, depth=2)
                related_entities.extend([n[0] for n in neighbors])

        # Step 4: Enhance results with graph context
        enhanced_results = []
        for result in vector_results:
            enhanced = dict(result)

            # Find entities mentioned in this chunk
            chunk_entities = []
            content = result.get('content', '')
            for entity in self.knowledge_graph.entities.values():
                if entity.name.lower() in content.lower():
                    chunk_entities.append(entity)

            # Get related context from graph
            graph_context = []
            for entity in chunk_entities[:3]:  # Limit to top 3 entities
                neighbors = self.knowledge_graph.get_neighbors(entity.id, depth=1)
                for neighbor_id, relation, data in neighbors[:5]:
                    neighbor = self.knowledge_graph.get_entity(neighbor_id)
                    if neighbor:
                        graph_context.append({
                            "entity": entity.name,
                            "relation": relation,
                            "related_entity": neighbor.name,
                            "related_type": neighbor.entity_type
                        })

            enhanced['graph_context'] = graph_context
            enhanced['mentioned_entities'] = [e.name for e in chunk_entities]
            enhanced_results.append(enhanced)

        return enhanced_results

    def generate_graph_aware_answer(self, query: str, results: List[Dict], case_context: Dict = None) -> str:
        """Generate answer using both retrieved chunks and graph context"""

        # Build context from results
        context_parts = []
        graph_insights = []

        for i, result in enumerate(results[:5]):
            content = result.get('content', '')
            context_parts.append(f"SOURCE {i+1}:\n{content}")

            # Add graph context
            if result.get('graph_context'):
                for gc in result['graph_context'][:3]:
                    insight = f"- {gc['entity']} {gc['relation']} {gc['related_entity']} ({gc['related_type']})"
                    if insight not in graph_insights:
                        graph_insights.append(insight)

        graph_section = "\n".join(graph_insights) if graph_insights else "No specific graph relationships found."

        prompt = f"""You are NEXUS AI with Graph-Enhanced RAG capabilities.

SOURCES:
{chr(10).join(context_parts)}

KNOWLEDGE GRAPH INSIGHTS:
{graph_section}

CASE CONTEXT: {json.dumps(case_context) if case_context else 'General query'}

QUESTION: {query}

Provide a comprehensive answer that:
1. Uses information from the sources
2. Leverages knowledge graph relationships to connect concepts
3. Cites sources appropriately
4. Highlights any important entity relationships discovered"""

        try:
            return self.model_manager.generate_with_fallback(prompt)
        except Exception as e:
            return f"Error generating answer: {e}"

    def get_graph_visualization_data(self) -> Dict:
        """Get data for graph visualization"""
        nodes = []
        edges = []

        for entity in self.knowledge_graph.entities.values():
            nodes.append({
                "id": entity.id,
                "label": entity.name,
                "type": entity.entity_type,
                "size": 10 + len(entity.source_chunks) * 2
            })

        for u, v, data in self.knowledge_graph.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "label": data.get('relation_type', ''),
                "weight": data.get('confidence', 1.0)
            })

        return {"nodes": nodes, "edges": edges}
