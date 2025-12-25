"""ChromaDB vector store for semantic search"""

import chromadb
from chromadb.config import Settings
import os
import shutil
from typing import List, Dict, Optional, Any

class VectorStore:
    def __init__(self, persist_dir: str = "data/chroma_db", embedding_dim: int = 384):
        self.persist_dir = persist_dir
        self.embedding_dim = embedding_dim
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create the main document collection"""
        try:
            collection = self.client.get_or_create_collection(
                name="knowledge_documents",
                metadata={
                    "description": "Policy and procedure documents for case management",
                    "embedding_dim": str(self.embedding_dim)
                }
            )
            return collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            # Try to recreate if there's a dimension mismatch
            try:
                self.client.delete_collection("knowledge_documents")
                return self.client.create_collection(
                    name="knowledge_documents",
                    metadata={
                        "description": "Policy and procedure documents for case management",
                        "embedding_dim": str(self.embedding_dim)
                    }
                )
            except Exception as e2:
                print(f"Error recreating collection: {e2}")
                raise

    def reset_collection(self):
        """Delete and recreate the collection (useful when embedding dimensions change)"""
        try:
            self.client.delete_collection("knowledge_documents")
            print("Deleted old collection")
        except Exception as e:
            print(f"No existing collection to delete: {e}")

        self.collection = self.client.create_collection(
            name="knowledge_documents",
            metadata={
                "description": "Policy and procedure documents for case management",
                "embedding_dim": str(self.embedding_dim)
            }
        )
        print(f"Created new collection with embedding_dim={self.embedding_dim}")
        return self.collection

    def add_embeddings(self,
                       ids: List[str],
                       embeddings: List[List[float]],
                       documents: List[str],
                       metadatas: List[Dict] = None):
        """Add document embeddings to the vector store"""
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{}] * len(ids)
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Handle dimension mismatch
            if "dimension" in error_msg or "dimensionality" in error_msg:
                print(f"Embedding dimension mismatch detected. Resetting collection...")
                self.reset_collection()
                # Retry the add
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas or [{}] * len(ids)
                )
            else:
                raise

    def search(self,
               query_embedding: List[float],
               n_results: int = 5,
               where: Dict = None,
               where_document: Dict = None) -> Dict:
        """Search for similar documents"""
        try:
            # Ensure we don't request more results than available
            doc_count = self.collection.count()
            actual_n = min(n_results, doc_count) if doc_count > 0 else n_results

            if actual_n == 0:
                return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_n,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )
            return results
        except Exception as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg:
                print(f"Dimension mismatch in search. Collection may need reindexing.")
                return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            raise

    def search_by_text(self,
                       query_text: str,
                       query_embedding: List[float],
                       n_results: int = 5,
                       filter_metadata: Dict = None) -> List[Dict]:
        """Search and return formatted results"""
        results = self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=filter_metadata
        )

        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'content': results['documents'][0][i] if results['documents'] else "",
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                    'relevance_score': 1 - (results['distances'][0][i] if results['distances'] else 0)
                })

        return formatted_results

    def get_document_count(self) -> int:
        """Get total number of documents in the collection"""
        return self.collection.count()

    def delete_all(self):
        """Delete all documents from the collection"""
        # Get all IDs and delete them
        all_data = self.collection.get()
        if all_data['ids']:
            self.collection.delete(ids=all_data['ids'])

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get a specific document by ID"""
        result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if result['ids']:
            return {
                'id': result['ids'][0],
                'content': result['documents'][0] if result['documents'] else "",
                'metadata': result['metadatas'][0] if result['metadatas'] else {}
            }
        return None
