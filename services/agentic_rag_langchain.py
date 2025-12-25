"""
Agentic RAG with LangChain
- Multi-tool agents for complex query handling
- Web scraping and ingestion
- Self-correcting retrieval with reflection
- Tool use for specialized tasks
"""

import os
import re
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from .model_manager import ChatModelManager

# LangChain imports
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool, StructuredTool
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.callbacks.base import BaseCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not fully installed. Run: pip install langchain langchain-google-genai")


@dataclass
class WebPage:
    """Represents a scraped web page"""
    url: str
    title: str
    content: str
    links: List[str]
    metadata: Dict


class WebScraper:
    """
    Web scraping utility for ingesting web content.
    """

    def __init__(self, max_content_length: int = 50000):
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_url(self, url: str) -> Optional[WebPage]:
        """Scrape a single URL and extract content"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get title
            title = soup.title.string if soup.title else urlparse(url).netloc

            # Get main content
            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': re.compile(r'content|main|article', re.I)})

            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            content = '\n'.join(lines)[:self.max_content_length]

            # Extract links
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('http'):
                    links.append(href)
                elif href.startswith('/'):
                    links.append(urljoin(url, href))

            return WebPage(
                url=url,
                title=title,
                content=content,
                links=links[:20],  # Limit links
                metadata={
                    'domain': urlparse(url).netloc,
                    'content_length': len(content)
                }
            )

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            # Return None so index_url can handle the error
            return None

    def scrape_multiple(self, urls: List[str]) -> List[WebPage]:
        """Scrape multiple URLs"""
        pages = []
        for url in urls:
            page = self.scrape_url(url)
            if page:
                pages.append(page)
        return pages


class WebContentIndexer:
    """
    Indexes web content into the vector store.
    """

    def __init__(self, vector_store, embedding_service, chunk_size: int = 500, chunk_overlap: int = 50):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.scraper = WebScraper()

    def index_url(self, url: str) -> Dict:
        """Scrape and index a URL"""
        page = self.scraper.scrape_url(url)
        if not page:
            return {"success": False, "url": url, "error": "Failed to scrape URL"}

        # Split content into chunks
        chunks = self._split_content(page.content)

        if not chunks:
            return {"success": False, "url": url, "error": "No content to index"}

        # Generate document ID
        doc_id = f"WEB_{urlparse(url).netloc.replace('.', '_')}_{hash(url) % 10000}"

        # Prepare for indexing
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        documents = chunks
        metadatas = [{
            "source_type": "web",
            "url": url,
            "title": page.title,
            "domain": page.metadata['domain'],
            "chunk_index": i,
            "document_id": doc_id
        } for i in range(len(chunks))]

        # Generate embeddings
        embeddings = self.embedding_service.get_embeddings_batch(documents)

        # Add to vector store
        self.vector_store.add_embeddings(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return {
            "success": True,
            "url": url,
            "title": page.title,
            "chunks_indexed": len(chunks),
            "document_id": doc_id
        }

    def index_multiple_urls(self, urls: List[str]) -> List[Dict]:
        """Index multiple URLs"""
        results = []
        for url in urls:
            result = self.index_url(url)
            results.append(result)
        return results

    def _split_content(self, content: str) -> List[str]:
        """Split content into chunks"""
        if not content:
            return []

        # Simple splitting by paragraphs then by size
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if len(c) > 50]  # Filter out tiny chunks


class AgenticRAGTools:
    """
    Tools for the Agentic RAG system.
    """

    def __init__(self, vector_store, embedding_service, retrieval_engine, graph_rag=None, api_key: str = None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.retrieval_engine = retrieval_engine
        self.graph_rag = graph_rag
        self.web_indexer = WebContentIndexer(vector_store, embedding_service)
        self.api_key = api_key

    def search_knowledge_base(self, query: str) -> str:
        """Search the internal knowledge base"""
        try:
            results = self.retrieval_engine.retrieve(query=query, top_k=5)
            if not results:
                return "No relevant documents found in knowledge base."

            output = []
            for i, r in enumerate(results[:3]):
                content = r.get('content', '')[:500]
                source = r.get('metadata', {}).get('title', 'Unknown')
                page = r.get('metadata', {}).get('page', 'N/A')
                output.append(f"[Source {i+1}: {source}, Page {page}]\n{content}")

            return "\n\n---\n\n".join(output)
        except Exception as e:
            return f"Error searching knowledge base: {e}"

    def search_web(self, query: str) -> str:
        """Search the web for information (simulated - uses scraped content)"""
        # In a real implementation, you'd use a search API
        return f"Web search for '{query}' - Use ingest_url tool to add specific web pages to the knowledge base."

    def ingest_url(self, url: str) -> str:
        """Scrape and index a web URL"""
        try:
            result = self.web_indexer.index_url(url)
            if result['success']:
                return f"Successfully indexed {result['url']}: {result['chunks_indexed']} chunks added. Document ID: {result['document_id']}"
            else:
                return f"Failed to index URL: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error indexing URL: {e}"

    def graph_query(self, entity_name: str) -> str:
        """Query the knowledge graph for entity relationships"""
        if not self.graph_rag:
            return "Knowledge graph not available."

        try:
            entities = self.graph_rag.knowledge_graph.search_entities(entity_name)
            if not entities:
                return f"No entities found matching '{entity_name}'"

            output = []
            for entity in entities[:3]:
                neighbors = self.graph_rag.knowledge_graph.get_neighbors(entity.id, depth=1)
                relations = [f"  - {rel}: {self.graph_rag.knowledge_graph.get_entity(nid).name if self.graph_rag.knowledge_graph.get_entity(nid) else nid}"
                            for nid, rel, _ in neighbors[:5]]
                output.append(f"Entity: {entity.name} ({entity.entity_type})\nRelationships:\n" + "\n".join(relations))

            return "\n\n".join(output) if output else "No relationships found."
        except Exception as e:
            return f"Error querying graph: {e}"

    def calculate(self, expression: str) -> str:
        """Perform calculations (for coverage amounts, premiums, etc.)"""
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Invalid characters in expression. Only numbers and basic operators allowed."
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    def summarize_document(self, doc_id: str) -> str:
        """Get a summary of a specific document"""
        try:
            # Search for chunks from this document
            results = self.vector_store.collection.get(
                where={"document_id": doc_id},
                limit=10,
                include=["documents", "metadatas"]
            )

            if not results['documents']:
                return f"Document {doc_id} not found."

            # Combine chunks
            content = "\n".join(results['documents'][:5])
            return f"Document {doc_id} summary (first 5 chunks):\n{content[:1500]}..."
        except Exception as e:
            return f"Error summarizing document: {e}"

    def get_tools(self) -> List[Tool]:
        """Get all available tools"""
        return [
            Tool(
                name="search_knowledge_base",
                func=self.search_knowledge_base,
                description="Search the internal knowledge base for policy documents, regulations, and procedures. Use this for questions about insurance policies, claims, coverage, etc."
            ),
            Tool(
                name="ingest_url",
                func=self.ingest_url,
                description="Scrape and index a web URL into the knowledge base. Use this when you need to add new web content. Input should be a valid URL."
            ),
            Tool(
                name="graph_query",
                func=self.graph_query,
                description="Query the knowledge graph to find relationships between entities like policies, regulations, organizations. Input should be an entity name."
            ),
            Tool(
                name="calculate",
                func=self.calculate,
                description="Perform mathematical calculations. Input should be a mathematical expression like '100000 * 0.02' for premium calculations."
            ),
            Tool(
                name="summarize_document",
                func=self.summarize_document,
                description="Get a summary of a specific document by its ID. Input should be a document ID."
            )
        ]


class AgenticRAGEngine:
    """
    Main Agentic RAG engine using LangChain.
    Orchestrates multiple tools and agents for complex query handling.
    """

    def __init__(self, api_key: str, vector_store, embedding_service, retrieval_engine, graph_rag=None, model_list: List[str] = None):
        self.api_key = api_key
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.retrieval_engine = retrieval_engine
        self.graph_rag = graph_rag

        # Initialize model manager
        if model_list is None:
            from config import GENERATION_MODELS
            model_list = GENERATION_MODELS
        self.chat_model_manager = ChatModelManager(api_key, model_list)

        # Initialize tools
        self.tools_provider = AgenticRAGTools(
            vector_store=vector_store,
            embedding_service=embedding_service,
            retrieval_engine=retrieval_engine,
            graph_rag=graph_rag,
            api_key=api_key
        )

        # Initialize LangChain components if available
        if LANGCHAIN_AVAILABLE:
            self.llm = self.chat_model_manager.get_chat_model(temperature=0.3)
            self.tools = self.tools_provider.get_tools()
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            self.agent = self._create_agent()
        else:
            self.agent = None

    def _create_agent(self):
        """Create the ReAct agent"""
        if not LANGCHAIN_AVAILABLE:
            return None

        # ReAct prompt template - optimized for efficiency
        template = """You are NEXUS AI, an intelligent assistant for insurance and policy knowledge retrieval.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL EFFICIENCY GUIDELINES:
1. ALWAYS search the knowledge base FIRST for policy/insurance questions - don't waste iterations
2. Use graph_query ONLY when you need entity relationships (not for every query)
3. Use ingest_url ONLY when explicitly asked to add web content
4. AVOID unnecessary tool calls - if you have enough information, generate the final answer
5. Keep iterations under 10 - be efficient with tool use
6. Provide citations when answering from sources
7. If knowledge base search returns "No relevant documents", try rephrasing or conclude clearly

OPTIMIZATION TIPS:
- Don't repeatedly search with the same query
- Combine information from first search to answer immediately if possible
- Only use additional tools if absolutely necessary

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)

        try:
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )

            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,  # Increased from 5 to allow more complex reasoning
                max_execution_time=120,  # 2 minutes timeout
                early_stopping_method="generate"  # Generate final answer if iterations exceeded
            )
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None

    def query(self, question: str, case_context: Dict = None) -> Dict:
        """
        Process a query using the agentic system.
        Returns structured response with answer, sources, and agent trace.
        """
        if not LANGCHAIN_AVAILABLE or not self.agent:
            # Fallback to simple retrieval
            return self._fallback_query(question, case_context)

        # Add context to question if provided
        if case_context:
            context_str = f"[Context: {case_context.get('case_type', '')}, {case_context.get('state', '')}] "
            full_question = context_str + question
        else:
            full_question = question

        # Try with automatic model fallback
        while True:
            try:
                # Run agent
                result = self.agent.invoke({"input": full_question})

                return {
                    "success": True,
                    "answer": result.get("output", ""),
                    "agent_trace": result.get("intermediate_steps", []),
                    "tools_used": self._extract_tools_used(result),
                    "mode": "agentic",
                    "model_used": self.chat_model_manager.get_current_model_name()
                }

            except Exception as e:
                error_str = str(e)
                print(f"Agent error with {self.chat_model_manager.get_current_model_name()}: {error_str[:200]}")

                # Check if it's a quota error
                if ("429" in error_str or "quota" in error_str.lower() or "resource_exhausted" in error_str.lower()):
                    # Try to switch to next model
                    if self.chat_model_manager.try_next_model():
                        print(f"Switching to fallback model: {self.chat_model_manager.get_current_model_name()}")
                        # Recreate LLM and agent with new model
                        self.llm = self.chat_model_manager.get_chat_model(temperature=0.3)
                        self.agent = self._create_agent()
                        continue
                    else:
                        # All models exhausted
                        print("All fallback models exhausted, using simple retrieval")
                        return self._fallback_query(question, case_context)
                else:
                    # Non-quota error - use fallback
                    print(f"Non-quota error: {error_str}")
                    return self._fallback_query(question, case_context)

    def _fallback_query(self, question: str, case_context: Dict = None) -> Dict:
        """Fallback to simple retrieval when agent fails"""
        try:
            # Simple search
            search_result = self.tools_provider.search_knowledge_base(question)

            return {
                "success": True,
                "answer": f"Based on the knowledge base:\n\n{search_result}",
                "agent_trace": [],
                "tools_used": ["search_knowledge_base (fallback)"],
                "mode": "fallback"
            }
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing query: {e}",
                "agent_trace": [],
                "tools_used": [],
                "mode": "error"
            }

    def _extract_tools_used(self, result: Dict) -> List[str]:
        """Extract list of tools used from agent result"""
        tools = []
        for step in result.get("intermediate_steps", []):
            if hasattr(step[0], 'tool'):
                tools.append(step[0].tool)
        return tools

    def ingest_web_content(self, urls: List[str]) -> List[Dict]:
        """Ingest multiple web URLs"""
        return self.tools_provider.web_indexer.index_multiple_urls(urls)

    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history from memory"""
        if not LANGCHAIN_AVAILABLE or not self.memory:
            return []

        messages = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages

    def clear_memory(self):
        """Clear conversation memory"""
        if LANGCHAIN_AVAILABLE and self.memory:
            self.memory.clear()
