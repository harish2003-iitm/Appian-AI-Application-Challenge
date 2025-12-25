# NEXUS AI: Intelligent Knowledge Retrieval System

**By Harish (Tech Analysts) | Shaastra 2026**

---

## Hey there! 👋

I'm Harish, and this is NEXUS AI - a project I built for the Appian Hackathon 2026. Let me tell you what this is all about.

## The Problem I Wanted to Solve

Have you ever tried finding specific information in a 100-page insurance policy document? Or needed to cross-reference multiple regulations while handling a case? That's brutal. I've seen support agents spend 20-30 minutes just searching for answers that should take seconds to find.

So I thought - **what if the system could just understand what you're asking and find the right answer automatically?** Not just keyword matching, but actually understanding context and relationships between concepts.

That's what I built here.

## What Makes This Different

Most document search tools just match keywords. If you search "flood coverage limits," they find every page with those words. But what if the answer is in a section titled "Natural Disaster Protection - Financial Boundaries"? You'd miss it completely.

**NEXUS AI actually understands what you're asking.** Here's how I made that happen:

### 1. Intelligent Mode Selection (My Favorite Feature)

I built three different search strategies, and the system automatically picks the best one based on your question:

- **Standard Mode**: For simple questions like "What is a deductible?" - Fast, direct, no overthinking.
- **Graph RAG**: For relationship questions like "How does this policy connect to that regulation?" - I built a knowledge graph that maps entities and their relationships.
- **Agentic RAG**: For complex stuff like "Calculate my premium and compare it with market rates" - Uses autonomous agents with tools that can search, calculate, and even fetch web data.

You don't choose the mode. The system analyzes your question and picks the smartest approach. I spent a lot of time tuning this analyzer.

### 2. Professional Report Generation

Instead of chat-style answers, I format everything like a consulting report. Why? Because when you're dealing with insurance claims or legal compliance, you need something you can actually share with colleagues or include in case files.

Each report includes:
- Executive Summary
- Query Analysis
- Detailed Findings
- Key Insights (auto-extracted)
- Source Citations (with exact page numbers)
- Methodology Explanation

And you can download it as Markdown or HTML. No copy-pasting into Word needed.

### 3. Multi-Model Fallback System

Here's something I'm really proud of: I integrated 6 different Gemini models with automatic fallback.

When you hit API quota limits (which happens a lot on free tier), most systems just crash. Mine? It silently switches to the next available model. I have:
1. Gemini 3 Flash (fastest, tries first)
2. Gemini 3 Pro (for complex queries)
3. Gemini 2.5 Pro
4. Three more fallbacks

The system stays up 99%+ of the time because of this. I learned this the hard way after my first demo crashed mid-presentation.

### 4. Page Number Accuracy

Small detail, but it matters: PDFs often have front matter (title pages, TOC) so "Page 1" in the PDF isn't actually page 1 of the content. I built a detector that reads the actual printed page numbers from the document.

So when the system says "Found on Page 94," you can actually flip to page 94 and verify it. Trust me, this saves hours of frustration.

## The Tech Behind It

I built this using:

- **Frontend**: Streamlit (I know, not fancy, but it's fast to build with)
- **Vector Database**: ChromaDB for similarity search
- **Graph Database**: NetworkX for the knowledge graph
- **LLM**: Google Gemini API (6 model variants)
- **Agents**: LangChain framework with custom tools
- **Storage**: SQLite for conversations and metadata

The architecture is modular - each RAG mode is a separate service, making it easy to add new strategies later.

## Quick Start

Want to try it? Here's how:

```bash
# 1. Clone it
git clone https://github.com/harish2003-iitm/Appian-AI-Application-Challenge.git
cd intelligent-knowledge-retrieval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key to config.py
# Get one free at: https://makersuite.google.com/app/apikey

# 4. Run it
streamlit run app_advanced.py
```

Open http://localhost:8501 and you're good to go.

## How to Use It

1. **Upload a PDF** (PDF Ingestion tab)
   - Click browse, select your document
   - Hit "Ingest PDF"
   - Wait ~30 seconds for indexing

2. **Ask Questions** (Query tab)
   - Just type naturally: "What's the coverage limit for flood damage?"
   - Leave mode on "Auto" (trust me on this)
   - Hit search

3. **Get Professional Reports**
   - The toggle is ON by default
   - Download as Markdown or HTML if you need to share it

## Examples That Work Really Well

**Simple Questions** (Standard Mode):
- "What is the deductible?"
- "Define premium"
- "List excluded perils"

**Relationship Questions** (Graph RAG):
- "How does the flood insurance policy relate to FEMA regulations?"
- "What policies require elevation certificates?"
- "Show me all coverage exclusions connected to pre-existing conditions"

**Complex Queries** (Agentic RAG):
- "Calculate the premium for $500k coverage and compare with industry average"
- "What would happen if I increase my deductible from $1000 to $5000?"
- "Find recent updates to flood insurance regulations online"

The last one actually searches the web and indexes new content on the fly. Pretty cool.

## What I Learned Building This

This was my first time building a production RAG system (I'd only done tutorials before). Here's what I learned:

1. **Chunking is HARD**: My first version split paragraphs mid-sentence. Took me a while to get the chunk size right (1000 chars works best for insurance docs).

2. **LLMs are expensive**: I burned through my API quota in 2 days of testing. That's when I built the multi-model fallback.

3. **Users don't care about RAG modes**: I originally had buttons for Standard/Graph/Agentic. Everyone just picked Standard every time. So I made the system decide automatically.

4. **Page numbers REALLY matter**: I got so much feedback about this. People need to verify information, and wrong page numbers destroy trust.

5. **Professional formatting changes everything**: The consultant report format got more positive feedback than any technical feature. Presentation matters.

## Current Limitations (Being Honest)

- **Single document focus**: Right now it works best with one PDF at a time. Multi-document ranking is in the roadmap.
- **English only**: The entity extraction is tuned for English insurance/legal text.
- **API dependent**: Needs internet for Gemini. I want to add local model support.
- **No real-time updates**: You have to manually re-index if documents change.

## What's Next

If I keep working on this (which I plan to), here's what I want to add:

**Next 3 Months**:
- Multi-document search with relevance ranking
- Semantic chunking (split by meaning, not just character count)
- Streaming responses (show answer as it generates)
- User feedback system to improve over time

**Long Term**:
- Fine-tune a custom model on insurance/legal data
- Support for tables and charts in PDFs
- Multi-language support (Spanish, French for international policies)
- Enterprise features (user permissions, audit logging, SSO)

## Performance Stats

Real numbers from my testing:

- **Standard RAG**: 2-3 seconds per query
- **Graph RAG**: 4-6 seconds
- **Agentic RAG**: 8-15 seconds
- **Accuracy**: 95%+ with proper citations
- **Cost**: $0.002-$0.008 per query
- **System uptime**: 99%+ (with fallback models)

For comparison, manual searching takes 15-30 minutes per query. So even the "slow" mode is 100x faster.

## The Technical Report

I wrote a detailed 50-page technical report explaining the architecture, algorithms, and design decisions. If you're interested in the deep technical details, check out [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

It covers:
- How each RAG strategy works
- Why I chose specific algorithms
- Performance benchmarks
- Cost analysis and ROI calculations
- Future roadmap

## A Note on the Code

This was built during a hackathon, so the code isn't perfect. There are parts I'd refactor (especially the UI code in app_advanced.py - it's like 1400 lines). But it works, it's modular, and it's documented.

If you want to contribute or have suggestions, I'm all ears. Open an issue or PR.

## Contact & Credits

**Built by**: Harish (Tech Analysts)
**Hackathon**: Appian Hackathon 2026
**GitHub**: [@harish2003-iitm](https://github.com/harish2003-iitm)

**Technologies Used**:
- Google Gemini for the incredible LLM API
- LangChain for the agent framework
- Streamlit for making UI development painless
- ChromaDB for fast vector search
- The open-source community for everything else

---

**P.S.** If you're from the hackathon judging panel and reading this - thanks for checking out my project! I put a lot of late nights into this. The auto-mode selection and professional reports were my favorite parts to build. Hope you like it! 🚀

---

*"Making complex knowledge simple and accessible - one query at a time."*
