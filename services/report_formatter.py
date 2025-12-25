"""
Professional Report Formatter - Consulting-style report generation
"""

from typing import Dict, List, Optional
from datetime import datetime
import re


class ConsultingReportFormatter:
    """
    Formats RAG responses into professional consulting reports.
    Includes executive summary, findings, analysis, and recommendations.
    """

    def __init__(self):
        self.report_counter = 1

    def format_response(
        self,
        query: str,
        answer: str,
        sources: List[Dict] = None,
        rag_mode: str = "Standard",
        case_context: Dict = None,
        metadata: Dict = None
    ) -> str:
        """
        Format a RAG response as a professional consulting report.

        Args:
            query: User's original question
            answer: RAG-generated answer
            sources: List of source documents/chunks used
            rag_mode: RAG mode used (Standard/Graph RAG/Agentic RAG)
            case_context: Case type, state, etc.
            metadata: Additional metadata (complexity, confidence, etc.)

        Returns:
            Formatted markdown report
        """
        report = []

        # Header with branding
        report.append(self._generate_header(query, case_context))

        # Executive Summary
        report.append(self._generate_executive_summary(query, answer, metadata))

        # Query Details
        report.append(self._generate_query_section(query, rag_mode, case_context, metadata))

        # Main Analysis/Findings
        report.append(self._generate_analysis_section(answer, rag_mode))

        # Key Insights (extracted from answer)
        report.append(self._generate_key_insights(answer))

        # Source Documentation
        if sources:
            report.append(self._generate_sources_section(sources))

        # Methodology
        report.append(self._generate_methodology_section(rag_mode, metadata))

        # Footer
        report.append(self._generate_footer())

        return "\n\n".join(report)

    def _generate_header(self, query: str, case_context: Dict = None) -> str:
        """Generate professional report header"""
        date_str = datetime.now().strftime("%B %d, %Y")
        time_str = datetime.now().strftime("%I:%M %p")

        case_info = ""
        if case_context:
            case_type = case_context.get('case_type', 'General')
            state = case_context.get('state', 'N/A')
            case_info = f"\n**Case Type:** {case_type} | **Jurisdiction:** {state}"

        return f"""---
# 📊 NEXUS AI INTELLIGENCE REPORT

**Report ID:** NEXUS-{self.report_counter:05d}
**Generated:** {date_str} at {time_str}
**System:** Intelligent Knowledge Retrieval Engine{case_info}

---"""

    def _generate_executive_summary(self, query: str, answer: str, metadata: Dict = None) -> str:
        """Generate executive summary section"""
        # Extract first 2-3 sentences as summary
        sentences = re.split(r'[.!?]+', answer.strip())
        summary_sentences = [s.strip() for s in sentences[:3] if s.strip()]
        summary = '. '.join(summary_sentences) + '.'

        confidence = ""
        if metadata and 'confidence' in metadata:
            conf_pct = metadata['confidence'] * 100
            confidence = f"\n\n**Confidence Level:** {conf_pct:.0f}% {'🟢' if conf_pct > 70 else '🟡' if conf_pct > 50 else '🔴'}"

        return f"""## 📋 EXECUTIVE SUMMARY

{summary}{confidence}

---"""

    def _generate_query_section(self, query: str, rag_mode: str, case_context: Dict = None, metadata: Dict = None) -> str:
        """Generate query details section"""
        complexity = metadata.get('complexity', 'medium').title() if metadata else 'Medium'

        complexity_icon = {
            'Simple': '🟢',
            'Medium': '🟡',
            'Complex': '🔴'
        }.get(complexity, '🟡')

        return f"""## ❓ QUERY ANALYSIS

**Original Query:**
> {query}

**Analysis Metrics:**
- **Processing Mode:** {rag_mode}
- **Query Complexity:** {complexity_icon} {complexity}
- **Analysis Type:** {'Relationship-based' if rag_mode == 'Graph RAG' else 'Multi-tool Reasoning' if rag_mode == 'Agentic RAG' else 'Direct Retrieval'}

---"""

    def _generate_analysis_section(self, answer: str, rag_mode: str) -> str:
        """Generate main analysis/findings section"""
        # Split answer into paragraphs for better formatting
        paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]

        formatted_answer = []
        for i, para in enumerate(paragraphs):
            # Check if paragraph is a list
            if para.strip().startswith('-') or para.strip().startswith('•'):
                formatted_answer.append(para)
            elif para.strip().startswith(('1.', '2.', '3.')):
                formatted_answer.append(para)
            else:
                # Add paragraph numbering for readability
                if len(paragraphs) > 1:
                    formatted_answer.append(f"**{i+1}.** {para}")
                else:
                    formatted_answer.append(para)

        return f"""## 🔍 DETAILED ANALYSIS & FINDINGS

{chr(10).join(formatted_answer)}

---"""

    def _generate_key_insights(self, answer: str) -> str:
        """Extract and format key insights from answer"""
        # Simple heuristic: extract sentences with keywords
        insight_keywords = [
            'important', 'key', 'critical', 'significant', 'note that',
            'however', 'therefore', 'consequently', 'requires', 'must',
            'should', 'covers', 'excludes', 'limit', 'maximum', 'minimum'
        ]

        sentences = re.split(r'[.!?]+', answer)
        insights = []

        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in insight_keywords):
                if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                    insights.append(f"• {sentence}.")

        if not insights:
            return ""

        # Limit to top 5 insights
        insights = insights[:5]

        return f"""## 💡 KEY INSIGHTS

{chr(10).join(insights)}

---"""

    def _generate_sources_section(self, sources: List[Dict]) -> str:
        """Generate sources and citations section"""
        if not sources:
            return ""

        source_entries = []
        for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
            title = source.get('metadata', {}).get('title', 'Unknown Document')
            page = source.get('metadata', {}).get('page', 'N/A')
            source_type = source.get('metadata', {}).get('source_type', 'document')
            score = source.get('score', 0)

            # Clean up content preview
            content = source.get('content', '')[:200].strip()
            if len(source.get('content', '')) > 200:
                content += "..."

            source_entries.append(f"""**[{i}]** {title} (Page {page})
- **Source Type:** {source_type.title()}
- **Relevance Score:** {score:.3f}
- **Extract:** "{content}" """)

        return f"""## 📚 SOURCE DOCUMENTATION

The following sources were consulted in generating this analysis:

{chr(10).join(source_entries)}

---"""

    def _generate_methodology_section(self, rag_mode: str, metadata: Dict = None) -> str:
        """Generate methodology section explaining the approach"""
        methodology_desc = {
            "Standard": """**Standard RAG** - Direct semantic search against the knowledge base using vector embeddings.
Best for factual queries requiring precise information retrieval from indexed documents.""",

            "Graph RAG": """**Graph RAG** - Knowledge graph-enhanced retrieval leveraging entity relationships and network traversal.
Analyzes connections between policies, regulations, organizations, and other entities to provide context-aware insights.""",

            "Agentic RAG": """**Agentic RAG** - Multi-tool reasoning system using LangChain agents with ReAct framework.
Employs autonomous tool selection (search, calculation, graph query) to handle complex, multi-step queries."""
        }

        desc = methodology_desc.get(rag_mode, "Standard retrieval methodology.")

        tools_used = ""
        if metadata and 'tools_used' in metadata:
            tools_list = ', '.join(metadata['tools_used'])
            tools_used = f"\n\n**Tools Utilized:** {tools_list}"

        model_info = ""
        if metadata and 'model_used' in metadata:
            model_info = f"\n**LLM Model:** {metadata['model_used']}"

        return f"""## 🔬 METHODOLOGY

{desc}{tools_used}{model_info}

---"""

    def _generate_footer(self) -> str:
        """Generate professional footer"""
        return f"""## ⚠️ DISCLAIMER

This report was generated by NEXUS AI, an intelligent knowledge retrieval system. The analysis is based on indexed documents and should be reviewed by qualified professionals for critical decisions.

**Prepared by:** NEXUS AI Intelligence Engine
**Report Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

*For questions or clarifications regarding this report, please consult the source documents or contact your legal/compliance team.*"""

    def increment_counter(self):
        """Increment report counter for unique IDs"""
        self.report_counter += 1


class ReportExporter:
    """
    Export reports to various formats (Markdown, HTML, plain text).
    """

    @staticmethod
    def to_markdown(report: str) -> str:
        """Return report as markdown (already in this format)"""
        return report

    @staticmethod
    def to_plain_text(report: str) -> str:
        """Convert markdown report to plain text"""
        # Remove markdown formatting
        text = report
        text = re.sub(r'#{1,6}\s', '', text)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'>\s', '', text)  # Remove blockquotes
        text = re.sub(r'---+', '', text)  # Remove horizontal rules
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links
        return text.strip()

    @staticmethod
    def to_html(report: str) -> str:
        """Convert markdown report to HTML"""
        try:
            import markdown
            html = markdown.markdown(report, extensions=['extra', 'codehilite'])

            # Add professional styling
            styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NEXUS AI Intelligence Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        blockquote {{
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 10px 20px;
            margin: 20px 0;
        }}
        hr {{
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 30px 0;
        }}
        strong {{
            color: #2980b9;
        }}
        ul, ol {{
            padding-left: 30px;
        }}
        .report-container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="report-container">
        {html}
    </div>
</body>
</html>
"""
            return styled_html
        except ImportError:
            # Fallback if markdown library not available
            return f"<html><body><pre>{report}</pre></body></html>"
