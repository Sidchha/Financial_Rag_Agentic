Financial RAG Agent – Design Document

1. Chunking Strategy
   Used RecursiveCharacterTextSplitter to split 10-K filings into semantic chunks of 1000 characters with 100-character overlap.

2. Embedding Model Choice

   Model: GoogleGenerativeAIEmbeddings (models/embedding-001)
   Reason: Provides high-quality embeddings optimized for text semantic similarity, which is essential for accurately retrieving relevant filings content across multiple companies and years. Integrates seamlessly with the FAISS vector store used for retrieval.

3. Agent / Query Decomposition

Architecture: Built using a state machine (StateGraph) with two main nodes:

LLM Node: Handles query understanding and decomposition.

Retriever Node: Executes retrieval tool calls and returns results.

Decomposition Logic:

Comparative or multi-company queries are split into sub-queries for each company/year.

Single-company queries are sent directly to the retriever.

Synthesis: The LLM synthesizes multiple retrieved results into a coherent JSON output with fields: query, answer, reasoning, sub_queries, and sources.

4. Challenges & Design Decisions

Vector store choice: Opted for FAISS to support fast in-memory similarity search without external dependencies.

JSON consistency: System prompt strictly enforces JSON output to simplify downstream processing and evaluation.
