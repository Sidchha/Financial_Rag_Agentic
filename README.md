# Financial RAG Agent (AI Engineering Sprint Challenge)

## Overview

This project is a **financial question-answering system** built on **RAG (Retrieval-Augmented Generation)** principles. It allows users to query 10-K filings for **Google, Microsoft, and NVIDIA (2022-2024)** via a simple CLI interface.  
The system supports **query decomposition**, **multi-step retrieval**, **cross-company analysis**, and **JSON output synthesis**.

---

## Features

- Loads and parses local 10-K filings in PDF format.
- Chunks filings using semantic splitting (1000 characters, 100 overlap).
- Embeds text using **Google Gemini embeddings**.
- Stores embeddings in **FAISS in-memory vector store**.
- Implements a **retriever tool** to fetch top-k relevant excerpts.
- Builds an **agent** capable of:
  - Decomposing complex or comparative queries.
  - Multi-step retrieval across filings.
  - Synthesizing results into structured JSON with sources.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo_url>
cd <repo_folder>
```

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Set Environment Variables

Create a .env file in the project root with your Google Gemini API key:

GOOGLE_API_KEY=<your_api_key>

### 4. Run the following command to start the application

python main.py
