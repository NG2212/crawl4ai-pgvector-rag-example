--- a/README.md
+++ b/README.md
@@
-# RAG Question-Answering Application
+# RAG Question-Answering Application (2025 refresh)
 
 A Retrieval-Augmented Generation (RAG) application that allows you to ask questions about web content. The app crawls web pages, stores their content with embeddings in a vector database, and uses OpenAI's GPT models to generate accurate answers based on the retrieved context.
 
 ## Features
-
-  * Web Crawling: Crawl and index any web page
-  * PDF Processing: Upload and index PDF documents
-  * Vector Search: Use pgvector for semantic similarity search
-  * Context-Aware Answers: Generate answers using relevant context
-  * Streamlit Interface: Easy-to-use web interface
-  * Chunk Management: Smart text chunking for optimal embedding
+  * Web Crawling: Crawl and index any web page (crawl4ai)
+  * PDF Processing: Upload and index PDF documents (pypdf)
+  * Vector Search: PostgreSQL + pgvector with **cosine distance** and **IVFFlat** index
+  * Context Answers: Uses **GPT-5-mini** by default (toggle via env)
+  * Embeddings: **text-embedding-3-small** (1536-dim) by default (toggle via env)
+  * Optional LLM Re-rank: Reorder top-10 retrievals for better accuracy
+  * UI niceties: Show cosine distance, list sources under the answer
+  * Optional (commented): paragraph-aware chunking with overlap
 
 ## Prerequisites
-  * Docker and Docker Compose
-  * OpenAI API key
-  * VS Code with Dev Containers extension (recommended)
+  * PostgreSQL with `pgvector` extension enabled
+  * OpenAI API key in `.env` (see example)
+  * (Optional) VS Code Dev Containers / Docker Compose
 
 ## Quick Start
@@
-2. Create a `.env` file in the root directory:
+2. Create a `.env` file in the root directory:
 
-      OPENAI_API_KEY=your_api_key_here
-  DB_HOST=db
-  DB_NAME=rag_db
-  DB_USER=rag_user
-  DB_PASSWORD=your_password_here
+      OPENAI_API_KEY=your_api_key_here
+      OPENAI_CHAT_MODEL=gpt-5-mini
+      OPENAI_EMBED_MODEL=text-embedding-3-small
+      EMBED_DIM=1536
+      DB_HOST=localhost
+      DB_NAME=rag_db
+      DB_USER=rag_user
+      DB_PASSWORD=your_password_here
 
-3. Open in VS Code with Dev Containers:
-     * Open the project in VS Code
-     * Click the green button in the bottom-left corner
-     * Select "Reopen in Container"
-4. Start the application:
+3. Install deps & start the app:
 
       streamlit run app.py
 
 The app will be available at `http://localhost:8501`
 
 ## Usage
@@
 ### Ask Questions
   1. Select "Ask a Question" from the sidebar
   2. Enter your question
-  3. Click "Get Answer"
+  3. Choose Top-K, toggle **LLM re-rank** if desired, click "Get Answer"
+  4. You’ll see cosine distances for each retrieved chunk (lower = closer)
+  5. Sources are listed under the generated answer
 
 ## Technical Details
-  * Web Crawling: Uses Playwright via crawl4ai for JavaScript-rendered content
-  * Embeddings: OpenAI's text-embedding-ada-002 model
-  * Vector Storage: PostgreSQL with pgvector extension
-  * Text Generation: OpenAI's GPT-3.5-turbo model
-  * Chunking: Smart text splitting with token counting via tiktoken
-  * PDF Processing: PyPDF2 for text extraction from PDF documents
+  * Crawling: crawl4ai (Playwright under the hood for JS pages)
+  * Embeddings: `text-embedding-3-small` (default 1536-dim; set `EMBED_DIM` if needed)
+  * Vector Store: PostgreSQL + pgvector (IVFFlat with cosine ops)
+  * Generation: `gpt-5-mini` by default; override with `OPENAI_CHAT_MODEL`
+  * Chunking: sentence-based by default; optional paragraph-aware with overlap (commented in code)
+  * PDF: pypdf
 
 ## About
 RAG example
