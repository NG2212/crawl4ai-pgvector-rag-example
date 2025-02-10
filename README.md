# RAG Question-Answering Application

A Retrieval-Augmented Generation (RAG) application that allows you to ask questions about web content. The app crawls web pages, stores their content with embeddings in a vector database, and uses OpenAI's GPT models to generate accurate answers based on the retrieved context.

## Features

- **Web Crawling**: Crawl and index any web page
- **Vector Search**: Use pgvector for semantic similarity search
- **Context-Aware Answers**: Generate answers using relevant context
- **Streamlit Interface**: Easy-to-use web interface
- **Chunk Management**: Smart text chunking for optimal embedding

## Prerequisites

- Docker and Docker Compose
- OpenAI API key
- VS Code with Dev Containers extension (recommended)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/warwickmcintosh/crawl4ai-pgvector-rag-example
   cd crawl4ai-pgvector-rag-example
   ```

2. Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   DB_HOST=db
   DB_NAME=rag_db
   DB_USER=rag_user
   DB_PASSWORD=your_password_here
   ```

3. Open in VS Code with Dev Containers:
   - Open the project in VS Code
   - Click the green button in the bottom-left corner
   - Select "Reopen in Container"

4. Start the application:
   ```bash
   streamlit run app.py
   ```

The app will be available at `http://localhost:8501`

## Usage

### Index a URL
1. Select "Index a URL" from the sidebar
2. Enter a URL to crawl
3. Click "Index URL"

### Ask Questions
1. Select "Ask a Question" from the sidebar
2. Enter your question
3. Click "Get Answer"

### Clear Database
1. Select "Clear Database" from the sidebar
2. Confirm the action to remove all indexed documents

