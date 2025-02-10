"""
RAG Application Example using crawl4ai, pgvector, OpenAI and Streamlit.

Requirements:
  - Python 3.8+
  - Environment variables:
      OPENAI_API_KEY
      DB_HOST (default: localhost)
      DB_NAME (e.g., mydatabase)
      DB_USER (e.g., myuser)
      DB_PASSWORD (e.g., mypassword)

Install dependencies:
    pip install openai psycopg2-binary pgvector streamlit crawl4ai
"""

import asyncio
import os
import openai
import psycopg2
from pgvector.psycopg2 import register_vector
import streamlit as st
import tiktoken

# Hypothetical import – adjust according to your actual crawling library.
# For example, if you have a CrawlAI class in crawl4ai:
from crawl4ai import AsyncWebCrawler

# Set your OpenAI API key (or ensure the environment variable is set)
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_db_connection():
    """
    Establish a connection to PostgreSQL and register pgvector type.
    """
    try:
        # Get database credentials with more specific error handling
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "rag_db")
        db_user = os.getenv("DB_USER", "rag_user")
        db_password = os.getenv("DB_PASSWORD")
        
        # Debug information
        print(f"Attempting connection with:")
        print(f"Host: {db_host}")
        print(f"Database: {db_name}")
        print(f"User: {db_user}")
        print(f"Password set: {'Yes' if db_password else 'No'}")
        
        if not db_password:
            raise ValueError("DB_PASSWORD environment variable is not set")
            
        conn = psycopg2.connect(
            host=db_host,
            dbname=db_name,
            user=db_user,
            password=db_password,
            # Add connection timeout
            connect_timeout=10
        )
        register_vector(conn)
        return conn
    except psycopg2.OperationalError as e:
        raise Exception(f"Failed to connect to database: {e}")
    except Exception as e:
        raise Exception(f"Error establishing database connection: {e}")


def setup_db(conn):
    """
    Create the pgvector extension and documents table if they do not exist.
    Assumes that the embedding dimension is 1536 (as returned by text-embedding-ada-002).
    """
    try:
        with conn.cursor() as cur:
            # First check if we have necessary permissions
            cur.execute("SELECT current_user;")
            current_user = cur.fetchone()[0]
            
            # Try to create extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create the documents table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    url TEXT,
                    content TEXT,
                    embedding vector(1536)
                );
                """
            )
            conn.commit()
    except psycopg2.Error as e:
        raise Exception(f"Database setup failed: {e}")


def get_embedding(text: str) -> list:
    """
    Get the embedding for a given text using OpenAI's embedding API.
    """
    try:
        client = openai.OpenAI()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        raise Exception(f"Error getting embedding: {e}")


def chunk_text(text: str, max_tokens: int = 2000) -> list[str]:
    """Split text into chunks that don't exceed max_tokens."""
    # Initialize tokenizer for ada-002
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # Split paragraphs into sentences
        sentences = paragraph.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence = sentence + '. '
            sentence_tokens = len(enc.encode(sentence))
            
            # If a single sentence is too long, split it into smaller pieces
            if sentence_tokens > max_tokens:
                words = sentence.split()
                current_piece = []
                current_piece_length = 0
                
                for word in words:
                    word_tokens = len(enc.encode(word + ' '))
                    if current_piece_length + word_tokens > max_tokens:
                        # Save current piece and start new one
                        if current_piece:
                            piece_text = ' '.join(current_piece) + ' '
                            chunks.append(piece_text)
                        current_piece = [word]
                        current_piece_length = word_tokens
                    else:
                        current_piece.append(word)
                        current_piece_length += word_tokens
                
                # Add the last piece if it exists
                if current_piece:
                    piece_text = ' '.join(current_piece) + ' '
                    if current_length + len(enc.encode(piece_text)) <= max_tokens:
                        current_chunk.append(piece_text)
                        current_length += len(enc.encode(piece_text))
                    else:
                        chunks.append(''.join(current_chunk))
                        current_chunk = [piece_text]
                        current_length = len(enc.encode(piece_text))
                
            # Normal case: add sentence if it fits, otherwise start new chunk
            elif current_length + sentence_tokens > max_tokens:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        # Add paragraph break
        if current_chunk:
            current_chunk.append('\n\n')
            current_length += 2  # Approximate token count for newlines
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    # Final safety check
    for i, chunk in enumerate(chunks):
        chunk_tokens = len(enc.encode(chunk))
        if chunk_tokens > max_tokens:
            print(f"Warning: Chunk {i} has {chunk_tokens} tokens, splitting further...")
            subchunks = chunk_text(chunk, max_tokens)  # Recursive split
            chunks = chunks[:i] + subchunks + chunks[i+1:]
    
    return chunks


async def crawl_and_index(url: str):
    """
    Crawl the URL using Crawl4AI, get its content, compute its embedding, and store it in the DB.
    """
    try:
        print(f"Starting to crawl URL: {url}")  # Debug print
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            # Try different content types
            content = None
            if hasattr(result, 'markdown') and result.markdown:
                content = result.markdown
                print("Using markdown content")
            elif hasattr(result, 'fit_markdown') and result.fit_markdown:
                content = result.fit_markdown
                print("Using fit_markdown content")
            elif hasattr(result, 'text') and result.text:
                content = result.text
                print("Using text content")
            
            # Print content length for debugging
            if content:
                print(f"Content length: {len(content)} characters")
            else:
                print("No content found in any format")

        if not content:
            st.error("No content found on the provided URL.")
            return None

        # Split content into chunks
        chunks = chunk_text(content)
        st.info(f"Split content into {len(chunks)} chunks")

        doc_ids = []
        for i, chunk in enumerate(chunks, 1):
            with st.spinner(f"Processing chunk {i}/{len(chunks)}..."):
                # Print chunk info for debugging
                print(f"Chunk {i} length: {len(chunk)} characters")
                
                # Compute embedding
                embedding = get_embedding(chunk)

                # Store in PostgreSQL
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO documents (url, content, embedding) VALUES (%s, %s, %s) RETURNING id;",
                        (f"{url}#chunk{i}", chunk, embedding)
                    )
                    doc_id = cur.fetchone()[0]
                    conn.commit()
                conn.close()
                doc_ids.append(doc_id)

        return doc_ids

    except Exception as e:
        print(f"Crawling error: {str(e)}")  # Debug print
        st.error(f"Error during crawling/indexing: {e}")
        return None


def search_documents(query: str, top_k: int = 5):
    """
    Given a query, compute its embedding and perform a similarity search against stored documents.
    """
    try:
        query_embedding = get_embedding(query)
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Cast the embedding array to vector type using the correct syntax
            cur.execute(
                """
                SELECT url, content
                FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (query_embedding, top_k)
            )
            results = cur.fetchall()
        conn.close()
        return results

    except Exception as e:
        st.error(f"Error during document search: {e}")
        return []


def generate_answer(query: str, retrieved_docs: list) -> str:
    """
    Generate an answer using OpenAI's ChatCompletion API by providing the retrieved document
    contents as context.
    """
    # Concatenate the retrieved document contents to form context.
    context = "\n\n".join([doc[1] for doc in retrieved_docs if doc[1]])
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": (
                    f"Answer the question based on the context provided below.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    f"Answer:"
                )}
            ],
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, an error occurred while generating the answer."


def clear_documents():
    """
    Clear all documents from the database.
    Returns the number of documents deleted.
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # First get the count
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
            
            # Then delete all records
            cur.execute("DELETE FROM documents;")
            # Reset the auto-increment counter
            cur.execute("ALTER SEQUENCE documents_id_seq RESTART WITH 1;")
            conn.commit()
        conn.close()
        return count
    except Exception as e:
        raise Exception(f"Error clearing documents: {e}")


def main():
    st.title("RAG App using crawl4ai, pgvector and OpenAI")
    st.markdown(
        """
        This application allows you to:
        - **Index a URL:** Crawl a web page and store its content with an embedding.
        - **Ask a Question:** Retrieve relevant documents and generate an answer using context.
        - **Clear Database:** Remove all indexed documents.
        """
    )

    # Ensure the database is set up with better error handling
    try:
        conn = get_db_connection()
        setup_db(conn)
        conn.close()
    except Exception as e:
        st.error(f"""
        Database connection error: {str(e)}
        
        Please check:
        1. Database environment variables are set correctly
        2. PostgreSQL is running and accessible
        3. Database user has necessary permissions
        4. pgvector extension is available
        """)
        return

    # Sidebar options to switch between modes
    app_mode = st.sidebar.selectbox(
        "Choose mode", 
        ["Index a URL", "Ask a Question", "Clear Database"]
    )

    if app_mode == "Index a URL":
        st.header("Index a URL")
        url = st.text_input("Enter a URL to crawl and index:")
        if st.button("Index URL") and url:
            with st.spinner("Crawling and indexing..."):
                # Run the async function using asyncio
                doc_ids = asyncio.run(crawl_and_index(url))
            if doc_ids:
                st.success(f"Documents indexed successfully with IDs: {', '.join(map(str, doc_ids))}")

    elif app_mode == "Ask a Question":
        st.header("Ask a Question")
        query = st.text_input("Enter your question:")
        if st.button("Get Answer") and query:
            with st.spinner("Searching for relevant documents and generating an answer..."):
                retrieved_docs = search_documents(query)
                if not retrieved_docs:
                    st.warning("No relevant documents found in the index.")
                else:
                    st.markdown("**Retrieved Documents:**")
                    for idx, doc in enumerate(retrieved_docs, start=1):
                        st.write(f"**Document {idx}**")
                        st.write(f"URL: {doc[0]}")
                        st.write(f"Content Snippet: {doc[1][:200]}...")  # show first 200 chars
                    answer = generate_answer(query, retrieved_docs)
                    st.markdown("### Generated Answer")
                    st.write(answer)

    elif app_mode == "Clear Database":
        st.header("Clear Database")
        st.warning("⚠️ This will delete all indexed documents from the database!")
        
        # Show current document count
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents;")
                count = cur.fetchone()[0]
            conn.close()
            st.info(f"Current number of documents in database: {count}")
        except Exception as e:
            st.error(f"Error getting document count: {e}")
            return

        # Add a confirmation checkbox
        confirm = st.checkbox("I understand that this action cannot be undone")
        
        if st.button("Clear Database", disabled=not confirm):
            try:
                deleted_count = clear_documents()
                st.success(f"Successfully deleted {deleted_count} documents from the database!")
            except Exception as e:
                st.error(f"Error clearing database: {e}")


if __name__ == "__main__":
    main()
