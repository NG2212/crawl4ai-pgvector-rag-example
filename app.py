"""
RAG Application Example using crawl4ai, pgvector, OpenAI and Streamlit.
2025 refresh:
 - GPT-5-mini (toggle via env) for chat
 - text-embedding-3-small (toggle via env) for embeddings
 - Cosine distance search with pgvector + IVFFlat index
 - Optional LLM re-rank over top-10
 - Source list under answers
"""

import asyncio
import os
import re
from io import BytesIO
from typing import List, Tuple

import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector
import streamlit as st
import tiktoken
from pypdf import PdfReader

from crawl4ai import AsyncWebCrawler
from openai import OpenAI

# ---------- OpenAI client & model defaults ----------
client = OpenAI()  # reads OPENAI_API_KEY from env
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))  # 1536 for 3-small; 3072 for 3-large


def get_db_connection():
    """
    Establish a connection to PostgreSQL and register pgvector type.
    """
    try:
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "rag_db")
        db_user = os.getenv("DB_USER", "rag_user")
        db_password = os.getenv("DB_PASSWORD")

        if not db_password:
            raise ValueError("DB_PASSWORD environment variable is not set")

        conn = psycopg2.connect(
            host=db_host,
            dbname=db_name,
            user=db_user,
            password=db_password,
            connect_timeout=10,
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
    Uses configurable embedding dimension (EMBED_DIM).
    """
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                  id SERIAL PRIMARY KEY,
                  url TEXT,
                  content TEXT,
                  embedding vector({EMBED_DIM})
                );
                """
            )
            # ANN index for cosine distance; adjust 'lists' as your corpus grows
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            )
            conn.commit()
    except psycopg2.Error as e:
        raise Exception(f"Database setup failed: {e}")


def get_embedding(text: str) -> List[float]:
    """Get the embedding for a given text using OpenAI's embedding API."""
    try:
        kwargs = {"model": EMBED_MODEL, "input": text}
        # Optional: if EMBED_DIM is set, request that dimension (3-series supports this)
        if os.getenv("EMBED_DIM"):
            kwargs["dimensions"] = EMBED_DIM
        resp = client.embeddings.create(**kwargs)
        return resp.data[0].embedding
    except Exception as e:
        raise Exception(f"Error getting embedding: {e}")


def chunk_text(text: str, max_tokens: int = 2000) -> List[str]:
    """Split text into chunks that don't exceed max_tokens (simple sentence-based)."""
    enc = tiktoken.get_encoding("cl100k_base")

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        sentences = paragraph.split(". ")

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence = sentence + ". "
            sentence_tokens = len(enc.encode(sentence))

            if sentence_tokens > max_tokens:
                words = sentence.split()
                current_piece: List[str] = []
                current_piece_length = 0

                for word in words:
                    word_tokens = len(enc.encode(word + " "))
                    if current_piece_length + word_tokens > max_tokens:
                        if current_piece:
                            piece_text = " ".join(current_piece) + " "
                            chunks.append(piece_text)
                        current_piece = [word]
                        current_piece_length = word_tokens
                    else:
                        current_piece.append(word)
                        current_piece_length += word_tokens

                if current_piece:
                    piece_text = " ".join(current_piece) + " "
                    if current_length + len(enc.encode(piece_text)) <= max_tokens:
                        current_chunk.append(piece_text)
                        current_length += len(enc.encode(piece_text))
                    else:
                        chunks.append("".join(current_chunk))
                        current_chunk = [piece_text]
                        current_length = len(enc.encode(piece_text))

            elif current_length + sentence_tokens > max_tokens:
                chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens

        if current_chunk:
            current_chunk.append("\n\n")
            current_length += 2  # rough newline tokens

    if current_chunk:
        chunks.append("".join(current_chunk))

    # Safety pass
    for i, c in enumerate(list(chunks)):  # list() to avoid modifying while iterating
        if len(enc.encode(c)) > max_tokens:
            # naive split fallback
            mid = len(c) // 2
            chunks[i : i + 1] = [c[:mid], c[mid:]]

    return chunks


# --- OPTIONAL: Paragraph-aware chunking with overlap (commented out) ---
# def chunk_text_paragraph_overlap(
#     text: str,
#     target_tokens_min: int = 200,
#     target_tokens_max: int = 400,
#     overlap_ratio: float = 0.15,
# ) -> List[str]:
#     """
#     Paragraph-first chunker with a token budget and ~10–20% overlap.
#     Enable by calling this instead of chunk_text() in the indexing paths.
#     """
#     enc = tiktoken.get_encoding("cl100k_base")
#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
#     chunks: List[str] = []
#     current: List[str] = []
#     cur_tokens = 0
#
#     def toklen(s: str) -> int:
#         return len(enc.encode(s))
#
#     for p in paragraphs:
#         t = toklen(p)
#         if t > target_tokens_max:
#             # break large paragraph into sentences
#             for s in re.split(r"(?<=[.!?])\s+", p):
#                 if not s:
#                     continue
#                 stoks = toklen(s)
#                 if cur_tokens + stoks > target_tokens_max and current:
#                     chunks.append("\n".join(current))
#                     # overlap from tail of last chunk
#                     if overlap_ratio > 0 and chunks:
#                         overlap_text = chunks[-1]
#                         overlap_tokens = int(toklen(overlap_text) * overlap_ratio)
#                         tail = enc.decode(enc.encode(overlap_text)[-overlap_tokens:]) if overlap_tokens else ""
#                         current, cur_tokens = ([tail] if tail else []), toklen(tail)
#                     else:
#                         current, cur_tokens = [], 0
#                 current.append(s)
#                 cur_tokens += stoks
#         else:
#             if cur_tokens + t > target_tokens_max and current:
#                 chunks.append("\n".join(current))
#                 if overlap_ratio > 0 and chunks:
#                     overlap_text = chunks[-1]
#                     overlap_tokens = int(toklen(overlap_text) * overlap_ratio)
#                     tail = enc.decode(enc.encode(overlap_text)[-overlap_tokens:]) if overlap_tokens else ""
#                     current, cur_tokens = ([tail] if tail else []), toklen(tail)
#                 else:
#                     current, cur_tokens = [], 0
#             current.append(p)
#             cur_tokens += t
#
#         if cur_tokens >= target_tokens_min:
#             chunks.append("\n".join(current))
#             current, cur_tokens = [], 0
#
#     if current:
#         chunks.append("\n".join(current))
#     return chunks


async def crawl_and_index(url: str):
    """
    Crawl the URL using Crawl4AI, get its content, compute its embedding, and store it in the DB.
    """
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

            content = None
            if hasattr(result, "markdown") and result.markdown:
                content = result.markdown
            elif hasattr(result, "fit_markdown") and result.fit_markdown:
                content = result.fit_markdown
            elif hasattr(result, "text") and result.text:
                content = result.text

        if not content:
            st.error("No content found on the provided URL.")
            return None

        # Choose your chunker (default simple; optional paragraph-aware is commented above)
        chunks = chunk_text(content)
        st.info(f"Split content into {len(chunks)} chunks")

        doc_ids = []
        for i, chunk in enumerate(chunks, 1):
            with st.spinner(f"Processing chunk {i}/{len(chunks)}..."):
                embedding = get_embedding(chunk)

                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO documents (url, content, embedding) VALUES (%s, %s, %s) RETURNING id;",
                        (f"{url}#chunk{i}", chunk, Vector(embedding)),
                    )
                    doc_id = cur.fetchone()[0]
                    conn.commit()
                conn.close()
                doc_ids.append(doc_id)

        return doc_ids

    except Exception as e:
        st.error(f"Error during crawling/indexing: {e}")
        return None


def llm_rerank(query: str, results: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    """
    Ask the LLM to re-order the 10 retrieved docs by relevance.
    Input: list of (url, content, distance). Output: same list, re-ordered.
    """
    slate_lines = []
    for i, (url, content, distance) in enumerate(results, 1):
        snippet = (content or "")[:500].replace("\n", " ")
        slate_lines.append(f"[{i}] url={url} | cos_dist={distance:.4f} | snippet: {snippet}")
    slate = "\n".join(slate_lines)

    prompt = (
        "Rank the following documents by their relevance to the user query, most relevant first. "
        "Return ONLY a comma-separated list of document numbers (e.g., '3,1,2,...').\n\n"
        f"User query: {query}\n\nDocuments:\n{slate}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful ranking model."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        order = [int(x) for x in re.findall(r"\d+", text)]
        seen, final = set(), []
        for idx in order:
            if 1 <= idx <= len(results) and idx not in seen:
                final.append(results[idx - 1])
                seen.add(idx)
        if len(final) < len(results):
            for j, r in enumerate(results, 1):
                if j not in seen:
                    final.append(r)
        return final
    except Exception as e:
        print(f"LLM re-rank failed: {e}")
        return results


def search_documents(query: str, top_k: int = 5, use_rerank: bool = False):
    """
    Given a query, compute its embedding and perform a cosine-distance search.
    Optionally LLM re-rank the top 10 to choose the final top_k.
    """
    try:
        query_embedding = get_embedding(query)
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT url, content, (embedding <=> %s)::float AS distance
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT 10;
                """,
                (Vector(query_embedding), Vector(query_embedding)),
            )
            results = cur.fetchall()  # [(url, content, distance), ...]
        conn.close()

        if not use_rerank:
            return results[:top_k]

        reranked = llm_rerank(query, results)
        return reranked[:top_k]

    except Exception as e:
        st.error(f"Error during document search: {e}")
        return []


def generate_answer(query: str, retrieved_docs: List[Tuple[str, str, float]]) -> str:
    """
    Generate an answer using OpenAI's ChatCompletion API by providing the retrieved document
    contents as context.
    """
    context = "\n\n".join([doc[1] for doc in retrieved_docs if doc[1]])
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Answer the question based on the context provided below.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "Answer:"
                    ),
                },
            ],
            temperature=0.2,
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
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
            cur.execute("DELETE FROM documents;")
            cur.execute("ALTER SEQUENCE documents_id_seq RESTART WITH 1;")
            conn.commit()
        conn.close()
        return count
    except Exception as e:
        raise Exception(f"Error clearing documents: {e}")


def process_pdf(pdf_file):
    """
    Extract text from a PDF file and return it as a string.
    """
    try:
        reader = PdfReader(BytesIO(pdf_file.getvalue()))
        text = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(text)
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")


async def index_pdf_content(content: str, source_name: str):
    """
    Split PDF content into chunks, compute embeddings, and store in database.
    """
    try:
        chunks = chunk_text(content)  # swap to chunk_text_paragraph_overlap(...) if desired
        st.info(f"Split PDF into {len(chunks)} chunks")

        doc_ids = []
        for i, chunk in enumerate(chunks, 1):
            with st.spinner(f"Processing chunk {i}/{len(chunks)}..."):
                embedding = get_embedding(chunk)

                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO documents (url, content, embedding) VALUES (%s, %s, %s) RETURNING id;",
                        (f"{source_name}#chunk{i}", chunk, Vector(embedding)),
                    )
                    doc_id = cur.fetchone()[0]
                    conn.commit()
                conn.close()
                doc_ids.append(doc_id)

        return doc_ids

    except Exception as e:
        st.error(f"Error during PDF indexing: {e}")
        return None


def _base_url(u: str) -> str:
    """Strip #chunk suffix to show a cleaner source URL."""
    return u.split("#")[0] if u else u


def main():
    st.title("RAG App using crawl4ai, pgvector and OpenAI")
    st.markdown(
        """
        This application allows you to:
        - **Index a URL:** Crawl a web page and store its content with an embedding.
        - **Upload PDF:** Extract text from a PDF and store its content with an embedding.
        - **Ask a Question:** Retrieve relevant documents (with cosine distance), optionally re-rank with an LLM, and generate an answer using context. Sources are listed under the answer.
        - **Clear Database:** Remove all indexed documents.
        """
    )

    # Ensure the database is set up
    try:
        conn = get_db_connection()
        setup_db(conn)
        conn.close()
    except Exception as e:
        st.error(
            f"""
        Database connection error: {str(e)}
        
        Please check:
        1. Database environment variables are set correctly
        2. PostgreSQL is running and accessible
        3. Database user has necessary permissions
        4. pgvector extension is available
        """
        )
        return

    # Sidebar
    st.sidebar.header("Mode")
    app_mode = st.sidebar.selectbox(
        "Choose mode", ["Index a URL", "Upload PDF", "Ask a Question", "Clear Database"]
    )

    if app_mode == "Ask a Question":
        st.sidebar.header("Retrieval Settings")
        top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=5, step=1)
        use_rerank = st.sidebar.checkbox("Use LLM re-ranker (top-10 → top-K)", value=False)
        show_scores = st.sidebar.checkbox("Show cosine distance for retrieved docs", value=True)
    else:
        top_k, use_rerank, show_scores = 5, False, True

    if app_mode == "Index a URL":
        st.header("Index a URL")
        url = st.text_input("Enter a URL to crawl and index:")
        if st.button("Index URL") and url:
            with st.spinner("Crawling and indexing..."):
                doc_ids = asyncio.run(crawl_and_index(url))
            if doc_ids:
                st.success(f"Documents indexed successfully with IDs: {', '.join(map(str, doc_ids))}")

    elif app_mode == "Upload PDF":
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.0f} KB",
            }
            st.write("File Details:")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")

            if st.button("Process PDF"):
                try:
                    with st.spinner("Processing PDF..."):
                        pdf_text = process_pdf(uploaded_file)
                        st.info(f"Extracted {len(pdf_text)} characters from PDF")

                        doc_ids = asyncio.run(index_pdf_content(pdf_text, uploaded_file.name))

                        if doc_ids:
                            st.success(f"PDF indexed successfully with {len(doc_ids)} chunks!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    elif app_mode == "Ask a Question":
        st.header("Ask a Question")
        query = st.text_input("Enter your question:")
        if st.button("Get Answer") and query:
            with st.spinner("Searching for relevant documents and generating an answer..."):
                retrieved_docs = search_documents(query, top_k=top_k, use_rerank=use_rerank)
                if not retrieved_docs:
                    st.warning("No relevant documents found in the index.")
                else:
                    st.markdown("**Retrieved Documents:**")
                    for idx, (url, content, distance) in enumerate(retrieved_docs, start=1):
                        st.write(f"**Document {idx}**")
                        st.write(f"URL: {_base_url(url)}")
                        if show_scores:
                            st.write(f"Cosine distance: {distance:.4f} (lower is better)")
                        st.write(f"Content Snippet: {content[:200]}...")

                    answer = generate_answer(query, retrieved_docs)
                    st.markdown("### Generated Answer")
                    st.write(answer)

                    # Sources (deduped base URLs)
                    st.markdown("**Sources:**")
                    seen = set()
                    for (url, _, _) in retrieved_docs:
                        base = _base_url(url)
                        if base and base not in seen:
                            st.markdown(f"- [{base}]({base})")
                            seen.add(base)

    elif app_mode == "Clear Database":
        st.header("Clear Database")
        st.warning("⚠️ This will delete all indexed documents from the database!")

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

        confirm = st.checkbox("I understand that this action cannot be undone")

        if st.button("Clear Database", disabled=not confirm):
            try:
                deleted_count = clear_documents()
                st.success(f"Successfully deleted {deleted_count} documents from the database!")
            except Exception as e:
                st.error(f"Error clearing database: {e}")


if __name__ == "__main__":
    main()
