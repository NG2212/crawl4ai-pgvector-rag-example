import asyncio
import os
from dotenv import load_dotenv
from app import get_db_connection, get_embedding
from crawl4ai import AsyncWebCrawler
import tiktoken

# Load environment variables
load_dotenv(override=True)

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
    
    print(f"Created {len(chunks)} chunks, token counts: {[len(enc.encode(chunk)) for chunk in chunks]}")
    return chunks

async def test_index_url():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    try:
        print(f"Crawling URL: {url}")
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            content = result.markdown  # or result.fit_markdown for main content
        
        if not content:
            print("No content found on the provided URL.")
            return None

        # Split content into chunks
        print("Splitting content into chunks...")
        chunks = chunk_text(content)
        print(f"Created {len(chunks)} chunks")

        doc_ids = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i}/{len(chunks)}...")
            
            print("Computing embedding...")
            embedding = get_embedding(chunk)

            print("Storing in database...")
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
            print(f"Stored chunk {i} with ID: {doc_id}")

        print(f"\nSuccessfully indexed all chunks. Document IDs: {doc_ids}")
        return doc_ids

    except Exception as e:
        print(f"Error during indexing: {e}")
        return None

if __name__ == "__main__":
    # Install tiktoken if not already installed
    try:
        import tiktoken
    except ImportError:
        print("Installing tiktoken...")
        import subprocess
        subprocess.check_call(["pip", "install", "tiktoken"])
        import tiktoken
    
    asyncio.run(test_index_url()) 