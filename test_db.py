import os
from dotenv import load_dotenv
from app import get_db_connection
import psycopg2

# Load environment variables from .env file
load_dotenv(override=True)

def test_postgres_connection():
    print("\n=== Testing Database Connection ===")
    
    # Print environment variables
    print("\nEnvironment Variables:")
    print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'not set')}")    
    print(f"DB_HOST: {os.getenv('DB_HOST', 'not set')}")
    print(f"DB_NAME: {os.getenv('DB_NAME', 'not set')}")
    print(f"DB_USER: {os.getenv('DB_USER', 'not set')}")
    print(f"DB_PASSWORD: {'set' if os.getenv('DB_PASSWORD') else 'not set'}")
    
    try:
        # Try basic psycopg2 connection first
        print("\nTesting basic PostgreSQL connection...")
        basic_conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            dbname=os.getenv("DB_NAME", "rag_db"),
            user=os.getenv("DB_USER", "rag_user"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=10
        )
        print("Basic PostgreSQL connection successful!")
        basic_conn.close()
        
        # Test full RAG app connection
        print("\nTesting RAG app connection with pgvector...")
        conn = get_db_connection()
        
        # Test pgvector extension
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            has_vector = cur.fetchone() is not None
            print(f"pgvector extension installed: {has_vector}")
        
        print("Full connection test successful!")
        conn.close()
        
    except Exception as e:
        print(f"\nConnection failed: {str(e)}")
        print("\nCommon solutions:")
        print("1. Ensure PostgreSQL is running:")
        print("   - On Linux: sudo service postgresql status")
        print("   - On macOS: brew services list")
        print("2. Check environment variables are set correctly")
        print("3. Verify PostgreSQL user exists and has proper permissions")
        print("4. Confirm pgvector extension is installed in the database")
        print("5. Check PostgreSQL is accepting connections:")
        print("   - Verify pg_hba.conf settings")
        print("   - Check PostgreSQL is listening on the correct port")

def test_vector_setup():
    print("\n=== Testing Vector Setup ===")
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Check if vector extension exists
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            vector_exists = cur.fetchone() is not None
            print(f"Vector extension installed: {vector_exists}")
            
            # Check table structure
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'documents';
            """)
            columns = cur.fetchall()
            print("\nTable structure:")
            for col in columns:
                print(f"Column: {col[0]}, Type: {col[1]}")
                
        conn.close()
    except Exception as e:
        print(f"Error testing vector setup: {e}")

def test_vector_operations():
    print("\n=== Testing Vector Operations ===")
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Test vector casting
            cur.execute("""
                SELECT '[1,2,3]'::vector(3);
            """)
            result = cur.fetchone()
            print(f"Vector cast test successful: {result}")
            
        conn.close()
    except Exception as e:
        print(f"Error testing vector operations: {e}")
        print("Make sure pgvector extension is installed and the user has proper permissions")

def test_documents_table():
    print("\n=== Testing Documents Table ===")
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Check document count
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
            print(f"Number of documents in database: {count}")
            
            if count > 0:
                # Check a sample document
                cur.execute("""
                    SELECT id, url, 
                           LEFT(content, 100) as content_preview,
                           array_length(embedding::float8[], 1) as embedding_dim
                    FROM documents LIMIT 1;
                """)
                doc = cur.fetchone()
                print("\nSample document:")
                print(f"ID: {doc[0]}")
                print(f"URL: {doc[1]}")
                print(f"Content preview: {doc[2]}...")
                print(f"Embedding dimensions: {doc[3]}")
            
        conn.close()
    except Exception as e:
        print(f"Error testing documents table: {e}")

if __name__ == "__main__":
    test_postgres_connection()
    test_vector_setup()
    test_vector_operations()
    test_documents_table() 