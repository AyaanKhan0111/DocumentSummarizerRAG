import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
import uuid
import os
import tempfile
import numpy as np

class EmbeddingRetrieval:
    """Handles document embedding and retrieval using ChromaDB."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.client = None
        self.collection = None
        self.chunks = None
        self.collection_name = "document_chunks"
        
    def _initialize_model(self):
        """Initialize the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully!")
        
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        if self.client is None:
            # Create a temporary directory for ChromaDB
            self.db_path = tempfile.mkdtemp()
            self.client = chromadb.PersistentClient(path=self.db_path)
        
    # Delete existing collection if it exists
        try:
            existing_collection = self.client.get_collection(name=self.collection_name)
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except:
            pass  # Collection doesn't exist, which is fine
    
    # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def build_index(self, chunks: List[str]) -> None:
        """Build ChromaDB index from document chunks."""
        if not chunks:
            raise ValueError("No chunks provided")
        
        self.chunks = chunks
        self._initialize_model()
        self._initialize_chroma()
        
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_chunks, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings.tolist())
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas = [{"chunk_id": i, "length": len(chunk)} for i, chunk in enumerate(chunks)]
        
        # Add to ChromaDB collection
        self.collection.add(
            embeddings=all_embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Built ChromaDB index with {len(chunks)} chunks")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve most relevant chunks for a query."""
        if self.collection is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.chunks) if self.chunks else 10)
        )
        
        # Extract relevant chunks and distances
        relevant_chunks = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Convert distances to similarity scores (ChromaDB returns distances, lower is better)
        similarity_scores = [max(0, 1 - dist) for dist in distances]
        
        return relevant_chunks, similarity_scores
    
    def get_context_for_summary(self, max_chunks: int = 6) -> str:
        """Get context for general document summarization."""
        if not self.chunks:
            return ""
        
        # For summarization, we want representative chunks
        # Use a general summary query
        summary_query = "main ideas key points important information summary overview"
        
        try:
            relevant_chunks, scores = self.retrieve_relevant_chunks(
                summary_query, top_k=min(max_chunks, len(self.chunks))
            )
            
            # Combine relevant chunks with some spacing
            context = "\n\n---\n\n".join(relevant_chunks)
            
            # Limit context length to avoid model limits
            max_context_length = 3000  # Conservative limit
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            # Fallback: return first few chunks with length limit
            fallback_chunks = self.chunks[:max_chunks]
            fallback_context = "\n\n---\n\n".join(fallback_chunks)
            if len(fallback_context) > 3000:
                fallback_context = fallback_context[:3000] + "..."
            return fallback_context
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.collection is None:
            return {"total_chunks": 0}
        
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "embedding_model": self.model_name
            }
        except:
            return {"total_chunks": len(self.chunks) if self.chunks else 0}
    
    def cleanup(self):
        """Clean up ChromaDB resources."""
        if self.client:
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass
        
        # Clean up temporary directory
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path, ignore_errors=True)
