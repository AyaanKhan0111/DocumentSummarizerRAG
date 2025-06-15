import os
import PyPDF2
import markdown
from typing import List, Union
from pathlib import Path

class DocumentParser:
    """Handles document ingestion and semantic chunking."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def load_document(self, file_path: Union[str, Path]) -> str:
        """Load document content based on file extension."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._load_pdf(file_path)
        elif extension == '.txt':
            return self._load_txt(file_path)
        elif extension in ['.md', '.markdown']:
            return self._load_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _load_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _load_txt(self, file_path: Path) -> str:
        """Load text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def _load_markdown(self, file_path: Path) -> str:
        """Load and convert markdown to plain text."""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
        # Convert markdown to HTML then extract text
        html = markdown.markdown(md_content)
        # Simple HTML tag removal
        import re
        clean_text = re.sub('<.*?>', '', html)
        return clean_text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into semantically meaningful chunks using sliding window."""
        # Split by sentences first
        sentences = self._split_sentences(text)
        chunks = []
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-self.overlap:]
                current_chunk = " ".join(overlap_words) + " " + sentence
                current_size = len(overlap_words) + sentence_size
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        import re
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences