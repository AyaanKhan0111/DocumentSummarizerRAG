# RAG Document Summarizer

A desktop-based document summarization system using Retrieval-Augmented Generation (RAG). This tool leverages local open-source Large Language Models to summarize PDF, TXT, and Markdown documents on your local machine without requiring an internet connection or API keys.

---

## Features

- Local open-source models
- Multiple summarization models supported
- Input support for PDF, TXT, and Markdown files
- Intelligent document chunking with overlap
- ChromaDB-based vector search for context retrieval
- Tkinter-based desktop GUI
- Completely offline and privacy-preserving
- Modular architecture with separate components

---

## Project Structure

```
DocumentSummarizer/
├── __pycache__/               # Python bytecode cache
├── Documents/                 # Input documents for summarization
├── Summaries/                 # Output summaries saved here
├── document_parser.py         # Handles loading and chunking documents
├── embedding_retrieval.py     # Embedding and semantic similarity retrieval
├── summary_generator.py       # Generates summaries using local models
├── main.py                    # Tkinter GUI application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── report.pdf                 # Final PDF project report
└── RAG DOCUMENT SUMMARIZER.docx  # MS Word version of the report
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **That's it!** No API keys or additional setup required.

## Usage

### Running the Desktop Application

1. **Start the application**:
   \`\`\`bash
   python main.py
   \`\`\`

2. **Load an AI model**:
   - Select your preferred AI model from the dropdown
   - Click " Load Model" (first-time loading may take a few minutes)
   - Wait for the green checkmark confirmation

3. **Configure processing settings**:
   - Choose summary length: short, medium, or long
   - Adjust chunk size (256-1024 words) and overlap (25-100 words) using sliders

4. **Process documents**:
   - Click " Browse" to select a PDF, TXT, or MD file
   - Click " Generate Summary" to process the document
   - Monitor progress with the progress bar and status updates

5. **View results**:
   - ** Summary**: Generated document summary
   - ** Statistics**: Processing metrics, token usage, and performance data
   - ** Retrieved Context**: Relevant chunks used for summarization

6. **Save results**:
   - Click "Save Summary" to save the generated summary to a text file

## Available Models

The application supports various open-source models:

- **facebook/bart-large-cnn** (Recommended) - Best quality for summarization
- **sshleifer/distilbart-cnn-12-6** - Faster, smaller BART variant
- **google/flan-t5-base** - Good instruction-following model
- **google/flan-t5-small** - Fastest, smallest model
- **microsoft/DialoGPT-medium** - Conversational model

## Configuration Options

### Document Parser
- `chunk_size`: Number of words per chunk (256-1024, default: 512)
- `overlap`: Overlap between chunks in words (25-100, default: 50)

### Embedding Retrieval
- `model_name`: SentenceTransformer model (default: "all-MiniLM-L6-v2")
- `max_chunks`: Number of chunks to retrieve for context (default: 6)

### Summary Generator
- `model`: Local Hugging Face model to use
- `summary_length`: "short", "medium", or "long"

## Output Information

The application provides comprehensive information:

- **Generated Summary**: AI-created document summary
- **Retrieved Context**: Relevant chunks used for summarization
- **Processing Statistics**: 
  - Document length and chunk count
  - Processing time and performance metrics
  - Token usage estimation
  - Model and device information
- **Visual Progress**: Real-time status updates and progress indicators

## Technical Details

### RAG Pipeline
1. **Document Ingestion**: Load and parse documents (PDF/TXT/MD)
2. **Semantic Chunking**: Split into meaningful segments with overlap
3. **Vector Embedding**: Convert chunks to numerical representations
4. **Index Building**: Create ChromaDB index for fast similarity search
5. **Context Retrieval**: Find most relevant chunks for summarization
6. **Summary Generation**: Use local AI models to create coherent summary

### Models Used
- **Embeddings**: SentenceTransformers "all-MiniLM-L6-v2" (local)
- **Summarization**: Various Hugging Face models (local)
- **Vector Search**: ChromaDB with cosine similarity

## Troubleshooting

### Common Issues

1. **Model Loading Takes Long**:
   - First-time model download can take several minutes
   - Models are cached locally for future use
   - Try using smaller models like flan-t5-small for faster loading

2. **Out of Memory Error**:
   - Use smaller models (flan-t5-small, distilbart)
   - Reduce chunk size in settings
   - Close other applications to free memory

3. **File Upload Issues**:
   - Verify file format is supported (PDF, TXT, MD)
   - Check file size (very large files may cause memory issues)
   - Ensure file is not corrupted

4. **Slow Processing**:
   - Use GPU if available (CUDA)
   - Try smaller, faster models
   - Reduce number of chunks retrieved

### Performance Tips

- **For faster processing**: Use distilbart or flan-t5-small models
- **For better quality**: Use facebook/bart-large-cnn
- **For GPU acceleration**: Ensure PyTorch with CUDA is installed
- **For large documents**: Increase chunk size to reduce total chunks

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for model downloads
- **GPU**: Optional but recommended for faster processing

## Sample Test Documents

1. **ColBERT Passage Search Via BERT.pdf**: Academic paper on RAG(500-2000 words)
2. **genai.md**: Document on Generative AI in md format
3. **datamining.txt**: Data Mining Lecture in txt format

## This project addresses all requirements:

- **Document Parsing**: Clean loading, chunking, and formatting
- **Embedding & Storage**: Efficient ChromaDB and embeddings
- **Retrieval Quality**: Relevant content selection
- **Summary Generation**: Fluent, accurate summaries using local models
- **Pipeline Design**: Modular, reproducible code
- **Output Presentation**: Enhanced GUI with comprehensive results
- **Documentation** : Detailed README and code comments

##Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Try using smaller models first
4. Ensure sufficient system memory

