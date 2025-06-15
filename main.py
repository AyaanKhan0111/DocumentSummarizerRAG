import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
from document_parser import DocumentParser
from embedding_retrieval import EmbeddingRetrieval
from summary_generator import SummaryGenerator

class RAGSummarizerGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Document Summarizer - Local Models")
        self.root.geometry("1000x700")
        
        # Initialize components
        self.parser = DocumentParser()
        self.retriever = EmbeddingRetrieval()
        self.generator = SummaryGenerator()
        self.model_loaded = False
        
        # Variables
        self.file_path_var = tk.StringVar()
        self.chunk_size_var = tk.IntVar(value=512)
        self.overlap_var = tk.IntVar(value=50)
        self.model_var = tk.StringVar(value="facebook/bart-large-cnn")
        self.summary_length_var = tk.StringVar(value="medium")
        self.processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsive design
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="📚 RAG Document Summarizer (Local Models)", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        # Model Selection
        ttk.Label(config_frame, text="AI Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        model_combo = ttk.Combobox(config_frame, textvariable=self.model_var, 
                                  values=self.generator.get_available_models(), 
                                  state="readonly", width=40)
        model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        self.load_model_btn = ttk.Button(config_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Model status
        self.model_status_label = ttk.Label(config_frame, text="❌ No model loaded", foreground="red")
        self.model_status_label.grid(row=1, column=0, columnspan=3, pady=(5, 0))
        
        # Summary length
        ttk.Label(config_frame, text="Summary Length:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        length_combo = ttk.Combobox(config_frame, textvariable=self.summary_length_var,
                                   values=["short", "medium", "long"], state="readonly", width=20)
        length_combo.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Chunk settings
        ttk.Label(config_frame, text="Chunk Size:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        chunk_size_scale = ttk.Scale(config_frame, from_=256, to=1024, variable=self.chunk_size_var, 
                                    orient=tk.HORIZONTAL, length=200)
        chunk_size_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(10, 0), padx=(0, 5))
        
        self.chunk_size_label = ttk.Label(config_frame, text="512")
        self.chunk_size_label.grid(row=3, column=2, pady=(10, 0))
        chunk_size_scale.configure(command=self.update_chunk_size_label)
        
        ttk.Label(config_frame, text="Overlap:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        overlap_scale = ttk.Scale(config_frame, from_=25, to=100, variable=self.overlap_var, 
                                 orient=tk.HORIZONTAL, length=200)
        overlap_scale.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=(5, 0), padx=(0, 5))
        
        self.overlap_label = ttk.Label(config_frame, text="50")
        self.overlap_label.grid(row=4, column=2, pady=(5, 0))
        overlap_scale.configure(command=self.update_overlap_label)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Document Selection", padding="10")
        file_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        file_select_frame.columnconfigure(0, weight=1)
        
        self.file_path_entry = ttk.Entry(file_select_frame, textvariable=self.file_path_var, state="readonly")
        self.file_path_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.browse_btn = ttk.Button(file_select_frame, text="Browse", command=self.browse_file)
        self.browse_btn.grid(row=0, column=1)
        
        # Process button
        self.process_btn = ttk.Button(file_frame, text="🚀 Generate Summary", 
                                     command=self.process_document,
                                     state="disabled")
        self.process_btn.grid(row=1, column=0, pady=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(file_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status label
        self.status_label = ttk.Label(file_frame, text="Please load a model first", foreground="orange")
        self.status_label.grid(row=3, column=0, pady=(5, 0))
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD, height=15)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Statistics tab
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, height=15)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Context tab
        context_frame = ttk.Frame(self.notebook)
        self.notebook.add(context_frame, text="Retrieved Context")
        context_frame.columnconfigure(0, weight=1)
        context_frame.rowconfigure(0, weight=1)
        
        self.context_text = scrolledtext.ScrolledText(context_frame, wrap=tk.WORD, height=15)
        self.context_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Save button
        self.save_btn = ttk.Button(results_frame, text="💾 Save Summary", 
                                  command=self.save_summary, state="disabled")
        self.save_btn.grid(row=1, column=0, pady=(10, 0))
        
    def load_model(self):
        """Load the selected model."""
        if self.processing:
            return
            
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model")
            return
        
        # Start model loading in background thread
        self.load_model_btn.config(state="disabled")
        self.model_status_label.config(text="⏳ Loading model... (This may take a few minutes)", foreground="orange")
        self.progress.start(10)
        
        thread = threading.Thread(target=self._load_model_thread, args=(model_name,))
        thread.daemon = True
        thread.start()
    
    def _load_model_thread(self, model_name):
        """Load model in background thread."""
        try:
            # Create new generator with selected model
            self.generator = SummaryGenerator(model_name)
            result = self.generator.load_model()
            
            if result['success']:
                self.model_loaded = True
                self.root.after(0, lambda: self.model_status_label.config(
                    text=f"✅ Model loaded: {model_name}", foreground="green"
                ))
                self.root.after(0, lambda: self.update_ui_state())
            else:
                error_msg = result.get('error', 'Unknown error')
                self.root.after(0, lambda: self.model_status_label.config(
                    text=f"❌ Failed to load model: {error_msg}", foreground="red"
                ))
                
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.root.after(0, lambda: self.model_status_label.config(
                text=f"❌ {error_msg}", foreground="red"
            ))
        
        finally:
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.load_model_btn.config(state="normal"))
    
    def on_model_change(self, event=None):
        """Handle model selection change."""
        # Reset model loaded status when model changes
        self.model_loaded = False
        self.model_status_label.config(text="❌ Model changed - please load the new model", foreground="orange")
        self.update_ui_state()
    
    def update_chunk_size_label(self, value):
        """Update chunk size label."""
        self.chunk_size_label.config(text=str(int(float(value))))
        
    def update_overlap_label(self, value):
        """Update overlap label."""
        self.overlap_label.config(text=str(int(float(value))))
    
    def update_ui_state(self):
        """Update UI state based on model and file status."""
        if self.model_loaded and self.file_path_var.get():
            self.process_btn.config(state="normal")
            self.status_label.config(text="✅ Ready to process documents", foreground="green")
        elif self.model_loaded:
            self.status_label.config(text="📁 Please select a document", foreground="orange")
        else:
            self.process_btn.config(state="disabled")
            self.status_label.config(text="🧠 Please load a model first", foreground="orange")
    
    def browse_file(self):
        """Browse and select document file."""
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("All Supported", "*.pdf;*.txt;*.md"),
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.update_ui_state()
    
    def process_document(self):
        """Process document in a separate thread."""
        if self.processing:
            return
            
        # Validate inputs
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load a model first")
            return
            
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select a document file")
            return
        
        # Start processing in background thread
        self.processing = True
        self.process_btn.config(state="disabled")
        self.progress.start(10)
        
        thread = threading.Thread(target=self._process_document_thread)
        thread.daemon = True
        thread.start()
    
    def _process_document_thread(self):
        """Process document in background thread."""
        try:
            # Update parser settings
            self.parser.chunk_size = self.chunk_size_var.get()
            self.parser.overlap = self.overlap_var.get()
            
            file_path = self.file_path_var.get()
            
            # Step 1: Document parsing
            self.root.after(0, lambda: self.status_label.config(text="📖 Loading document...", foreground="orange"))
            document_text = self.parser.load_document(file_path)
            
            if not document_text.strip():
                raise ValueError("Document appears to be empty")
            
            # Step 2: Chunking
            self.root.after(0, lambda: self.status_label.config(text="✂️ Creating chunks...", foreground="orange"))
            chunks = self.parser.chunk_text(document_text)
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Step 3: Embedding and indexing
            self.root.after(0, lambda: self.status_label.config(text="🔍 Building search index...", foreground="orange"))
            self.retriever.build_index(chunks)
            
            # Step 4: Context retrieval
            self.root.after(0, lambda: self.status_label.config(text="📋 Retrieving context...", foreground="orange"))
            context = self.retriever.get_context_for_summary()
            
            # Step 5: Summary generation
            self.root.after(0, lambda: self.status_label.config(text="🤖 Generating summary...", foreground="orange"))
            summary_result = self.generator.generate_summary(context, self.summary_length_var.get())
            
            if not summary_result['success']:
                raise ValueError(summary_result.get('error', 'Unknown error'))
            
            # Prepare results
            result = {
                "success": True,
                "document_length": len(document_text),
                "num_chunks": len(chunks),
                "context": context,
                "summary": summary_result['summary'],
                "token_usage": summary_result['token_usage'],
                "latency": summary_result['latency'],
                "model": summary_result['model'],
                "chunks": chunks
            }
            
            # Update UI with results
            self.root.after(0, lambda: self._display_results(result))
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            self.root.after(0, lambda: self._display_error(error_msg))
        
        finally:
            # Clean up
            self.processing = False
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
    
    def _display_results(self, result):
        """Display processing results in the UI."""
        # Clear previous results
        self.summary_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)
        self.context_text.delete(1.0, tk.END)
        
        # Display summary
        self.summary_text.insert(tk.END, result['summary'])
        
        # Display statistics
        stats = f"""📊 DOCUMENT PROCESSING STATISTICS
{'='*50}

📄 Document Information:
   • File: {Path(self.file_path_var.get()).name}
   • Document Length: {result['document_length']:,} characters
   • Number of Chunks: {result['num_chunks']}

🧠 AI Model Information:
   • Model: {result['model']}
   • Device: {self.generator.device}
   • CUDA Available: {self.generator.get_model_info()['cuda_available']}

⚙️ Processing Settings:
   • Chunk Size: {self.chunk_size_var.get()} words
   • Overlap: {self.overlap_var.get()} words
   • Summary Length: {self.summary_length_var.get()}

🚀 Performance Metrics:
   • Processing Time: {result['latency']:.2f} seconds
   • Input Tokens (est.): {result['token_usage']['prompt_tokens']:,}
   • Output Tokens (est.): {result['token_usage']['completion_tokens']:,}
   • Total Tokens (est.): {result['token_usage']['total_tokens']:,}

🔍 Retrieval Information:
   • Retrieved Chunks: {len(result['context'].split('---'))}
   • Context Length: {len(result['context']):,} characters

✅ Status: Summary generated successfully using local AI model!
"""
        self.stats_text.insert(tk.END, stats)
        
        # Display context
        self.context_text.insert(tk.END, result['context'])
        
        # Enable save button
        self.save_btn.config(state="normal")
        
        # Update status
        self.status_label.config(text="✅ Summary generated successfully!", foreground="green")
        
        # Switch to summary tab
        self.notebook.select(0)
    
    def _display_error(self, error_msg):
        """Display error message."""
        self.status_label.config(text=f"❌ {error_msg}", foreground="red")
        messagebox.showerror("Processing Error", error_msg)
    
    def save_summary(self):
        """Save summary to file."""
        if self.summary_text.get(1.0, tk.END).strip():
            file_path = filedialog.asksaveasfilename(
                title="Save Summary",
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.summary_text.get(1.0, tk.END))
                    messagebox.showinfo("Success", f"Summary saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file: {str(e)}")

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    
    # Create application
    app = RAGSummarizerGUI(root)
    
    # Handle window closing
    def on_closing():
        try:
            app.retriever.cleanup()
            if hasattr(app, 'generator') and app.generator:
                app.generator.unload_model()
        except:
            pass
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
