import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Dict, Any
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class SummaryGenerator:
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summary generator with a local open-source model.
        
        open-source options:
        - "facebook/bart-large-cnn" - Best for summarization (recommended)
        - "sshleifer/distilbart-cnn-12-6" - Smaller, faster BART variant
        - "google/flan-t5-base" - Good instruction-following model
        - "microsoft/DialoGPT-medium" - Conversational model
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.summarizer = None
        self.model_loaded = False
        
        print(f"Initializing with model: {model_name}")
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model_loaded:
            return {"success": True, "message": "Model already loaded"}
        
        try:
            print(f"Loading model: {self.model_name}")
            print("This may take a few minutes for the first time...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # For summarization models like BART
            if "bart" in self.model_name.lower():
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                # Move model to device
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                
                # Create summarization pipeline
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    framework="pt"
                )
                
            # For T5 models
            elif "t5" in self.model_name.lower():
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                
                self.summarizer = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    framework="pt"
                )
            
            # For other models, try summarization pipeline
            else:
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    framework="pt"
                )
            
            self.model_loaded = True
            print(f"✅ Model {self.model_name} loaded successfully!")
            
            return {"success": True, "message": f"Model {self.model_name} loaded successfully"}
            
        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {str(e)}"
            print(f" {error_msg}")
            
            # Try fallback to a smaller model
            if self.model_name != "sshleifer/distilbart-cnn-12-6":
                print("Trying fallback model: sshleifer/distilbart-cnn-12-6")
                self.model_name = "sshleifer/distilbart-cnn-12-6"
                return self.load_model()
            
            return {"success": False, "error": error_msg}
    
    def generate_summary(self, context: str, summary_length: str = "medium") -> Dict[str, Any]:
        """Generate summary from retrieved context using the loaded model."""
        
        if not self.model_loaded:
            return {
                'summary': "Model not loaded. Please load a model first.",
                'error': "Model not loaded",
                'success': False
            }
        
        if not context or not context.strip():
            return {
                'summary': "No content provided for summarization.",
                'error': "Empty context",
                'success': False
            }
        
        # Define length parameters
        length_params = {
            "short": {"max_length": 200, "min_length": 30},
            "medium": {"max_length": 350, "min_length": 50},
            "long": {"max_length": 600, "min_length": 80}
        }
        
        params = length_params.get(summary_length, length_params["medium"])
        
        try:
            start_time = datetime.now()
            
            # Limit input length to avoid memory issues
            max_input_length = 1000  # Conservative limit
            if len(context) > max_input_length:
                context = context[:max_input_length] + "..."
            
            # Generate summary based on model type
            if "t5" in self.model_name.lower():
                # T5 models need instruction prefix
                prompt = f"summarize: {context}"
                result = self.summarizer(
                    prompt,
                    max_length=params["max_length"],
                    min_length=params["min_length"],
                    do_sample=False,
                    truncation=True
                )
                summary = result[0]['generated_text']
            else:
                # BART and other summarization models
                result = self.summarizer(
                    context,
                    max_length=params["max_length"],
                    min_length=params["min_length"],
                    do_sample=False,
                    truncation=True
                )
                summary = result[0]['summary_text']
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            # Estimate token usage (approximation)
            input_tokens = len(context.split()) * 1.3
            output_tokens = len(summary.split()) * 1.3
            
            token_usage = {
                'prompt_tokens': int(input_tokens),
                'completion_tokens': int(output_tokens),
                'total_tokens': int(input_tokens + output_tokens)
            }
            
            return {
                'summary': summary,
                'token_usage': token_usage,
                'latency': latency,
                'model': self.model_name,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'summary': error_msg,
                'error': str(e),
                'success': False
            }
    
    def get_available_models(self) -> list:
        """Get list of available local summarization models."""
        return [
            "facebook/bart-large-cnn",
            "sshleifer/distilbart-cnn-12-6",
            "google/flan-t5-base",
            "google/flan-t5-small",
            "microsoft/DialoGPT-medium"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self.model_loaded,
            'cuda_available': torch.cuda.is_available()
        }
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.summarizer
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = None
        self.tokenizer = None
        self.summarizer = None
        self.model_loaded = False
        
        print("Model unloaded and memory cleared")
