import joblib
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from typing import List, Union

class ClassicalModelLoader:
    """
    Loads and uses a saved classical ML model (SVM pipeline).
    """
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to the directory containing classical_model.pkl
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Loads the saved SVM pipeline from disk."""
        model_file = os.path.join(self.model_path, "classical_model.pkl")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}")
        
        print(f"Loading classical model from {model_file}...")
        self.model = joblib.load(model_file)
        print("Classical model loaded successfully.")
    
    def predict(self, texts: Union[List[str], str]) -> list:
        """
        Predicts labels for given text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            List of predicted label integers
        """
        if self.model is None:
            raise Exception("Model not loaded!")
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.model.predict(texts)
        return predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)


class LLMModelLoader:
    """
    Loads and uses a saved LLM model with LoRA adapters.
    """
    def __init__(self, model_path: str, label_names: list, base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Args:
            model_path: Path to the directory containing saved LoRA adapters
            label_names: List of emotion label names (e.g., ['disgust', 'anger', ...])
            base_model_name: Name of the base model used for training
        """
        self.model_path = model_path
        self.label_names = label_names
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Loads the base model and applies saved LoRA adapters."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading base model: {self.base_model_name}...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(self.label_names),
            problem_type="single_label_classification",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        print(f"Loading LoRA adapters from {self.model_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("LLM model loaded successfully.")
    
    def predict(self, texts: Union[List[str], str]) -> list:
        """
        Predicts labels for given text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            List of predicted label integers
        """
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded!")
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Create prompts with label names
        label_str = ", ".join(self.label_names)
        prompts = [
            f"Instruct: Classify the emotion of the following movie review into one of these categories: {label_str}.\n"
            f"Input: {text}\n"
            f"Output:" 
            for text in texts
        ]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy().tolist()