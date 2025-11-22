import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import pandas as pd
import numpy as np

class LLMLabeler:
    def __init__(self, num_labels: int, label_names: list, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.num_labels = num_labels
        self.label_names = label_names
        
        print(f"Loading Tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Loading Base Model: {model_name}...")
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            problem_type="single_label_classification",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # --- LORA CONFIGURATION ---
        print("Applying LoRA (Low-Rank Adaptation)...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, # Sequence Classification
            inference_mode=False,
            r=8,            # Rank (Adjustable: 8, 16, 32)
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ] 
        )
        
        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters()
        self.model.to(self.device)

    def _tokenize_function(self, examples):
    # Create label string: "disgust, anger, sadness, joy, neutral, fear, surprise"
        label_str = ", ".join(self.label_names)
    
        prompts = [
            f"Instruct: Classify the emotion of the following movie review into one of these categories: {label_str}.\n"
            f"Input: {text}\n"
            f"Output:" 
            for text in examples["text"]
        ]
        return self.tokenizer(
            prompts, 
            truncation=True, 
            max_length=512, 
            padding="max_length"
        )

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir="./results_qwen_lora"):
        """Fine-tunes the LoRA adapters."""
        print(f"Training LoRA Model...")
        
        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        val_ds = Dataset.from_pandas(val_df[['text', 'label']])

        tokenized_train = train_ds.map(self._tokenize_function, batched=True)
        tokenized_val = val_ds.map(self._tokenize_function, batched=True)
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=10,
            bf16=torch.cuda.is_available(),
            optim="adamw_torch",
            report_to="none"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        self.model.config.use_cache = False 
        
        self.trainer.train()
        print("LoRA Training Complete.")

    def save_model(self, path: str):
        """
        Saves the LoRA adapters and tokenizer.
        Note: This does NOT save the full base model, only the small adapter weights.
        """
        print(f"Saving model to {path}...")
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save only the LoRA adapters (small file size)
        self.model.save_pretrained(path)
        # Save tokenizer so it matches the training processing
        self.tokenizer.save_pretrained(path)
        print("Model saved successfully.")

    def predict(self, texts: list) -> list:
        """Runs inference."""
        label_str = ", ".join(self.label_names)
    
        prompts = [
            f"Instruct: Classify the emotion of the following movie review into one of these categories: {label_str}.\n"
            f"Input: {text}\n"
            f"Output:" 
            for text in texts
        ]
        
        inputs = self.tokenizer(
            prompts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy().tolist()