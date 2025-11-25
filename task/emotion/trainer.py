from modules.data_utils import DataManager
from modules.model_classical import ClassicalMLLabeler
from modules.model_llm import LLMLabeler
from sklearn.metrics import classification_report
import os

# --- CONFIGURATION ---
TEXT_FILE = "text_all_merged.json"
LABEL_FILE = "label_all_merged.json"

def main():
    # 1. Load and Merge Data
    print("--- Phase 1: Data Loading ---")
    try:
        dm = DataManager(TEXT_FILE, LABEL_FILE)
        train_df, test_df = dm.get_splits()
        max_label_id = max(dm.label_map.values())
        target_names = [dm.id_to_label.get(i, "Unknown") for i in range(max_label_id + 1)]
    except Exception as e:
        print(f"Critical Error: {e}")
        return

    # 2. Classical Model (SVM)
    print("\n--- Phase 2: Classical ML ---")
    ml_model = ClassicalMLLabeler()
    ml_model.train(train_df) 
    ml_model.evaluate(test_df, target_names)
    ml_model.save_model("./saved_classical_model")

    # 3. LLM (Qwen + LoRA)
    print("\n--- Phase 3: LLM (Qwen LoRA) ---")
    print("Preparing Balanced Data for LLM...")
    balanced_train_df = dm.balance_training_data(train_df)
    
    dl_model = LLMLabeler(num_labels=len(dm.label_map), label_names=target_names)
    dl_model.train(balanced_train_df, test_df)
    
    # SAVE THE MODEL
    dl_model.save_model("./saved_emotion_lora")
    
    # Evaluate
    print("\n--- Deep Learning Report ---")
    dl_preds = dl_model.predict(test_df['text'].tolist())
    
    unique_labels = sorted(test_df['label'].unique())
    filtered_names = [target_names[i] for i in unique_labels]
    
    print(classification_report(
        test_df['label'], 
        dl_preds, 
        labels=unique_labels, 
        target_names=filtered_names, 
        zero_division=0
    ))

if __name__ == "__main__":
    main()