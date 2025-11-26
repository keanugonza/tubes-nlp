from modules.data_utils import DataManager
from modules.model_classical import ClassicalMLLabeler
from modules.model_llm import LLMLabeler
import os

# --- CONFIGURATION ---
TEXT_FILE = "text_all_merged.json"
LABEL_FILE = "label_all_merged.json"

EXTERNAL_TEXT_FILE = "test_text.json"
EXTERNAL_LABEL_FILE = "test_label.json"

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
    
    dl_model.evaluate_with_trainer(test_df)

    # 4. EXTERNAL TEST SET EVALUATION
    print("\n--- Phase 4: External Test Set Evaluation ---")
    
    if os.path.exists(EXTERNAL_TEXT_FILE) and os.path.exists(EXTERNAL_LABEL_FILE):
        try:
            # Reuse DataManager logic to load the new files
            dm_ext = DataManager(EXTERNAL_TEXT_FILE, EXTERNAL_LABEL_FILE)
            external_df = dm_ext.load_data()
            
            if not external_df.empty:
                print(f"Successfully loaded {len(external_df)} external test samples.")
                
                # 1. Evaluate Classical Model
                print("\n>>> [External Test] Classical Model (SVM):")
                ml_model.evaluate(external_df, target_names)
                
                # 2. Evaluate LLM
                print("\n>>> [External Test] LLM (Qwen LoRA):")
                dl_model.evaluate_with_trainer(external_df)
            else:
                print("External dataset loaded but resulted in 0 valid samples (check IDs/Labels match).")
                
        except Exception as e:
            print(f"Error processing external data: {e}")
    else:
        print(f"External test files not found ({EXTERNAL_TEXT_FILE}, {EXTERNAL_LABEL_FILE}). Skipping Phase 4.")

if __name__ == "__main__":
    main()