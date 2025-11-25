import json
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional

class DataManager:
    def __init__(self, text_json_path: str, label_json_path: str):
        self.text_json_path = text_json_path
        self.label_json_path = label_json_path
        
        self.label_map = {
            "disgust": 0,
            "anger": 1,
            "sadness": 2,
            "joy": 3,
            "neutral": 4,
            "fear": 5,
            "surprise": 6
        }
        self.id_to_label = {v: k for k, v in self.label_map.items()}

    def load_data(self) -> pd.DataFrame:
        """Merges text and label JSONs on 'id'."""
        
        # 1. Load Text
        try:
            with open(self.text_json_path, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
                reviews = text_data.get('reviews', [])
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find text file: {self.text_json_path}")

        text_lookup = {item['id']: item['text'] for item in reviews if 'id' in item}

        # 2. Load Labels
        try:
            with open(self.label_json_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                results = label_data.get('results', [])
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {self.label_json_path}")

        # 3. Merge
        merged_data = []
        for item in results:
            item_id = item.get('id')
            label_str = item.get('emotion', {}).get('emotion')
            text_content = text_lookup.get(item_id)

            if item_id and label_str and text_content:
                merged_data.append({
                    'id': item_id,
                    'text': text_content,
                    'label_str': label_str.lower()
                })

        df = pd.DataFrame(merged_data)
        df = df[df['label_str'].isin(self.label_map.keys())].copy()
        df['label'] = df['label_str'].map(self.label_map).astype(int)
        
        print(f"Loaded {len(df)} valid samples.")
        return df

    def balance_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Upsamples minority classes to match majority class count."""
        if df.empty: return df
        
        max_count = df['label'].value_counts().max()
        balanced_dfs = []
        
        for label_id in df['label'].unique():
            class_subset = df[df['label'] == label_id]
            balanced_dfs.append(class_subset) # Add original
            
            if len(class_subset) < max_count:
                diff = max_count - len(class_subset)
                upsampled = class_subset.sample(n=diff, replace=True, random_state=42)
                balanced_dfs.append(upsampled)
        
        balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Data Balancing: Increased size from {len(df)} to {len(balanced_df)} samples.")
        return balanced_df

    def get_splits(self, test_size=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns stratified Train and Test sets, handling rare classes."""
        df = self.load_data()
        
        label_counts = df['label'].value_counts()
        rare_labels = label_counts[label_counts < 2].index
        if len(rare_labels) > 0:
            rare_rows = df[df['label'].isin(rare_labels)]
            df = pd.concat([df, rare_rows], ignore_index=True)

        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['label'], random_state=42
        )
        return train_df, test_df