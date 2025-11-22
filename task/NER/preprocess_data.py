import json
import pandas as pd
from collections import defaultdict

def load_data(text_path, label_path):
    """Load text and label JSON files"""
    with open(text_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)

    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    return text_data, label_data

def convert_to_token_classification(text_data, label_data):
    """
    Convert the NER data to token classification format.
    Returns a list of dictionaries with 'id', 'text', and 'entities' fields.
    """
    processed_data = []

    # Create a mapping from id to text
    id_to_text = {review['id']: review['text'] for review in text_data['reviews']}

    # Process each result
    for result in label_data['results']:
        review_id = result['id']

        # Get the corresponding text
        if review_id not in id_to_text:
            continue

        text = id_to_text[review_id]

        # Get NER entities
        ner_data = result.get('ner', {})
        entities = ner_data.get('entities', [])

        # Skip reviews with no entities
        if not entities:
            continue

        # Convert entities to the format with character positions
        formatted_entities = []
        for entity in entities:
            entity_text = entity['text']
            entity_label = entity['label']

            # Find all occurrences of the entity in the text
            start_idx = 0
            while True:
                pos = text.find(entity_text, start_idx)
                if pos == -1:
                    break

                formatted_entities.append({
                    'text': entity_text,
                    'label': entity_label,
                    'start': pos,
                    'end': pos + len(entity_text)
                })

                # For now, just take the first occurrence
                break

        if formatted_entities:
            processed_data.append({
                'id': review_id,
                'text': text,
                'entities': formatted_entities
            })

    return processed_data

def create_bio_tags(text, entities):
    """
    Create BIO tags for the text based on entities.
    Returns tokens and their corresponding BIO tags.
    """
    # Simple whitespace tokenization
    tokens = text.split()
    labels = ['O'] * len(tokens)

    # Track character position for each token
    char_to_token = {}
    current_pos = 0
    for token_idx, token in enumerate(tokens):
        token_start = text.find(token, current_pos)
        token_end = token_start + len(token)
        for char_idx in range(token_start, token_end):
            char_to_token[char_idx] = token_idx
        current_pos = token_end

    # Assign BIO tags
    for entity in entities:
        entity_start = entity['start']
        entity_end = entity['end']
        entity_label = entity['label']

        # Find tokens that overlap with this entity
        entity_tokens = set()
        for char_idx in range(entity_start, entity_end):
            if char_idx in char_to_token:
                entity_tokens.add(char_to_token[char_idx])

        # Assign B- and I- tags
        entity_tokens = sorted(entity_tokens)
        for i, token_idx in enumerate(entity_tokens):
            if i == 0:
                labels[token_idx] = f'B-{entity_label}'
            else:
                labels[token_idx] = f'I-{entity_label}'

    return tokens, labels

def create_ner_dataset(text_path, label_path, output_path):
    """
    Create NER dataset in CSV format for training.
    Format: id, tokens, ner_tags (space-separated)
    """
    print("Loading data...")
    text_data, label_data = load_data(text_path, label_path)

    print("Converting to token classification format...")
    processed_data = convert_to_token_classification(text_data, label_data)

    print(f"Found {len(processed_data)} reviews with NER annotations")

    # Create rows for CSV
    rows = []
    for item in processed_data:
        tokens, labels = create_bio_tags(item['text'], item['entities'])

        rows.append({
            'id': item['id'],
            'tokens': ' '.join(tokens),
            'ner_tags': ' '.join(labels)
        })

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(df)}")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    all_tags = []
    for tags in df['ner_tags']:
        all_tags.extend(tags.split())

    tag_counts = defaultdict(int)
    for tag in all_tags:
        tag_counts[tag] += 1

    print(f"Total tokens: {len(all_tags)}")
    print("\nTag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_tags)) * 100
        print(f"  {tag}: {count} ({percentage:.2f}%)")

    return df

if __name__ == "__main__":
    TEXT_PATH = "../../data/text_all_merged.json"
    LABEL_PATH = "../../data/label_all_merged.json"
    OUTPUT_PATH = "train_ner.csv"

    df = create_ner_dataset(TEXT_PATH, LABEL_PATH, OUTPUT_PATH)
