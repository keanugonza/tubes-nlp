import json
import pandas as pd

TEXT_PATH = "data/text_all_merged.json"
LABEL_PATH = "data/label_all_merged.json"

def load_texts(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["id"]: r["text"] for r in data["reviews"]}

texts = load_texts(TEXT_PATH)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_data = json.load(f)

rows = []
for item in label_data["results"]:
    review_id = item["id"]
    review_text = texts.get(review_id)
    if review_text is None:
        continue

    aspects = item.get("absa", {}).get("aspects", [])
    for asp in aspects:
        rows.append(
            {
                "id": review_id,
                "text": review_text,
                "aspect": asp["aspect"],
                "sentimen": asp["sentiment"],
            }
        )

df = pd.DataFrame(rows, columns=["id", "text", "aspect", "sentimen"])
print(df.head())
df.to_csv("absa_flat.csv", index=False)
