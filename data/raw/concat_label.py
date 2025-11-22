import json
import glob

folder_path = "label/*.json"
files = glob.glob(folder_path)

all_results = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "results" in data:
            all_results.extend(data["results"])

output = {
    "results": all_results
}

with open("result/label_all_merged.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Selesai! Total data:", len(all_results))
print("File output: label_all_merged.json")
