import json
import glob

# Folder file text
folder_path = "text/*.json"  # sesuaikan jika foldermu berbeda

all_reviews = []

# Loop semua file JSON
for file in glob.glob(folder_path):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

        # format setiap file: { "dataset":..., "description":..., "total_reviews":..., "reviews":[...] }
        if "reviews" in data:
            all_reviews.extend(data["reviews"])

# Buat file final
output = {
    "dataset": "IMDB Movie Reviews (Merged)",
    "description": "Gabungan seluruh file JSON dari folder text",
    "total_reviews": len(all_reviews),
    "reviews": all_reviews
}

# Simpan
with open("result/text_all_merged.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Selesai! Total reviews:", len(all_reviews))
print("Output: text_all_merged.json")
