import json
import pandas as pd
from pathlib import Path

# Correct path to the JSONL file
jsonl_path = Path(r"C:\Users\...\2025-05-07-06-14-12_oss_eval.jsonl")

# Dermatology-related keywords
keywords = [
    "Dermatology", "skin conditions", "hair disorders", "nail disorders",
    "dermatitis", "acne", "vitiligo", "alopecia", "lichen planus",
    "cosmetics", "eczema", "malodor", "hyperhidrosis", "bromhidrosis",
    "olfactory reference syndrome", "rosacea", "sunburn"
]
keywords_lower = [k.lower() for k in keywords]

matches = []
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        prompt_field = rec.get("prompt", [])

        # Make sure it's a list of messages with 'content'
        if isinstance(prompt_field, list):
            full_text = " ".join(m.get("content", "") for m in prompt_field)
        else:
            continue  # skip if it's not a list

        if any(kw in full_text.lower() for kw in keywords_lower):
            matches.append({
                "prompt_id": rec.get("prompt_id", ""),
                "prompt_text": full_text.replace("\n", " "),
                "rubrics": json.dumps(rec.get("rubrics", {}))
            })

# Convert to DataFrame
df = pd.DataFrame(matches)

# Preview
print(f"✅ Found {len(df)} matching conversations.")
display(df.head())

# Save as CSV
output_csv = jsonl_path.parent / "dermatology_subset.csv"
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"📁 Exported to: {output_csv}")

