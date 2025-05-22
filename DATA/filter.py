import argparse
import json

def filter_conversations(input_path, output_path, keywords):
    """
    Reads a JSONL file of conversations, filters for those containing any of the specified keywords,
    and writes the matching records to a new JSONL file.
    """
    # Normalize keywords for case-insensitive matching
    keywords_lower = [kw.lower() for kw in keywords]
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            record = json.loads(line)
            
            # Extract all message contents from the conversation
            msgs = record.get('conversation', [])
            text_content = " ".join(msg.get('content', '') for msg in msgs).lower()
            
            # Check if any keyword is present
            if any(kw in text_content for kw in keywords_lower):
                outfile.write(json.dumps(record) + '\n')

def main():
    parser = argparse.ArgumentParser(
        description="Filter HealthBench OSS JSONL for dermatology-related conversations"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input OSS JSONL file"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to write the filtered JSONL output"
    )
    args = parser.parse_args()

    keywords = [
        "Dermatology", "skin conditions", "hair disorders", "nail disorders",
        "dermatitis", "acne", "vitiligo", "alopecia", "lichen planus",
        "cosmetics", "eczema", "malodor", "hyperhidrosis", "bromhidrosis",
        "olfactory reference syndrome", "rosacea", "sunburn"
    ]

    filter_conversations(args.input, args.output, keywords)
    print(f"Filtered conversations written to {args.output}")

if __name__ == "__main__":
    main()

