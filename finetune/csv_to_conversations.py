#!/usr/bin/env python3
"""
Convert SpookyBench metadata CSV to the conversation-style JSON format.

$ python csv_to_conversations.py spookybench_metadata.csv \
    -o spookybench_conversations.json
"""
import csv
import json
import ast
import os
import argparse

# Human prompts keyed by the Category column (case-insensitive)
PROMPTS = {
    "words": (
        "This video contains text encoded through temporal patterns. "
        "What specific word or phrase do you see? The text is only visible "
        "through the temporal changes. Please respond with just the text you "
        "identify."
    ),
    "shapes": (
        "This video contains a geometric shape encoded through temporal "
        "patterns. What shape emerges from the temporal changes? Please "
        "respond with just the shape name (e.g., circle, square, triangle)."
    ),
    "objects": (
        "This video contains a common object encoded through temporal "
        "patterns. Individual frames may appear as noise, but an object is "
        "visible through the temporal changes. What object do you see? "
        "Please respond with just the object name."
    ),
    "dynamics": (
        "This video contains movement or action encoded through temporal "
        "patterns. The content is only visible through temporal changes, "
        "not in individual frames. What is being shown? Please respond with "
        "just 1-3 words describing what you see."
    ),
}


def make_human_prompt(category: str) -> str:
    """Return the correct <video> prompt string for a category."""
    return f"<video>\n{PROMPTS.get(category, 'What is going on in this video?')}"


def main(in_csv: str, out_json: str) -> None:
    dataset = []
    with open(in_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Basic fields
            path = row["Path"].strip()
            category = row["Category"].strip().lower()
            vid_id = os.path.splitext(os.path.basename(path))[0]

            # Parse the label column safely (it’s stored like "['label']")
            try:
                label_list = ast.literal_eval(row["Label"])
                # Flatten in case of multiple labels
                label = " ".join(map(str, label_list))
            except Exception:
                # Fallback to the raw string if parsing fails
                label = row["Label"].strip(" []'\"")

            # Assemble the JSON entry
            dataset.append(
                {
                    "id": vid_id,
                    "video": path,
                    "conversations": [
                        {"from": "human", "value": make_human_prompt(category)},
                        {"from": "gpt", "value": label},
                    ],
                }
            )

    with open(out_json, "w") as fh:
        json.dump(dataset, fh, indent=2)
    print(f"Wrote {len(dataset)} items → {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV ➜ conversation-style JSON converter"
    )
    parser.add_argument("csv_file", help="Input CSV file (spookybench_metadata.csv)")
    parser.add_argument(
        "-o",
        "--output",
        default="spookybench_conversations.json",
        help="Output JSON filename (default: spookybench_conversations.json)",
    )
    args = parser.parse_args()
    main(args.csv_file, args.output)
