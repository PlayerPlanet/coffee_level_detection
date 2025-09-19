import os
import json

MANUAL_LEVELS_DIR = 'manual_levels'
OUTPUT_FILE = 'compiled_coffee_level_annotations.json'

annotations = []

for fname in os.listdir(MANUAL_LEVELS_DIR):
    if fname.startswith('level_') and fname.endswith('.json'):
        fpath = os.path.join(MANUAL_LEVELS_DIR, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotations.append(data)

compiled = {'annotation_data': annotations}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(compiled, f, indent=2)

print(f"✅ Compiled {len(annotations)} annotations into {OUTPUT_FILE}")