import random, json, os, cv2, datetime
from coffee_level_detection.dataset_collection.models import ZeroCoffeeLevelRelabeler

def relabel_zeros_main():
        """
        Main entry point for relabeling zero coffee_level images as None or 0.
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_dir', type=str, default='processed_images')
        parser.add_argument('--ann_dir', type=str, default='manual_levels')
        parser.add_argument('--samples', type=int, default=20)
        args = parser.parse_args()

        img_dir = args.img_dir
        ann_dir = args.ann_dir
        ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        zero_files = []
        for f in ann_files:
            with open(os.path.join(ann_dir, f), 'r', encoding='utf-8') as jf:
                ann = json.load(jf)
                if ann.get('coffee_level', None) == 0:
                    zero_files.append(f)
        print(f"Found {len(zero_files)} zero-labeled annotation files.")
        random.shuffle(zero_files)
        zero_files = zero_files[:args.samples]

        for ann_file in zero_files:
            with open(os.path.join(ann_dir, ann_file), 'r', encoding='utf-8') as jf:
                ann = json.load(jf)
            img_path = os.path.join(img_dir, ann['filename'])
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}, skipping.")
                continue
            img = cv2.imread(img_path)
            relabeler = ZeroCoffeeLevelRelabeler(img, ann['filename'])
            new_label = relabeler.show()
            if new_label is None and not relabeler.selected:
                print(f"Skipped {ann['filename']}.")
                continue
            # Save new annotation
            ann['coffee_level'] = new_label
            ann['timestamp'] = __import__('datetime').datetime.now().isoformat()
            ann['annotator'] = 'zero_relabeler'
            out_path = os.path.join(ann_dir, ann_file)
            with open(out_path, 'w', encoding='utf-8') as jf:
                json.dump(ann, jf, indent=2)
            print(f"Relabeled {ann['filename']} as {new_label}.")


if __name__ == "__main__":
    relabel_zeros_main()