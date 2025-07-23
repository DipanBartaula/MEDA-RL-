import json
from PIL import Image
import numpy as np

class CADDataset:
    def __init__(self, data_path):
        """
        Expects examples.json entries with:
          - prompt: str
          - target_image: path to PNG
          - expected_extents: [dx, dy, dz]
        """
        with open(data_path, 'r') as f:
            self.examples = json.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex['prompt']
        img = Image.open(ex['target_image']).convert('L')
        target_img = np.array(img) / 255.0
        expected_extents = tuple(ex['expected_extents'])
        return prompt, target_img, expected_extents

if __name__ == "__main__":
    print("Ensure data/examples.json exists with prompt, target_image & expected_extents.")
