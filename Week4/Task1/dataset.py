import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

# Dataset paths
DEFAULT_DATASET_ROOT = os.environ.get('VIZWIZ_ROOT', '/ghome/group07/MCV-C5-Group7/Week3/dataset')


def get_dataset_paths(dataset_root: str | None = None) -> dict[str, str]:
    root = dataset_root or DEFAULT_DATASET_ROOT
    return {
        'train_img_path': os.path.join(root, 'train'),
        'val_img_path':   os.path.join(root, 'val'),
        'train_json':     os.path.join(root, 'annotations', 'train.json'),
        'val_json':       os.path.join(root, 'annotations', 'val.json'),
    }


def load_vizwiz_split(json_path: str) -> list[dict]:
    """Load a VizWiz split and return a list of {image, captions} dicts."""
    with open(json_path) as f:
        raw = json.load(f)

    id_to_filename = {img['id']: img['file_name'] for img in raw['images']}

    grouped: dict[int, dict] = {}
    for ann in raw['annotations']:
        if ann.get('is_rejected') or ann.get('is_precanned'):
            continue
        caption = ann.get('caption', '').replace('\r', ' ').replace('\n', ' ').strip()
        if not caption:
            continue
        iid = ann['image_id']
        if iid not in grouped:
            grouped[iid] = {'image': id_to_filename[iid], 'captions': []}
        grouped[iid]['captions'].append(caption)

    return [item for item in grouped.values() if item['captions']]


class VizWizVisionDataset(Dataset):
    """Dataset for HuggingFace image-captioning models.

    Returns (PIL_image, caption_str, image_name, all_captions_list).
    """

    def __init__(self, samples: list[dict], img_path: str):
        self.samples  = samples
        self.img_path = img_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item    = self.samples[idx]
        img     = Image.open(os.path.join(self.img_path, item['image'])).convert('RGB')
        caption = random.choice(item['captions'])
        return img, caption, item['image'], item['captions']


class VisionEncoderDecoderCollator:
    """Collate raw PIL images and caption strings into model-ready tensors.

    Uses the HuggingFace image_processor and GPT-2 BPE tokenizer directly —
    no custom vocabulary involved.
    """

    def __init__(self, image_processor, tokenizer, max_target_length: int = 64):
        self.image_processor   = image_processor
        self.tokenizer         = tokenizer
        self.max_target_length = max_target_length

    def __call__(self, batch):
        images, captions, img_names, all_captions = zip(*batch)
        pixel_values = self.image_processor(
            images=list(images), return_tensors='pt'
        ).pixel_values
        tokens = self.tokenizer(
            list(captions),
            padding='max_length',
            truncation=True,
            max_length=self.max_target_length,
            return_tensors='pt',
        )
        labels = tokens.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'pixel_values': pixel_values,
            'labels':       labels,
            'image_names':  list(img_names),
            'references':   [list(caps) for caps in all_captions],
        }
