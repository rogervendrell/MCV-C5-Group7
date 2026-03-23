import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2

from vocabulary import char2idx, TEXT_MAX_LEN

# Dataset paths
_BASE = '/ghome/group07/MCV-C5-Group7/Week3/dataset'
TRAIN_IMG_PATH = os.path.join(_BASE, 'train')
VAL_IMG_PATH   = os.path.join(_BASE, 'val')
TRAIN_JSON     = os.path.join(_BASE, 'annotations', 'train.json')
VAL_JSON       = os.path.join(_BASE, 'annotations', 'val.json')


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


class VizWizDataset(Dataset):
    """VizWiz image-captioning dataset.

    Returns (image_tensor, caption_tensor, image_name, all_captions) where
    caption_tensor is one randomly chosen caption encoded as character indices.
    """

    def __init__(self, samples: list[dict], img_path: str, max_len: int = TEXT_MAX_LEN):
        self.samples  = samples
        self.img_path = img_path
        self.max_len  = max_len

        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def encode_caption(self, caption: str) -> torch.Tensor:
        tokens = ['<SOS>'] + list(caption.lower()) + ['<EOS>']
        tokens = tokens[:self.max_len]
        indices = [char2idx.get(ch, char2idx['<PAD>']) for ch in tokens]
        while len(indices) < self.max_len:
            indices.append(char2idx['<PAD>'])
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = Image.open(os.path.join(self.img_path, item['image'])).convert('RGB')
        img = self.img_proc(img)
        caption = random.choice(item['captions'])
        cap_tensor = self.encode_caption(caption)
        return img, cap_tensor, item['image'], item['captions']


def collate_fn(batch):
    """Custom collate that handles variable-length caption lists per image."""
    imgs, caps, img_names, all_captions = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(caps),
        list(img_names),
        list(all_captions),   # list of lists — keep as-is, not stacked
    )
