import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def load_sam(model_type, checkpoint, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)

    return sam


def get_transform():
    return ResizeLongestSide(1024)


def preprocess_image(image, sam, transform, device):
    original_size = image.shape[:2]
    image = transform.apply_image(image)
    image = torch.as_tensor(image).permute(2,0,1).float()
    image = image.unsqueeze(0).to(device)
    image = sam.preprocess(image)
    return image, original_size