import albumentations as A
import cv2

def get_transforms(enable=True):

    transforms = []

    if enable:
        transforms.extend([

            A.HueSaturationValue(
                hue_shift_limit=3,
                sat_shift_limit=178,
                val_shift_limit=102,
                p=1.0
            ),

            A.Affine(
                translate_percent=0.1,
                shear=(-10, 10),
                p=1.0
            ),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),

            A.RandomBrightnessContrast(p=0.5),
     ])

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"]
        )
    )