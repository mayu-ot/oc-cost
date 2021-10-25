import cv2
import albumentations as A
import os
from tqdm import tqdm


def compile_coco_corrupt():
    transform = A.Compose(
        # [A.GaussNoise(always_apply=True, var_limit=(10.0, 50.0))]
        [
            A.ImageCompression(
                always_apply=True,
                quality_lower=80,
                quality_upper=95,
                compression_type=0,
            )
        ]
    )
    for fn in tqdm(os.listdir("data/coco/val2017")):
        if fn.endswith(".jpg"):
            image = cv2.imread(f"data/coco/val2017/{fn}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            transformed_image = cv2.cvtColor(
                transformed_image, cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(
                f"data/processed/coco-corrupted/val2017/ImageCompression/{fn}",
                transformed_image,
            )


if __name__ == "__main__":
    compile_coco_corrupt()
