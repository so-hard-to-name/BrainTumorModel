import pathlib
import pandas as pd
from PIL import Image
import csv
from sklearn.model_selection import train_test_split

def prepare_dataset_index(dir, task="train"):
    data = []
    classification_dir = pathlib.Path(dir) / "classification_task" / task
    segmentation_dir = pathlib.Path(dir) / "segmentation_task" / task

    seg_image_dir = segmentation_dir / "images"
    seg_mask_dir = segmentation_dir / "masks"

    for folder in classification_dir.iterdir():
        for item in folder.iterdir():
            if "_ax_" in item.name:
                direction = "axial"
            elif "_co_" in item.name:
                direction = "coronal"
            elif "_sa_" in item.name:
                direction = "sagittal"
            else:
                direction = "unknown"

            if folder.name != "no_tumor" and (seg_mask_dir / item.name.replace(".jpg", ".png")).exists():
                has_mask = True
            else:
                has_mask = False

            if folder.name != "no_tumor" and (seg_image_dir / item.name).exists():
                has_seg = True
            else:
                has_seg = False

            image_path = classification_dir / folder.name / item.name
            img_size = Image.open(image_path).size

            data.append({
                "task": task,
                "filename": item.name,
                "image_path": str(classification_dir / folder.name / item.name),
                "direction": direction,
                "tumor_type": str(folder.name),
                "has_seg": has_seg,
                "segmentation_image_path": str(seg_image_dir / item.name) if has_seg == True else "N/A",
                "has_mask": has_mask,
                "segmentation_mask_path": str(seg_mask_dir / item.name.replace(".jpg", ".png")) if has_mask == True else "N/A",
                "Image_size": img_size
            })

    df = pd.DataFrame(data)

    if task == "train":
        train_df, val_df = train_test_split(df, test_size=0.125, random_state=42, stratify=df["tumor_type"], shuffle=True)

        val_df["task"] = "val"
        df = pd.concat([train_df, val_df], ignore_index=True)

    out_csv = (f'{task}_dataset_index.csv')
    df.to_csv(out_csv, sep=";", index=False)
    return df