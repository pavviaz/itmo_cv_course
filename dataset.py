import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
from typing import Callable, Optional, Tuple, Union


class PigDataset(Dataset):
    C = 2
    index2label = ["pig", "person"]
    label2index = {"pig": 0, "person": 1}

    def __init__(
        self, img_dir: str, annot_dir: str, transforms: Optional[Callable] = None
    ) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transforms = transforms

        self.annot_files = sorted(
            [
                os.path.join(annot_dir, f)
                for f in os.listdir(annot_dir)
                if f.endswith(".txt")
            ]
        )

        if not self.annot_files:
            raise FileNotFoundError(f"No file: {annot_dir}")

    def __len__(self) -> int:
        return len(self.annot_files)

    def __getitem__(self, idx: int) -> Tuple[Union[th.Tensor, Image.Image], th.Tensor]:
        annot_path = self.annot_files[idx]
        base_filename = os.path.splitext(os.path.basename(annot_path))[0]

        possible_extensions = [".jpg", ".png", ".jpeg", ".bmp", ".webp"]
        img_path = None
        for ext in possible_extensions:
            potential_path = os.path.join(self.img_dir, base_filename + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None:
            raise FileNotFoundError(f"No img: {img_path}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error with {img_path}: {e}")
            raise e

        target = []
        try:
            with open(annot_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        target.append([class_id, cx, cy, w, h])
                    else:
                        if line.strip():
                            print(f"Line ignored because of wrong format: {line}")

        except Exception as e:
            print(f"Error reading file {annot_path}: {e}")
            raise e

        if target:
            target_tensor = th.tensor(target, dtype=th.float32)
        else:
            target_tensor = th.zeros((0, 5), dtype=th.float32)

        if self.transforms is not None:
            img, target_tensor = self.transforms((img, target_tensor))

        return img, target_tensor
