import torch
import torch as th
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import logging
from typing import Dict, Any, Optional

from model import YOLOv1
import utils
from dataset import PigDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_model_from_checkpoint(
    checkpoint_path: str, device: str
) -> tuple[Optional[YOLOv1], Optional[Dict[str, Any]]]:
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        return None, None
    try:
        map_location = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        logging.info(f"Checkpoint loaded: {checkpoint_path}")

        if "config" not in checkpoint:
            logging.error("No config found while checkpoint loading")
            return None, None
        config = checkpoint["config"]

        model_config = config["model"]
        model = YOLOv1(
            S=model_config["S"],
            B=model_config["B"],
            C=model_config["C"],
            backbone_type=model_config["backbone"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logging.info("Model successfully created and loaded")

        return model, config

    except Exception as e:
        logging.error(f"Error while loading model: {e}")
        return None, None


def draw_boxes(image: Image.Image, boxes: th.Tensor, img_size: int) -> Image.Image:
    draw = ImageDraw.Draw(image)
    orig_width, orig_height = image.size
    font_size = max(15, int(0.03 * min(orig_width, orig_height)))
    font = ImageFont.load_default()

    w_scale = orig_width / img_size
    h_scale = orig_height / img_size

    for box in boxes:
        class_idx = int(box[0])
        confidence = float(box[1])
        xmin, ymin, xmax, ymax = box[2:].tolist()

        xmin_orig = xmin * w_scale
        ymin_orig = ymin * h_scale
        xmax_orig = xmax * w_scale
        ymax_orig = ymax * h_scale

        label = PigDataset.index2label[class_idx]
        color = "#ffffff"

        draw.rectangle(
            [(xmin_orig, ymin_orig), (xmax_orig, ymax_orig)],
            outline=color,
            width=max(3, int(0.005 * min(orig_width, orig_height))),
        )

        text = f"{label}: {confidence:.2f}"
        text_bbox = draw.textbbox(
            (xmin_orig, ymin_orig - font_size - 2), text, font=font
        )
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            (
                xmin_orig,
                ymin_orig - text_height - 2,
                xmin_orig + text_width + 2,
                ymin_orig,
            ),
            fill=color,
        )

        draw.text(
            (xmin_orig + 1, ymin_orig - text_height - 1), text, fill="black", font=font
        )

    return image


def run_inference(
    model_path: str,
    image_path: str,
    output_path: str,
    conf_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    model, config = load_model_from_checkpoint(model_path, device)
    if model is None or config is None:
        return

    try:
        img_size = config["dataset"]["image_size"]
        normalize_mean = config["dataset"]["normalize_mean"]
        normalize_std = config["dataset"]["normalize_std"]

        conf_thresh_default = config["evaluation"]["conf_threshold"]
        nms_thresh_default = config["evaluation"]["nms_iou_thresh"]
    except KeyError as e:
        logging.error(f"No key in: {e}")
        return

    conf_thresh = conf_threshold if conf_threshold is not None else conf_thresh_default
    nms_thresh = nms_threshold if nms_threshold is not None else nms_thresh_default
    logging.info(f"Using Confidence Threshold: {conf_thresh}")
    logging.info(f"Using NMS Threshold: {nms_thresh}")

    transforms = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    try:
        img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logging.error(f"File not found: {image_path}")
        return
    except Exception as e:
        logging.error(f"Error while loading {image_path}: {e}")
        return

    img_tensor = transforms(img_pil).unsqueeze(0).to(device)

    logging.info("Inference processing...")
    with torch.no_grad():
        predictions = model(img_tensor)

    logging.info("Post-processing...")
    final_boxes = utils.postprocessing(
        predictions,
        prob_threshold=conf_thresh,
        conf_mode="class",
        nms_threshold=nms_thresh,
        S=config["model"]["S"],
        B=config["model"]["B"],
        C=config["model"]["C"],
        img_dim=img_size,
        device=device,
    )

    logging.info(f"Bboxes found: {final_boxes.shape[0]}")

    final_boxes = final_boxes.cpu()

    logging.info("Dwaring bboxes...")
    img_with_boxes = draw_boxes(img_pil.copy(), final_boxes, img_size)

    try:

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_with_boxes.save(output_path)
        logging.info(f"Bboxed image saved into: {output_path}")
    except Exception as e:
        logging.error(f"Error while saving file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO inference")
    parser.add_argument(
        "--model-path",
        "-m",
        required=True,
        type=str,
        help="Model ckpt (.pth.tar).",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        required=True,
        type=str,
        help="Input img.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        type=str,
        help="Bboxed img path",
    )
    parser.add_argument(
        "--conf-thresh",
        "-ct",
        type=float,
        default=None,
        help="confidence threshold",
    )
    parser.add_argument(
        "--nms-thresh",
        "-nt",
        type=float,
        default=None,
        help="Non-Maximum Suppression threshold",
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        image_path=args.image_path,
        output_path=args.output_path,
        conf_threshold=args.conf_thresh,
        nms_threshold=args.nms_thresh,
    )
