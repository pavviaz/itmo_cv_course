import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Literal, Dict, Any

from model import YOLOv1
from loss import get_bb_corners, iou


def rescale_bboxes(
    y: torch.Tensor, S: int, C: int, B: int, img_dim: int, device: str
) -> torch.Tensor:
    row, col = torch.meshgrid(
        torch.arange(S, device=device), torch.arange(S, device=device), indexing="ij"
    )
    col = col.reshape(1, S, S, 1).float()
    row = row.reshape(1, S, S, 1).float()

    x_indices = [C + i * 5 + 1 for i in range(B)]
    y_indices = [C + i * 5 + 2 for i in range(B)]
    wh_indices = [C + i * 5 + j for j in [3, 4] for i in range(B)]

    y[..., x_indices] = (y[..., x_indices] + col) * (img_dim / S)
    y[..., y_indices] = (y[..., y_indices] + row) * (img_dim / S)
    y[..., wh_indices] = (y[..., wh_indices] ** 2) * img_dim

    return y


def get_detected_boxes(
    y_rescaled: torch.Tensor,
    prob_threshold: float,
    conf_mode: Literal["objectness", "class"],
    C: int,
    B: int,
    img_dim: int,
    device: str,
) -> torch.Tensor:
    assert conf_mode in ["objectness", "class"]
    N = y_rescaled.shape[0]
    all_boxes = []

    y_rescaled_softmax = y_rescaled.clone()
    y_rescaled_softmax[..., :C] = F.softmax(y_rescaled_softmax[..., :C], dim=-1)

    for i in range(N):
        y_img = y_rescaled_softmax[i]
        class_score, class_ind = torch.max(y_img[..., :C], dim=-1)
        objectness_scores = y_img[..., [C + b * 5 for b in range(B)]]
        best_objectness, best_box_ind = torch.max(objectness_scores, dim=-1)

        best_box_ind_exp = best_box_ind.unsqueeze(-1).unsqueeze(-1)
        idx_base = C + best_box_ind_exp * 5
        coord_indices = torch.cat(
            [idx_base + 1, idx_base + 2, idx_base + 3, idx_base + 4], dim=-1
        ).squeeze(-2)
        best_bboxes_coords = torch.gather(y_img, dim=-1, index=coord_indices)

        detection_mask = best_objectness > prob_threshold
        valid_class_ind = class_ind[detection_mask]
        valid_objectness = best_objectness[detection_mask]
        valid_class_score = class_score[detection_mask]
        valid_coords = best_bboxes_coords[detection_mask]

        if valid_coords.numel() == 0:
            continue

        if conf_mode == "class":
            valid_conf = valid_class_score * valid_objectness
        else:
            valid_conf = valid_objectness

        valid_corners = get_bb_corners(valid_coords)
        valid_corners = valid_corners.clamp(min=0, max=img_dim)

        img_boxes = torch.cat(
            [valid_class_ind.unsqueeze(-1), valid_conf.unsqueeze(-1), valid_corners],
            dim=-1,
        )
        all_boxes.append(img_boxes)

    if not all_boxes:
        return torch.empty((0, 6), device=device, dtype=torch.float32)
    else:
        return all_boxes[0]


def non_max_suppression(
    boxes: torch.Tensor, nms_threshold: float, device: str
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes

    nms_boxes = []
    sort_ind = torch.argsort(boxes[:, 1], descending=True)
    boxes = boxes[sort_ind, :]

    while boxes.shape[0] > 0:
        box1 = boxes[0]
        nms_boxes.append(box1)
        boxes = boxes[1:]
        if boxes.shape[0] == 0:
            break

        box1_class = box1[0]
        box1_coords = box1[2:]
        same_class_mask = box1_class == boxes[:, 0]
        iou_scores = torch.zeros(boxes.shape[0], device=device)

        if torch.any(same_class_mask):
            iou_scores[same_class_mask] = iou(
                box1_coords.unsqueeze(0), boxes[same_class_mask, 2:]
            )

        valid_boxes_mask = iou_scores < nms_threshold
        boxes = boxes[valid_boxes_mask]

    if not nms_boxes:
        return torch.empty((0, 6), device=device, dtype=boxes.dtype)
    else:
        return torch.stack(nms_boxes, dim=0)


def postprocessing(
    y: torch.Tensor,
    prob_threshold: float,
    conf_mode: Literal["objectness", "class"],
    nms_threshold: float,
    S: int,
    C: int,
    B: int,
    img_dim: int,
    device: str,
) -> torch.Tensor:
    y_rescaled = rescale_bboxes(y.clone(), S, C, B, img_dim, device)
    detected_boxes = get_detected_boxes(
        y_rescaled, prob_threshold, conf_mode, C, B, img_dim, device
    )
    final_boxes = non_max_suppression(detected_boxes, nms_threshold, device)
    return final_boxes


def evaluate_predictions(
    bboxes_gt: torch.Tensor,
    bboxes_pred: torch.Tensor,
    iou_threshold: float,
    device: str,
) -> torch.Tensor:
    n_pred = bboxes_pred.shape[0]
    true_positives = torch.zeros(n_pred, dtype=torch.bool, device=device)
    gt_detected_mask = torch.zeros(bboxes_gt.shape[0], dtype=torch.bool, device=device)

    for p_idx, p_box in enumerate(bboxes_pred):
        p_class = p_box[0].long()
        p_coords = p_box[2:]
        best_iou = -1.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(bboxes_gt):
            if gt_detected_mask[gt_idx]:
                continue
            gt_class = gt_box[0].long()
            gt_coords = gt_box[1:]

            if p_class == gt_class:
                current_iou = iou(p_coords.unsqueeze(0), gt_coords.unsqueeze(0)).item()
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

        if best_iou > iou_threshold and best_gt_idx != -1:
            if not gt_detected_mask[best_gt_idx]:
                true_positives[p_idx] = True
                gt_detected_mask[best_gt_idx] = True

    return true_positives.float()


def calculate_map(
    all_predictions: List[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    iou_threshold: float,
    device: str,
) -> Tuple[float, List[float]]:
    predictions_list = []
    total_gt_boxes_per_class = torch.zeros(num_classes, device=device)

    for pred_boxes, gt_boxes in all_predictions:
        if gt_boxes.numel() > 0:
            gt_classes = gt_boxes[:, 0].long()
            total_gt_boxes_per_class += torch.bincount(
                gt_classes, minlength=num_classes
            )
        if pred_boxes.numel() == 0:
            continue
        is_tp = evaluate_predictions(gt_boxes, pred_boxes, iou_threshold, device)
        pred_classes = pred_boxes[:, 0]
        pred_confs = pred_boxes[:, 1]
        predictions_list.append(torch.stack([pred_classes, pred_confs, is_tp], dim=-1))

    if not predictions_list:
        return 0.0, [0.0] * num_classes

    total_predictions_tensor = torch.cat(predictions_list, dim=0)
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        class_preds = total_predictions_tensor[total_predictions_tensor[:, 0] == c]
        if class_preds.numel() == 0:
            ap = 0.0
            average_precisions.append(ap)
            continue

        class_preds = class_preds[torch.argsort(class_preds[:, 1], descending=True)]
        tp = class_preds[:, 2]
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(1 - tp, dim=0)

        recalls = tp_cumsum / (total_gt_boxes_per_class[c] + epsilon)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)

        recalls = torch.cat((torch.tensor([0.0], device=device), recalls))
        precisions = torch.cat(
            (
                torch.tensor(
                    [precisions[0] if precisions.numel() > 0 else 0.0], device=device
                ),
                precisions,
            )
        )

        for i in range(precisions.shape[0] - 1, 0, -1):
            precisions[i - 1] = torch.maximum(precisions[i - 1], precisions[i])

        recall_change_indices = torch.where(recalls[1:] != recalls[:-1])[0]
        ap = torch.sum(
            (recalls[recall_change_indices + 1] - recalls[recall_change_indices])
            * precisions[recall_change_indices + 1]
        )
        average_precisions.append(ap.item() * 100)

    valid_aps = [ap for ap in average_precisions]
    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
    return mAP, average_precisions


def evaluate_model(
    model: YOLOv1, test_loader: DataLoader, config: Dict[str, Any], device: str
) -> Tuple[float, List[float]]:

    model_config = config["model"]
    eval_config = config["evaluation"]
    dataset_config = config["dataset"]

    S = model_config["S"]
    B = model_config["B"]
    C = model_config["C"]
    img_dim = dataset_config["image_size"]
    num_classes = C
    iou_threshold = eval_config["map_iou_thresh"]
    prob_threshold = eval_config["conf_threshold"]
    nms_threshold = eval_config["nms_iou_thresh"]
    conf_mode = eval_config.get("conf_mode", "class")

    model.eval()
    all_img_predictions = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_loader):
            x = x.to(device)
            target = target.squeeze(0).to(device)

            if target.numel() > 0:
                gt_classes = target[:, 0:1]
                gt_yolo_coords = target[:, 1:]
                gt_corner_coords = get_bb_corners(gt_yolo_coords)
                gt_corner_coords_abs = gt_corner_coords * img_dim
                bboxes_gt_corners = torch.cat(
                    [gt_classes, gt_corner_coords_abs], dim=-1
                )
            else:
                bboxes_gt_corners = torch.empty(
                    (0, 5), device=device, dtype=torch.float32
                )

            y_pred = model(x)
            bboxes_pred = postprocessing(
                y_pred,
                prob_threshold=prob_threshold,
                conf_mode=conf_mode,
                nms_threshold=nms_threshold,
                S=S,
                C=C,
                B=B,
                img_dim=img_dim,
                device=device,
            )
            all_img_predictions.append(
                (bboxes_pred.to(device), bboxes_gt_corners.to(device))
            )

    mAP, average_precisions = calculate_map(
        all_img_predictions, num_classes, iou_threshold, device
    )
    return mAP, average_precisions
