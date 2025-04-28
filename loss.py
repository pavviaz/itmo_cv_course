import torch
import torch.nn as nn


def get_bb_corners(bboxes_coords: torch.Tensor) -> torch.Tensor:
    x_center = bboxes_coords[..., 0]
    y_center = bboxes_coords[..., 1]
    width = bboxes_coords[..., 2]
    height = bboxes_coords[..., 3]

    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2

    bb_corners = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    return bb_corners


def iou(bboxes1_coords: torch.Tensor, bboxes2_coords: torch.Tensor) -> torch.Tensor:
    xmin_inter = torch.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin_inter = torch.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])
    xmax_inter = torch.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax_inter = torch.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])

    area_bb1 = (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) * (
        bboxes1_coords[..., 3] - bboxes1_coords[..., 1]
    )
    area_bb2 = (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) * (
        bboxes2_coords[..., 3] - bboxes2_coords[..., 1]
    )

    intersection_width = (xmax_inter - xmin_inter).clamp(min=0)
    intersection_height = (ymax_inter - ymin_inter).clamp(min=0)
    intersection = intersection_width * intersection_height

    union = area_bb1 + area_bb2 - intersection

    iou_score = intersection / (union + 1e-6)
    return iou_score


class YOLO_Loss(nn.Module):
    def __init__(
        self, S: int, B: int, C: int, L_coord: float = 5.0, L_noobj: float = 0.5
    ):
        super(YOLO_Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.L_coord = L_coord
        self.L_noobj = L_noobj

        indices = []
        for i in range(B):
            start_idx = self.C + i * 5
            indices.append(
                [start_idx, start_idx + 1, start_idx + 2, start_idx + 3, start_idx + 4]
            )
        self.register_buffer("pred_bb_ind", torch.tensor(indices, dtype=torch.long))

    def forward(self, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        N = y_pred.shape[0]

        exists_obj_i = y_gt[..., 0:1]
        gt_class = y_gt[..., 1 : 1 + self.C]
        gt_box_coords = y_gt[..., None, 1 + self.C :]

        pred_class = y_pred[..., : self.C]
        pred_box_conf = y_pred[..., self.pred_bb_ind[:, 0]]
        pred_box_coords = y_pred[..., self.pred_bb_ind[:, 1:]]

        cell_y_offset = (
            torch.arange(self.S, device=y_pred.device, dtype=torch.float32)
            .repeat(N, self.S, 1)
            .unsqueeze(-1)
        )
        cell_x_offset = cell_y_offset.permute(0, 2, 1, 3)

        gt_box_abs = gt_box_coords.clone().detach()
        gt_box_abs[..., 0] = (gt_box_abs[..., 0] + cell_x_offset) / self.S
        gt_box_abs[..., 1] = (gt_box_abs[..., 1] + cell_y_offset) / self.S

        pred_box_abs = pred_box_coords.clone().detach()

        pred_box_abs[..., 0] = (pred_box_abs[..., 0] + cell_x_offset) / self.S

        pred_box_abs[..., 1] = (pred_box_abs[..., 1] + cell_y_offset) / self.S

        pred_box_abs[..., 2:4] = pred_box_abs[..., 2:4] ** 2

        gt_corners = get_bb_corners(gt_box_abs)
        pred_corners = get_bb_corners(pred_box_abs)

        iou_scores = iou(gt_corners, pred_corners)

        _, max_iou_index = torch.max(iou_scores, dim=-1, keepdim=True)

        is_best_box = torch.zeros_like(iou_scores, device=y_pred.device)
        is_best_box.scatter_(-1, max_iou_index, 1)

        exists_obj_ij = exists_obj_i * is_best_box

        loc_xy_loss = torch.sum(
            exists_obj_ij[..., None]
            * ((gt_box_coords[..., :2] - pred_box_coords[..., :2]) ** 2)
        )
        gt_box_sqrt_wh = torch.sqrt(gt_box_coords[..., 2:4].clamp(min=1e-6))
        loc_wh_loss = torch.sum(
            exists_obj_ij[..., None]
            * ((gt_box_sqrt_wh - pred_box_coords[..., 2:4]) ** 2)
        )
        localization_loss = self.L_coord * (loc_xy_loss + loc_wh_loss)

        objectness_obj_loss = torch.sum(
            exists_obj_ij * ((iou_scores.detach() - pred_box_conf) ** 2)
        )

        noobj_mask = 1 - exists_obj_i
        not_responsible_mask = exists_obj_i * (1 - is_best_box)

        full_noobj_mask = noobj_mask + not_responsible_mask
        objectness_noobj_loss = self.L_noobj * torch.sum(
            full_noobj_mask * (pred_box_conf**2)
        )
        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        classification_loss = torch.sum(
            exists_obj_i
            * torch.sum(((gt_class - pred_class) ** 2), dim=-1, keepdim=True)
        )

        total_loss = (localization_loss + objectness_loss + classification_loss) / N
        return total_loss
