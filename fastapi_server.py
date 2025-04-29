from fastapi import FastAPI, WebSocket

import asyncio
import glob
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# from track_5_rr3_bb10 import track_data, country_balls_amount
# from track_5_rr5_bb25 import track_data, country_balls_amount
# from track_5_rr10_bb50 import track_data, country_balls_amount

# from track_10_rr3_bb10 import track_data, country_balls_amount
# from track_10_rr5_bb25 import track_data, country_balls_amount
# from track_10_rr10_bb50 import track_data, country_balls_amount

# from track_20_rr3_bb10 import track_data, country_balls_amount
# from track_20_rr5_bb25 import track_data, country_balls_amount
from track_20_rr10_bb50 import track_data, country_balls_amount


app = FastAPI(title="Tracker assignment")
imgs = glob.glob("imgs/*")

country_balls = [
    {"cb_id": x, "img": imgs[x % len(imgs)]} for x in range(country_balls_amount)
]
print("Started")


active_tracks = (
    {}
)  # current bboxes: {track_id: {'bbox': [x1,y1,x2,y2], 'age': 0, 'hits': 0, ...}}
next_track_id = 0
MAX_AGE = 5  # how much frames a track can live without detection

MAX_AGE_STRONG = 7
IOU_THRESHOLD_STRONG = 0.1

MAX_AGE_SOFT = 3
DISTANCE_THRESHOLD_SOFT = 100  # max distance between tracks


def get_bbox_center(bbox):
    if not bbox:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def calc_distance(center1, center2):
    if center1 is None or center2 is None:
        return float("inf")
    return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5


def calc_iou(bbox1, bbox2):
    if bbox1 is None or bbox2 is None or not bbox1.any() or not bbox2.any():
        return 0.0

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    box1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def bbox_to_center_wh(bbox):
    """Converts [x1, y1, x2, y2] bbox to [cx, cy, w, h]"""
    if not bbox or len(bbox) != 4:
        return None
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    return np.array([cx, cy, w, h])


def center_wh_to_bbox(center_wh):
    """Converts [cx, cy, w, h] to [x1, y1, x2, y2] bbox"""
    if center_wh is None or len(center_wh) < 4:
        return None
    cx, cy, w, h = center_wh[:4]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=int)


active_tracks_kalman = {}
next_track_id_kalman = 0


class KalmanTrack:
    def __init__(self, detection_bbox, track_id):
        self.id = track_id
        self.kf = KalmanFilter(dim_x=6, dim_z=4)

        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=float,
        )

        self.kf.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0]) * 10
        self.kf.R = np.diag([5.0, 5.0, 5.0, 5.0]) * 1
        self.kf.Q = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0]) * 0.1

        initial_state_z = bbox_to_center_wh(detection_bbox)
        self.kf.x = np.array(
            [
                initial_state_z[0],
                initial_state_z[1],
                initial_state_z[2],
                initial_state_z[3],
                0,
                0,
            ],
            dtype=float,
        )  # Initial velocity is 0

        self.bbox = detection_bbox
        self.predicted_bbox = detection_bbox
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()

        predicted_state_z = self.kf.x[:4]
        self.predicted_bbox = center_wh_to_bbox(predicted_state_z)
        self.age += 1
        self.time_since_update += 1
        return self.predicted_bbox

    def update(self, detection_bbox):
        measurement_z = bbox_to_center_wh(detection_bbox)
        if measurement_z is not None:
            self.kf.update(measurement_z)
            self.bbox = detection_bbox
            self.time_since_update = 0
            self.age = 0
            self.hits += 1
        else:
            print(
                f"Warning: Failed to convert detection bbox {detection_bbox} for track {self.id}"
            )


def reset_tracker_state():
    global active_tracks, next_track_id
    active_tracks = {}
    next_track_id = 0


def tracker_soft(frame_data):
    global active_tracks, next_track_id

    current_detections = frame_data["data"]

    # Filtering empty bounding_boxes and compute centers
    valid_detections = []
    for i, det in enumerate(current_detections):
        if det["bounding_box"]:
            center = get_bbox_center(det["bounding_box"])
            if center:
                valid_detections.append(
                    {"original_index": i, "bbox": det["bounding_box"], "center": center}
                )

    track_ids = list(active_tracks.keys())
    detection_indices = list(range(len(valid_detections)))

    # making cost matrix of all distances between tracks and detections
    cost_matrix = np.full((len(track_ids), len(detection_indices)), float("inf"))
    for t_idx, track_id in enumerate(track_ids):
        track_center = get_bbox_center(active_tracks[track_id]["bbox"])
        for d_idx, det in enumerate(valid_detections):
            dist = calc_distance(track_center, det["center"])
            if dist < DISTANCE_THRESHOLD_SOFT:
                cost_matrix[t_idx, d_idx] = dist
            else:
                cost_matrix[t_idx, d_idx] = 10000.0

    # Hungarian algo
    matched_indices = linear_sum_assignment(cost_matrix)
    matched_pairs = []
    unmatched_tracks = set(track_ids)
    unmatched_detections = set(detection_indices)

    for t_idx, d_idx in zip(*matched_indices):
        if cost_matrix[t_idx, d_idx] < DISTANCE_THRESHOLD_SOFT:
            matched_pairs.append((track_ids[t_idx], detection_indices[d_idx]))
            unmatched_tracks.discard(track_ids[t_idx])
            unmatched_detections.discard(detection_indices[d_idx])

    # Update tracks
    for track_id, det_idx in matched_pairs:
        original_det_index = valid_detections[det_idx]["original_index"]
        active_tracks[track_id]["bbox"] = valid_detections[det_idx]["bbox"]
        active_tracks[track_id]["age"] = 0
        active_tracks[track_id]["hits"] += 1
        current_detections[original_det_index]["track_id"] = track_id

    # For non-detected tracks
    tracks_to_delete = []
    for track_id in unmatched_tracks:
        active_tracks[track_id]["age"] += 1
        if active_tracks[track_id]["age"] > MAX_AGE_SOFT:
            tracks_to_delete.append(track_id)

    # Create new tracks for non-matched
    for det_idx in unmatched_detections:
        original_det_index = valid_detections[det_idx]["original_index"]
        new_id = next_track_id
        active_tracks[new_id] = {
            "bbox": valid_detections[det_idx]["bbox"],
            "age": 0,
            "hits": 1,
            "id": new_id,
        }
        current_detections[original_det_index]["track_id"] = new_id
        next_track_id += 1

    # Remove old tracks
    for track_id in tracks_to_delete:
        if track_id in active_tracks:
            del active_tracks[track_id]

    for det in frame_data["data"]:
        if "track_id" not in det:
            det["track_id"] = None

    return frame_data


def tracker_strong(frame_data):
    global active_tracks_kalman, next_track_id_kalman

    current_detections = frame_data["data"]

    predicted_track_bboxes = []
    track_ids_list = list(active_tracks_kalman.keys())
    for track_id in track_ids_list:
        pred_bbox = active_tracks_kalman[track_id].predict()
        predicted_track_bboxes.append(pred_bbox)

    valid_detections = []
    valid_detection_indices = []
    for i, det in enumerate(current_detections):
        if det["bounding_box"]:
            valid_detections.append(det["bounding_box"])
            valid_detection_indices.append(i)

    matched_indices_list = []
    unmatched_detections = list(range(len(valid_detections)))
    unmatched_tracks = list(range(len(track_ids_list)))

    if len(valid_detections) > 0 and len(predicted_track_bboxes) > 0:
        cost_matrix = np.ones((len(predicted_track_bboxes), len(valid_detections)))
        for t_idx, pred_bbox in enumerate(predicted_track_bboxes):
            for d_idx, det_bbox in enumerate(valid_detections):
                # Ensure bboxes are valid numpy arrays for calc_iou
                pred_bbox_np = np.array(pred_bbox) if pred_bbox is not None else None
                det_bbox_np = np.array(det_bbox)
                iou = calc_iou(pred_bbox_np, det_bbox_np)
                if iou > IOU_THRESHOLD_STRONG:
                    cost_matrix[t_idx, d_idx] = 1.0 - iou
                else:
                    cost_matrix[t_idx, d_idx] = 1.0  # High cost if below threshold

        # Hungarian algorithm
        matched_indices_row, matched_indices_col = linear_sum_assignment(cost_matrix)

        # Filter matches based on the cost (IoU threshold)
        for t_idx, d_idx in zip(matched_indices_row, matched_indices_col):
            if cost_matrix[t_idx, d_idx] < (1.0 - IOU_THRESHOLD_STRONG):
                matched_indices_list.append((t_idx, d_idx))
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_detections:
                    unmatched_detections.remove(d_idx)

    for t_idx, d_idx in matched_indices_list:
        track_id = track_ids_list[t_idx]
        detection_bbox = valid_detections[d_idx]
        active_tracks_kalman[track_id].update(detection_bbox)
        # Assign track_id to the original detection object
        original_det_index = valid_detection_indices[d_idx]
        current_detections[original_det_index]["track_id"] = track_id

    for d_idx in unmatched_detections:
        detection_bbox = valid_detections[d_idx]
        original_det_index = valid_detection_indices[d_idx]
        new_id = next_track_id_kalman
        active_tracks_kalman[new_id] = KalmanTrack(detection_bbox, new_id)
        current_detections[original_det_index]["track_id"] = new_id
        next_track_id_kalman += 1

    tracks_to_delete = []
    for track_id, track in list(active_tracks_kalman.items()):
        if track.time_since_update > MAX_AGE_STRONG:
            tracks_to_delete.append(track_id)
        else:
            pass

    for track_id in tracks_to_delete:
        if track_id in active_tracks_kalman:
            del active_tracks_kalman[track_id]

    for i, det in enumerate(current_detections):
        if "track_id" not in det:
            det["track_id"] = None  # Assign None if no track was assigned

    return frame_data


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Accepting client connection...")
    await websocket.accept()

    reset_tracker_state()
    tracking_history = []

    await websocket.send_text(str(country_balls))

    for frame_info in track_data:
        await asyncio.sleep(0.01)

        current_frame_data = {
            "frame_id": frame_info["frame_id"],
            "data": [d.copy() for d in frame_info["data"]],
        }

        # processed_frame_data = tracker_soft(current_frame_data)
        # OR
        processed_frame_data = tracker_strong(current_frame_data)

        tracking_history.append(processed_frame_data)

        print(processed_frame_data)
        await websocket.send_json(processed_frame_data)

    print("Calculating metrics...")
    metrics = calculate_tracking_metrics(tracking_history, track_data)
    print(f"Tracking Metrics: {metrics}")

    print("Bye..")


def calculate_tracking_metrics(tracking_history, ground_truth_data):
    id_switches = 0
    fragmentation_counts = {}
    gt_object_lifetimes = {}
    tracker_assignments = {}

    for frame_idx, processed_frame in enumerate(tracking_history):
        frame_id = processed_frame["frame_id"]

        gt_frame = ground_truth_data[frame_idx]
        if frame_id != gt_frame["frame_id"]:
            print(f"Warning: Frame ID mismatch at index {frame_idx}!")
            continue

        if len(processed_frame["data"]) != len(gt_frame["data"]):
            print(f"Warning: Data length mismatch in frame {frame_id}!")

            continue

        for i, processed_obj in enumerate(processed_frame["data"]):
            gt_obj = gt_frame["data"][i]
            cb_id = gt_obj["cb_id"]
            assigned_track_id = processed_obj.get("track_id")

            if assigned_track_id is not None:
                tracker_assignments[(frame_id, cb_id)] = assigned_track_id
                fragmentation_counts.setdefault(cb_id, set()).add(assigned_track_id)

            if cb_id not in gt_object_lifetimes:
                gt_object_lifetimes[cb_id] = [frame_id, frame_id]
            else:
                gt_object_lifetimes[cb_id][1] = frame_id

    sorted_frames = sorted(list(set(f_id for f_id, _ in tracker_assignments.keys())))

    for frame_id in sorted_frames:
        if frame_id == sorted_frames[0]:
            continue

        prev_frame_id = frame_id - 1

        current_frame_objs = {
            cb_id: tid
            for (f_id, cb_id), tid in tracker_assignments.items()
            if f_id == frame_id
        }
        prev_frame_objs = {
            cb_id: tid
            for (f_id, cb_id), tid in tracker_assignments.items()
            if f_id == prev_frame_id
        }

        common_cb_ids = set(current_frame_objs.keys()) & set(prev_frame_objs.keys())

        for cb_id in common_cb_ids:
            current_track_id = current_frame_objs[cb_id]
            prev_track_id = prev_frame_objs[cb_id]
            if current_track_id != prev_track_id:
                id_switches += 1

    total_fragments = sum(len(ids) for ids in fragmentation_counts.values())
    num_gt_tracks = len(fragmentation_counts)
    avg_fragments = total_fragments / num_gt_tracks if num_gt_tracks > 0 else 0

    return {
        "ID_Switches": id_switches,
        "Average_Fragments_Per_Track": avg_fragments,
        "Total_GT_Tracks_Detected": num_gt_tracks,
    }
