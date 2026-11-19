"""
Detektoriai, grąžinantys „ar matomas automobilio žibintas“ tikimybes.

BlobDetector  – dėmių išskyrimas + išmokytas klasifikatorius (veikia CPU, be GPU).
YoloDetector  – YOLO11 objektų detektorius (Ultralytics), išmokytas naktiniais
                vaizdais su klasėmis „headlight“ ir „taillight“ (žr. train_yolo.py).
Abu turi tą pačią sąsają: detect(frame) -> list[dict(bbox, cls, prob)].
"""
import numpy as np
import joblib

from features import extract_blobs, blob_features


class BlobDetector:
    def __init__(self, model_path="blob_classifier.joblib"):
        self.model = joblib.load(model_path)
        if hasattr(self.model, "n_jobs"):   # vienas kadras – lygiagretumas tik lėtina
            self.model.n_jobs = 1
        # v1 modelis naudojo 15 požymių (su x_rel – horizontalia padėtimi kadre)
        self.legacy_v1 = getattr(self.model, "n_features_in_", 14) == 15

    def detect(self, frame):
        blobs, hsv = extract_blobs(frame)
        if not blobs:
            return []
        X = blob_features(frame, blobs, hsv)
        if self.legacy_v1:   # v1 požymiai: be ring_sat/ring_v, su x_rel
            x_rel = np.array([b["c"][0] / frame.shape[1] for b in blobs], np.float32)
            X = np.insert(X[:, :14], 12, x_rel, axis=1)
        else:                # senesni modeliai naudoja tik pirmuosius požymius
            X = X[:, :getattr(self.model, "n_features_in_", X.shape[1])]
        P = self.model.predict_proba(X)
        out = []
        for b, p in zip(blobs, P):
            k = int(np.argmax(p))
            out.append({"bbox": b["bbox"], "c": b["c"], "cls": k,
                        "prob": float(p[0] + p[1])})   # P(priekinis) + P(galinis)
        return out


class YoloDetector:
    """Reikia: pip install ultralytics ; modelis, išmokytas train_yolo.py."""
    def __init__(self, weights="runs/detect/train/weights/best.pt", conf=0.35):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        for box, c, p in zip(r.boxes.xywh.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                             r.boxes.conf.cpu().numpy()):
            x, y, w, h = box
            out.append({"bbox": (int(x - w / 2), int(y - h / 2), int(w), int(h)),
                        "c": (float(x), float(y)), "cls": int(c), "prob": float(p)})
        return out


class OnnxYoloDetector:
    """
    YOLO11 per OpenCV DNN (ONNX formatas) – nereikia nei PyTorch, nei ultralytics.

    Numatytasis modelis yolo11n.onnx yra išmokytas COCO duomenų rinkiniu (80 klasių,
    daugiausia dienos vaizdai). Automobiliu laikomos klasės car, motorcycle, bus,
    truck. Savo modeliui, išmokytam naktiniais vaizdais (yolo_colab.ipynb), kurio
    visos klasės reiškia automobilį ar jo žibintus, nurodykite vehicle_classes=None.
    """
    COCO_VEHICLES = (2, 3, 5, 7)          # car, motorcycle, bus, truck

    def __init__(self, onnx_path="yolo11n.onnx", conf=0.25, size=640,
                 vehicle_classes=COCO_VEHICLES, geometry_filter=False):
        import cv2
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.conf, self.size = conf, size
        # atmesti rėmelius, kurie negali būti priešais esantis automobilis:
        # liečiantys kadro apačią (kapotas, skydelio atspindžiai, mūsų apšviestas kelias)
        # arba platesni nei 40 % kadro
        self.geometry_filter = geometry_filter
        self.vehicle_classes = None if vehicle_classes is None else tuple(vehicle_classes)

    def detect(self, frame):
        cv2, S = self.cv2, self.size
        h, w = frame.shape[:2]
        r = S / max(h, w)                                   # „letterbox“: išlaikomos proporcijos
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((S, S, 3), 114, np.uint8)
        top, left = (S - nh) // 2, (S - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (S, S), swapRB=True)
        self.net.setInput(blob)
        pred = self.net.forward()[0].T                      # (8400, 4 + klasės)
        cls_scores = pred[:, 4:]
        scores = cls_scores if self.vehicle_classes is None else cls_scores[:, list(self.vehicle_classes)]
        best = scores.max(axis=1)
        keep = best >= min(self.conf, 0.1)                  # žemesnės – sekimui
        boxes, confs = [], []
        for (cx, cy, bw, bh), p in zip(pred[keep, :4], best[keep]):
            x = (cx - bw / 2 - left) / r
            y = (cy - bh / 2 - top) / r
            boxes.append([int(x), int(y), int(bw / r), int(bh / r)])
            confs.append(float(p))
        out = []
        if boxes:
            for i in np.array(cv2.dnn.NMSBoxes(boxes, confs, 0.1, 0.5)).flatten():
                x, y, bw, bh = boxes[i]
                if self.geometry_filter and (y + bh > 0.92 * h or bw > 0.4 * w):
                    continue
                out.append({"bbox": (x, y, bw, bh), "c": (x + bw / 2, y + bh / 2),
                            "cls": 0, "prob": confs[i]})
        return out


class RuleDetector:
    """
    Palyginimui: taisyklėmis paremtas detektorius be mašininio mokymosi
    („kiekviena pakankamai ryški dėmė žemiau horizonto yra automobilis“).
    """
    def detect(self, frame):
        blobs, _ = extract_blobs(frame)
        H = frame.shape[0]
        return [{"bbox": b["bbox"], "c": b["c"], "cls": 0, "prob": 1.0}
                for b in blobs if b["area"] >= 2 and b["c"][1] > 0.4 * H]
