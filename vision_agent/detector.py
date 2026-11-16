"""YOLO object detection + ByteTrack multi-object tracking (persistent IDs across frames)."""
from ultralytics import YOLO


class Detector:
    def __init__(self, weights="yolo11n.pt", classes=None, conf=0.35, imgsz=640):
        self.model = YOLO(weights)  # auto-downloads on first run
        self.names = self.model.names
        wanted = {c.strip().lower() for c in classes} if classes else None
        self.class_ids = [i for i, n in self.names.items() if n in wanted] if wanted else None
        if wanted and not self.class_ids:
            raise ValueError(f"None of {sorted(wanted)} are YOLO classes. Valid: {sorted(self.names.values())}")
        self.conf = conf
        self.imgsz = imgsz

    def track(self, frame):
        """Return list of {id, cls, conf, box[x1,y1,x2,y2]} for tracked objects in this frame."""
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self.class_ids,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        out = []
        for box, tid, cls, cf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.int().cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            out.append({"id": tid, "cls": self.names[cls], "conf": float(cf), "box": box.tolist()})
        return out
