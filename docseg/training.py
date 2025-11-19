from typing import Optional, Dict, Any
import os

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class SegmenterTrain:
    def __init__(self):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. pip install ultralytics")

    def train_new(
        self,
        data_yaml: str,
        model_arch: str = "yolov8n-seg.pt",
        imgsz: int = 1024,
        epochs: int = 200,
        batch: int = 8,
        lr0: float = 0.003,
        project: str = "runs/segment",
        name: str = "docseg",
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        model = YOLO(model_arch)
        kwargs = dict(
            data=data_yaml,
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            project=project,
            name=name,
            pretrained=True,
            save=True,
            patience=100,
            hsv_h=0.02,
            hsv_s=0.2,
            hsv_v=0.2,
            degrees=10.0,
            translate=0.05,
            scale=0.15,
            shear=6.0,
            mosaic=0.05,
            mixup=0.0,
            copy_paste=0.05,
        )
        if extra:
            kwargs.update(extra)
        model.train(**kwargs)
        return os.path.join(project, name)

    def finetune_existing(
        self,
        weights_path: str,
        data_yaml: str,
        imgsz: int = 1024,
        epochs: int = 100,
        batch: int = 8,
        lr0: float = 0.003,
        project: str = "runs/segment",
        name: str = "docseg_ft",
        freeze: int = 10,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        model = YOLO(weights_path)
        kwargs = dict(
            data=data_yaml,
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            project=project,
            name=name,
            pretrained=True,
            save=True,
            patience=100,
            freeze=freeze,
            hsv_h=0.02,
            hsv_s=0.2,
            hsv_v=0.2,
            degrees=8.0,
            translate=0.05,
            scale=0.12,
            shear=5.0,
            mosaic=0.05,
            mixup=0.0,
            copy_paste=0.05,
        )
        if extra:
            kwargs.update(extra)
        model.train(**kwargs)
        return os.path.join(project, name)
