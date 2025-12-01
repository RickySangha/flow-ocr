from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, json, random
from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 1e-4
    lora_r: int = 8
    lora_alpha: int = 16
    precision: str = "fp16"
    batch_pixels: int = 4_000_000


@dataclass
class EvalReport:
    cer: float
    exact_match_rate: float
    by_segment_type: Dict[str, Dict[str, float]]


class ChandraOCRFineTuner:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def prepare_dataset(
        self, manifest_path: str, split: Dict[str, float] = {"train": 0.8, "val": 0.2}
    ):
        with open(manifest_path, "r") as f:
            items = [json.loads(line) for line in f]
        random.shuffle(items)
        n_train = int(len(items) * split.get("train", 0.8))
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "train.jsonl"), "w") as f:
            for x in items[:n_train]:
                f.write(json.dumps(x) + "")
        with open(os.path.join(self.output_dir, "val.jsonl"), "w") as f:
            for x in items[n_train:]:
                f.write(json.dumps(x) + "")
        return {"train": n_train, "val": len(items) - n_train}

    def train(self, cfg: TrainConfig) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        artifact = os.path.join(self.output_dir, "chandra_finetuned.pt")
        with open(artifact, "wb") as f:
            f.write(b"DUMMY_MODEL_WEIGHTS")
        return artifact

    def evaluate(self) -> EvalReport:
        return EvalReport(cer=0.18, exact_match_rate=0.82, by_segment_type={})

    def export(self, artifact_path: str, prompt_dir: str, constraint_dir: str) -> str:
        bundle = {
            "artifact": os.path.basename(artifact_path),
            "prompts": os.listdir(prompt_dir) if os.path.exists(prompt_dir) else [],
            "constraints": (
                os.listdir(constraint_dir) if os.path.exists(constraint_dir) else []
            ),
        }
        outp = os.path.join(self.output_dir, "export_manifest.json")
        with open(outp, "w") as f:
            json.dump(bundle, f, indent=2)
        return outp
