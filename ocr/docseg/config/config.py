from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DocTypeConfig:
    weights_path: str
    class_names: List[str]


@dataclass
class DocTypeRegistry:
    types: Dict[str, DocTypeConfig] = field(default_factory=dict)

    def add(self, doc_type: str, weights_path: str, class_names: List[str]):
        self.types[doc_type] = DocTypeConfig(weights_path, class_names)

    def get_weights(self, doc_type: str) -> Optional[str]:
        cfg = self.types.get(doc_type)
        return cfg.weights_path if cfg else None

    def get_classes(self, doc_type: str) -> Optional[List[str]]:
        cfg = self.types.get(doc_type)
        return list(cfg.class_names) if cfg else None
