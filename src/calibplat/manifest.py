"""Reproducibility manifests.

A manifest pins everything needed to reproduce a calibrator version:
dataset fingerprint, config, seeds, model/calibrator versions, and a
content hash over all of it. ``verify()`` recomputes the hash and
fails on any tampering — the same contract the sibling projects use.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True).encode()


@dataclass
class ReproducibilityManifest:
    version_name: str
    dataset_fingerprint: str
    config: dict
    model_config: dict
    calibrator_config: dict
    seeds: dict
    content_hash: str = ""
    environment: dict = field(default_factory=dict)

    @classmethod
    def build(cls, version_name: str, dataset_fingerprint: str,
              config: dict, model_config: dict, calibrator_config: dict,
              seeds: dict) -> "ReproducibilityManifest":
        manifest = cls(
            version_name=version_name,
            dataset_fingerprint=dataset_fingerprint,
            config=config, model_config=model_config,
            calibrator_config=calibrator_config, seeds=seeds,
            environment={"python": sys.version.split()[0],
                         "platform": platform.platform()})
        manifest.content_hash = manifest._compute_hash()
        return manifest

    def _compute_hash(self) -> str:
        payload = {"version_name": self.version_name,
                   "dataset_fingerprint": self.dataset_fingerprint,
                   "config": self.config,
                   "model_config": self.model_config,
                   "calibrator_config": self.calibrator_config,
                   "seeds": self.seeds}
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def verify(self) -> bool:
        return self._compute_hash() == self.content_hash

    def to_dict(self) -> dict:
        return {"version_name": self.version_name,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config": self.config,
                "model_config": self.model_config,
                "calibrator_config": self.calibrator_config,
                "seeds": self.seeds,
                "content_hash": self.content_hash,
                "environment": self.environment}

    @classmethod
    def from_dict(cls, d: dict) -> "ReproducibilityManifest":
        return cls(version_name=d["version_name"],
                   dataset_fingerprint=d["dataset_fingerprint"],
                   config=d["config"], model_config=d["model_config"],
                   calibrator_config=d["calibrator_config"],
                   seeds=d["seeds"], content_hash=d["content_hash"],
                   environment=d.get("environment", {}))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReproducibilityManifest):
            return NotImplemented
        return self.to_dict() == other.to_dict()
