from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ONNXModel:
    def __init__(self, model_path: str, providers: Optional[list[str]] = None):
        if ort is None:
            raise RuntimeError("Install onnxruntime: pip install onnxruntime")
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, inputs: dict) -> dict:
        outputs = self.session.run(self.output_names, inputs)
        return dict(zip(self.output_names, outputs))

    def benchmark(
        self, inputs: dict, warmup: int = 50, iterations: int = 200
    ) -> dict:
        for _ in range(warmup):
            self.run(inputs)
        latencies_ms = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.run(inputs)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        arr = np.array(latencies_ms)
        return {
            "iterations": iterations,
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(arr.mean()), 3),
            "throughput_rps": round(iterations / (arr.sum() / 1000), 1),
        }

    @property
    def metadata(self) -> dict:
        meta = self.session.get_modelmeta()
        return {
            "name": meta.graph_name or "unknown",
            "inputs": [
                {"name": i.name, "shape": i.shape, "type": i.type}
                for i in self.session.get_inputs()
            ],
            "outputs": [
                {"name": o.name, "shape": o.shape, "type": o.type}
                for o in self.session.get_outputs()
            ],
            "providers": self.session.get_providers(),
        }
