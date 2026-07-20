from __future__ import annotations

import unittest
from unittest.mock import patch

from face_yolo_identity import apple_runtime


class RuntimeSelectionTests(unittest.TestCase):
    def test_apple_mps_precedes_other_accelerators(self) -> None:
        with patch.object(apple_runtime, "_torch_mps_available", return_value=True), patch.object(apple_runtime, "_torch_accelerator", return_value=(True, None, "12.4")), patch.object(apple_runtime, "available_onnx_providers", return_value=["CPUExecutionProvider"]):
            info = apple_runtime.runtime_info()
        self.assertEqual(info.yolo_device, "mps")
        self.assertEqual(info.accelerator, "apple_mps")

    def test_nvidia_cuda(self) -> None:
        with patch.object(apple_runtime, "_torch_mps_available", return_value=False), patch.object(apple_runtime, "_torch_accelerator", return_value=(True, None, "12.4")), patch.object(apple_runtime, "available_onnx_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            info = apple_runtime.runtime_info()
        self.assertEqual(info.accelerator, "nvidia_cuda")
        self.assertEqual(info.face_providers[0], "CUDAExecutionProvider")

    def test_amd_rocm_uses_torch_cuda_api(self) -> None:
        with patch.object(apple_runtime, "_torch_mps_available", return_value=False), patch.object(apple_runtime, "_torch_accelerator", return_value=(True, "6.4", None)), patch.object(apple_runtime, "available_onnx_providers", return_value=["MIGraphXExecutionProvider", "CPUExecutionProvider"]):
            info = apple_runtime.runtime_info()
        self.assertEqual(info.yolo_device, "cuda")
        self.assertEqual(info.accelerator, "amd_rocm")
        self.assertEqual(info.face_providers[0], "MIGraphXExecutionProvider")

    def test_cpu_fallback(self) -> None:
        with patch.object(apple_runtime, "_torch_mps_available", return_value=False), patch.object(apple_runtime, "_torch_accelerator", return_value=(False, None, None)), patch.object(apple_runtime, "available_onnx_providers", return_value=["CPUExecutionProvider"]):
            info = apple_runtime.runtime_info()
        self.assertEqual(info.yolo_device, "cpu")
        self.assertEqual(info.accelerator, "cpu")


if __name__ == "__main__":
    unittest.main()
