from __future__ import annotations

import os
from pathlib import Path


def _set_shape(value_info, shape: tuple[int, ...]) -> None:
    dims = value_info.type.tensor_type.shape.dim
    if len(dims) != len(shape):
        raise ValueError(f"Cannot replace rank-{len(dims)} shape with {shape}")
    for dim, value in zip(dims, shape):
        dim.ClearField("dim_param")
        dim.dim_value = value


def _make_model_static(source: Path, destination: Path, det_size: tuple[int, int]) -> None:
    import onnx  # type: ignore

    model = onnx.load(source)
    height, width = det_size

    if source.name == "det_10g.onnx":
        _set_shape(model.graph.input[0], (1, 3, height, width))

        # Buffalo-L SCRFD has two anchors at strides 8, 16, and 32. Its ONNX
        # output metadata is fixed for 640x640 even though its input is dynamic,
        # which is what trips CoreML when the application supplies 320x320.
        rows = [2 * (height // stride) * (width // stride) for stride in (8, 16, 32)]
        output_rows = rows + rows + rows
        output_columns = [1, 1, 1, 4, 4, 4, 10, 10, 10]
        if len(model.graph.output) != len(output_rows):
            raise ValueError("Unexpected SCRFD output layout")
        for output, row_count, column_count in zip(model.graph.output, output_rows, output_columns):
            _set_shape(output, (row_count, column_count))
    else:
        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        if not input_shape:
            raise ValueError(f"Model has no input shape: {source}")
        input_shape[0].ClearField("dim_param")
        input_shape[0].dim_value = 1

    onnx.checker.check_model(model)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        onnx.save(model, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_coreml_model_pack(
    det_size: tuple[int, int],
    root: str = "~/.insightface",
) -> tuple[str, str]:
    """Create a fixed-shape Buffalo-L copy suitable for ONNX Runtime CoreML."""
    from insightface.utils import ensure_available  # type: ignore

    width, height = int(det_size[0]), int(det_size[1])
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("CoreML face detection size must be positive and divisible by 32")

    expanded_root = str(Path(root).expanduser())
    source_dir = Path(ensure_available("models", "buffalo_l", root=expanded_root))
    pack_name = f"buffalo_l_coreml_{width}x{height}"
    destination_dir = Path(expanded_root) / "models" / pack_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    # det_size is expressed as (width, height), while tensor shapes use HxW.
    tensor_size = (height, width)
    sources = sorted(source_dir.glob("*.onnx"))
    if not sources:
        raise FileNotFoundError(f"No Buffalo-L ONNX models found in {source_dir}")

    for source in sources:
        destination = destination_dir / source.name
        if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            continue
        _make_model_static(source, destination, tensor_size)

    return expanded_root, pack_name
