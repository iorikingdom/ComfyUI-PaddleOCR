# ComfyUI-PaddleOCR-VL

A ComfyUI custom node package for PaddleOCR-VL document parsing.

This repository exposes a real PaddleOCR-VL document parsing node. The VL node uses `paddleocr.PaddleOCRVL` with the `transformers` engine by default, so it can share ComfyUI's existing PyTorch/CUDA runtime instead of installing PaddlePaddle into the ComfyUI environment.

Target runtime for the deployed ComfyUI machine:

- `torch 2.10.0+cu130`
- CUDA `13.0`
- NVIDIA GPU via `engine=transformers`

## Features

- **PaddleOCR-VL Document Parser**: Parse document images into Markdown, plain text, and structured JSON with PaddleOCR-VL v1/v1.5/v1.6.
- **PyTorch/Transformers Runtime**: Defaults to `engine=transformers` for ComfyUI-friendly inference and validates PyTorch CUDA availability before initializing PaddleOCR-VL.
- **Document Layout Understanding**: Optional layout detection, chart recognition, seal recognition, document orientation classification, and document unwarping.
- **Legacy PP-OCR Nodes**: Existing ordinary OCR nodes remain registered, but they still require PaddlePaddle and are not recommended for a PyTorch-only ComfyUI environment.
- **Model Version Selection**: Choose between PaddleOCR-VL pipeline versions for document parsing.

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone git@github.com:kaili-yang/ComfyUI-PaddleOCR-VL.git
   ```

3. Install the required dependencies:
   ```bash
   pip install -r ComfyUI-PaddleOCR-VL/requirements.txt
   ```

   `requirements.txt` intentionally does not install `paddlepaddle` or `torch`. It installs PaddleOCR 3.7.0 or newer with document parsing support plus Transformers-side dependencies, while reusing ComfyUI's existing PyTorch runtime. If you are updating an existing environment manually, run:
   ```bash
   pip install -U "paddleocr[doc-parser]>=3.7.0" transformers accelerate "opencv-python-headless<5" "numpy<2"
   ```

   Do not install `paddlepaddle` or `paddlepaddle-gpu` into the same venv as ComfyUI unless you have checked CUDA library compatibility. PaddlePaddle GPU wheels can replace NVIDIA runtime packages that PyTorch depends on.

   Do not install or upgrade `torch` from this package. The node is intended to reuse the ComfyUI environment's existing `torch 2.10.0+cu130`.

4. Restart ComfyUI.

## Usage

### PaddleOCR-VL Document Parser

1. Right-click in the ComfyUI canvas and search for `PaddleOCR-VL Document Parser`.
2. Connect an image source, such as `Load Image`, to the `image` input.
3. Choose a `pipeline_version`:
   - `v1.6`: latest default document parsing pipeline.
   - `v1.5`: previous VL pipeline with improved real-world robustness.
   - `v1`: original PaddleOCR-VL pipeline.
4. Configure optional modules:
   - `engine`: keep the default `transformers` for PyTorch/ComfyUI compatibility.
   - `device`: defaults to `gpu:0`; use `cpu` only for CPU inference.
   - `use_layout_detection`: use the full document parsing workflow.
   - `use_doc_orientation_classify`: classify and correct document orientation.
   - `use_doc_unwarping`: correct warped document images.
   - `use_chart_recognition`: enable chart parsing.
   - `use_seal_recognition`: enable seal/stamp recognition.
   - `format_block_content`: format block content as Markdown where supported.
5. Read the outputs:
   - `markdown`: page-level Markdown.
   - `text`: plain extracted text.
   - `json_output`: structured PaddleOCR-VL result JSON.

The first PaddleOCR-VL run may take a long time because PaddleOCR downloads and initializes large model files.

### Standard PP-OCR Nodes

Use `PaddleOCR Unified (PP-OCR)` or `PaddleOCR (Legacy)` only when you explicitly want ordinary text detection and recognition and have installed PaddlePaddle. These nodes use `PaddleOCR(...)` with PP-OCRv5/v4/v3 and are not PaddleOCR-VL.

## Verification

Run:
```bash
python verify_vl.py
```

If it prints `PaddleOCRVL NOT found`, install:
```bash
pip install -U "paddleocr[doc-parser]>=3.7.0" transformers accelerate "opencv-python-headless<5" "numpy<2"
```

## Credits

This project wraps the amazing [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) library by PaddlePaddle.

## License

Apache 2.0
