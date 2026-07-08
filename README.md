# ComfyUI-PaddleOCR-VL

A ComfyUI custom node package for PaddleOCR.

This repository now exposes both standard PP-OCR scene-text nodes and a real PaddleOCR-VL document parsing node. The VL node uses `paddleocr.PaddleOCRVL`, not the ordinary `PaddleOCR` API.

## Features

- **PaddleOCR-VL Document Parser**: Parse document images into Markdown, plain text, and structured JSON with PaddleOCR-VL v1/v1.5/v1.6.
- **Document Layout Understanding**: Optional layout detection, chart recognition, seal recognition, document orientation classification, and document unwarping.
- **Standard PP-OCR Nodes**: Keep lightweight text detection and recognition with PP-OCRv5/v4/v3.
- **Model Version Selection**: Choose between PP-OCR versions for scene OCR, or PaddleOCR-VL pipeline versions for document parsing.

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

   `requirements.txt` already includes the PaddleOCR document parsing extra. If you are updating an existing environment manually, run:
   ```bash
   pip install -U "paddleocr[doc-parser]"
   ```

   If you use CUDA, install the PaddlePaddle GPU package that matches your CUDA version before installing this node's requirements. The plain `paddlepaddle` package is CPU-only.

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

Use `PaddleOCR Unified (PP-OCR)` or `PaddleOCR (Legacy)` when you only need ordinary text detection and recognition. These nodes use `PaddleOCR(...)` with PP-OCRv5/v4/v3 and are not PaddleOCR-VL.

## Verification

Run:
```bash
python verify_vl.py
```

If it prints `PaddleOCRVL NOT found`, install:
```bash
pip install -U "paddleocr[doc-parser]"
```

## Credits

This project wraps the amazing [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) library by PaddlePaddle.

## License

Apache 2.0
