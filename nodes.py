import json
import traceback
from .utils import tensor_to_cv2_img, get_paddle_hw_kwargs

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    from paddleocr import PaddleOCRVL
    PaddleOCRVL_IMPORT_ERROR = None
except Exception as e:
    PaddleOCRVL = None
    PaddleOCRVL_IMPORT_ERROR = e


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_markdown_text(markdown):
    if isinstance(markdown, dict):
        text = markdown.get("markdown_texts", "")
        if isinstance(text, list):
            return "\n\n".join(str(item) for item in text if item)
        return str(text or "")
    if isinstance(markdown, list):
        return "\n\n".join(str(item) for item in markdown if item)
    if isinstance(markdown, str):
        return markdown
    return ""


def _extract_plain_text_from_vl_json(data):
    data = data.get("res", data) if isinstance(data, dict) else data
    if not isinstance(data, dict):
        return ""

    parsing_res = data.get("parsing_res_list")
    if isinstance(parsing_res, list):
        chunks = []
        for block in parsing_res:
            if not isinstance(block, dict):
                continue
            content = block.get("block_content") or block.get("content") or block.get("text")
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
        if chunks:
            return "\n".join(chunks)

    chunks = []

    def walk(value):
        if isinstance(value, dict):
            for key in ("block_content", "content", "text", "rec_text"):
                item = value.get(key)
                if isinstance(item, str) and item.strip():
                    chunks.append(item.strip())
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(data)
    return "\n".join(dict.fromkeys(chunks))


def _normalize_engine(engine):
    return None if engine == "auto" else engine


def _normalize_optional_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value == "" or value.lower() == "auto" else value


def _normalize_optional_number(value):
    if value is None:
        return None
    return None if value < 0 else value


def _device_requests_cuda(device):
    if not device:
        return False
    device = str(device).lower()
    return "gpu" in device or "cuda" in device


def _validate_transformers_runtime(device):
    try:
        import torch
    except Exception as e:
        raise ImportError(
            "PaddleOCR-VL engine=transformers requires the PyTorch runtime from "
            "ComfyUI. Install or repair torch before running this node."
        ) from e

    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_available = bool(torch.cuda.is_available())
    print(
        "DEBUG: PaddleOCR-VL transformers runtime - "
        f"torch={torch_version}, torch.version.cuda={cuda_version}, "
        f"cuda_available={cuda_available}, device={device}"
    )

    if _device_requests_cuda(device) and not cuda_available:
        raise RuntimeError(
            "PaddleOCR-VL is configured for GPU inference, but PyTorch reports "
            "cuda.is_available() == False. For this ComfyUI install the expected "
            "runtime is torch 2.10.0+cu130 with CUDA 13.0."
        )


class PaddleOCR_Node:
    """
    Main PaddleOCR Custom Node.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "language": (["ch", "en", "japan", "korean", "chinese_cht", "french", "german"], {"default": "ch"}),
                # Renamed from use_angle_cls
                "vertical_direction": ("BOOLEAN", {"default": True}),
                # Added ocr_version
                "ocr_version": (["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"], {"default": "PP-OCRv5"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "apply_ocr"
    CATEGORY = "PaddleOCR"

    def apply_ocr(self, image, language, vertical_direction, ocr_version):
        try:
            if PaddleOCR is None:
                raise ImportError("PaddleOCR library is not installed.")

            print(f"DEBUG: Initializing PaddleOCR. Lang: {language}, Vertical: {vertical_direction}, Version: {ocr_version}")

            # Instantiate PaddleOCR
            # We pass 'use_textline_orientation' (which vertical_direction maps to)
            # and 'ocr_version' to let the internal logic handle model selection.
            # Get hardware kwargs (handles GPU/CPU/OneDNN automatically)
            hw_kwargs = get_paddle_hw_kwargs()
            print(f"DEBUG: Hardware Kwargs: {hw_kwargs}")

            try:
                 ocr = PaddleOCR(
                     use_textline_orientation=vertical_direction, 
                     lang=language,
                     ocr_version=ocr_version,
                     **hw_kwargs
                 )
            except TypeError as e:
                 print(f"DEBUG: Initialization TypeError: {e}")
                 # Fallback for older/standard versions that might not support keys
                 # We try 'use_angle_cls' if 'use_textline_orientation' fails, etc.
                 # But since the user is using the Pipeline wrapper, the above SHOULD work.
                 try:
                     ocr = PaddleOCR(use_angle_cls=vertical_direction, lang=language, **hw_kwargs)
                 except:
                     ocr = PaddleOCR(lang=language, **hw_kwargs)
            
            # process
            cv_images = tensor_to_cv2_img(image)
            full_text_results = []
            
            for i, img_numpy in enumerate(cv_images):
                # ocr() method
                try:
                    result = ocr.ocr(img_numpy, use_textline_orientation=vertical_direction)
                except TypeError:
                    # Fallback
                    result = ocr.ocr(img_numpy, cls=vertical_direction)
                
                if not result:
                    continue
                
                if result[0] is None:
                    continue    

                # Flatten 
                lines = result
                # Handle batch or odd structure
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and isinstance(result[0][0], list):
                     lines = result[0]

                for line in lines:
                    # Handle if line is dictionary (PaddleX Pipeline structure)
                    if isinstance(line, dict):
                         rec_texts = line.get('rec_texts', [])
                         if isinstance(rec_texts, list):
                             full_text_results.extend(rec_texts)
                         elif isinstance(rec_texts, str):
                             full_text_results.append(rec_texts)
                         else:
                             text = line.get('text', line.get('rec_text', ''))
                             if text:
                                 full_text_results.append(text)
                         continue
                    
                    # Standard structure
                    if isinstance(line, (list, tuple)) and len(line) > 1:
                        if isinstance(line[1], (list, tuple)):
                             text = line[1][0]
                        else:
                             text = line[0] if isinstance(line[0], str) else str(line)
                        full_text_results.append(text)

            full_text_string = "\n".join(full_text_results)
            return (full_text_string,)
            
        except Exception as e:
            print(f"CRITICAL ERROR in PaddleOCR_Node: {e}")
            traceback.print_exc()
            raise RuntimeError(f"PaddleOCR Failed: {e}\n{traceback.format_exc()}")


class PaddleOCR_TestNode:
    """
    A simple test node that adds 1 to the input integer.
    Useful for verifying basic custom node functionality.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int_input": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int_output",)
    FUNCTION = "test_add"
    CATEGORY = "PaddleOCR"

    def test_add(self, int_input):
        return (int_input + 1,)


class PaddleOCR_Unified_Node:
    """
    Reviewer: User (Designer)
    Concept: Pure OCR Node (v5/v4/v3)
    A single node acting as a facade for standard PaddleOCR capabilities.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ocr_version": (["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"], {"default": "PP-OCRv5"}),
                "language": (["ch", "en", "japan", "korean", "chinese_cht", "french", "german"], {"default": "ch"}),
                "use_angle_cls": ("BOOLEAN", {"default": True, "label_on": "Enable Angle Classification", "label_off": "Disable"}),
            },
            "optional": {
                "use_tensorrt": ("BOOLEAN", {"default": False, "label_on": "Enable TensorRT (Faster)", "label_off": "Disable TensorRT"}),
                "precision": (["fp32", "fp16", "int8"], {"default": "fp32"}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON")
    RETURN_NAMES = ("text", "json_output")
    FUNCTION = "apply_unified_ocr"
    CATEGORY = "PaddleOCR"

    def apply_unified_ocr(self, image, ocr_version, language, use_angle_cls, use_tensorrt, precision):
        hw_kwargs = get_paddle_hw_kwargs()
        
        # Inject user overrides for high-end optimization
        if use_tensorrt:
            hw_kwargs["use_tensorrt"] = True
            hw_kwargs["precision"] = precision
            print(f"DEBUG: TensorRT Enabled with precision {precision}")
            
        print(f"DEBUG: Unified Node (Pure OCR) - Ver: {ocr_version}, Lang: {language}, Angle: {use_angle_cls}, HW: {hw_kwargs}")

        try:
            cv_images = tensor_to_cv2_img(image)
            results_txt = []
            results_json = []

            if PaddleOCR is None:
                    raise ImportError("PaddleOCR not installed.")
            
            # Standard Init
            ocr = PaddleOCR(ocr_version=ocr_version, lang=language, use_angle_cls=use_angle_cls, **hw_kwargs)
            
            for img_numpy in cv_images:
                result = ocr.ocr(img_numpy, cls=use_angle_cls) 
                # Result structure: [[[[x1,y1],[x2,y2]..], ("text", score)], ...]
                
                page_txt = []
                page_json = []
                
                if result:
                        # Handle batch wrapper if needed
                        if isinstance(result, list) and len(result)>0 and isinstance(result[0], list) and isinstance(result[0][0], list):
                            lines = result[0]
                        else:
                            lines = result
                        
                        for line in lines:
                            # line: [box, (text, score)]
                            if len(line) >= 2:
                                text_info = line[1]
                                text = text_info[0]
                                score = text_info[1]
                                box = line[0]
                                
                                page_txt.append(text)
                                
                                page_json.append({
                                    "text": text,
                                    "confidence": float(score),
                                    "box": box
                                })
                
                results_txt.append("\n".join(page_txt))
                results_json.append(page_json)

            # Final Aggregation
            final_txt = "\n\n".join(results_txt)
            
            # JSON needs to be serialize-safe
            import json
            try:
                final_json_str = json.dumps(results_json, ensure_ascii=False, indent=2)
            except:
                final_json_str = str(results_json)

            return (final_txt, final_json_str)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Unified OCR Failed: {e}")


class PaddleOCR_VL_Node:
    """
    True PaddleOCR-VL document parsing node.
    Uses paddleocr.PaddleOCRVL with the Transformers engine by default so it can
    share ComfyUI's PyTorch/CUDA runtime instead of installing PaddlePaddle.
    """
    def __init__(self):
        self._pipeline = None
        self._pipeline_key = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipeline_version": (["v1.6", "v1.5", "v1"], {"default": "v1.6"}),
                "engine": (["transformers", "auto", "paddle", "paddle_static", "paddle_dynamic"], {"default": "transformers"}),
                "device": ("STRING", {"default": "gpu:0"}),
                "use_doc_orientation_classify": ("BOOLEAN", {"default": False}),
                "use_doc_unwarping": ("BOOLEAN", {"default": False}),
                "use_layout_detection": ("BOOLEAN", {"default": True}),
                "use_chart_recognition": ("BOOLEAN", {"default": False}),
                "use_seal_recognition": ("BOOLEAN", {"default": False}),
                "use_ocr_for_image_block": ("BOOLEAN", {"default": False}),
                "format_block_content": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "layout_threshold": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 1}),
                "vl_rec_server_url": ("STRING", {"default": ""}),
                "vl_rec_api_model_name": ("STRING", {"default": ""}),
                "use_tensorrt": ("BOOLEAN", {"default": False}),
                "precision": (["fp32", "fp16", "int8"], {"default": "fp32"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "JSON")
    RETURN_NAMES = ("markdown", "text", "json_output")
    FUNCTION = "apply_vl"
    CATEGORY = "PaddleOCR"

    def _get_pipeline(self, init_kwargs):
        key = json.dumps(_json_safe(init_kwargs), sort_keys=True, ensure_ascii=False)
        if self._pipeline is None or self._pipeline_key != key:
            print(f"DEBUG: Initializing PaddleOCRVL with {init_kwargs}")
            self._pipeline = PaddleOCRVL(**init_kwargs)
            self._pipeline_key = key
        return self._pipeline

    def apply_vl(
        self,
        image,
        pipeline_version,
        engine,
        device,
        use_doc_orientation_classify,
        use_doc_unwarping,
        use_layout_detection,
        use_chart_recognition,
        use_seal_recognition,
        use_ocr_for_image_block,
        format_block_content,
        layout_threshold=-1.0,
        max_new_tokens=0,
        vl_rec_server_url="",
        vl_rec_api_model_name="",
        use_tensorrt=False,
        precision="fp32",
    ):
        if PaddleOCRVL is None:
            detail = f" Import error: {PaddleOCRVL_IMPORT_ERROR}" if PaddleOCRVL_IMPORT_ERROR else ""
            raise ImportError(
                "PaddleOCRVL is not available. Install PaddleOCR with document parsing "
                "support and a Transformers/PyTorch backend, for example: "
                'pip install -U "paddleocr[doc-parser]" transformers accelerate.'
                f"{detail}"
            )

        try:
            init_kwargs = {
                "pipeline_version": pipeline_version,
                "use_doc_orientation_classify": use_doc_orientation_classify,
                "use_doc_unwarping": use_doc_unwarping,
                "use_layout_detection": use_layout_detection,
                "use_chart_recognition": use_chart_recognition,
                "use_seal_recognition": use_seal_recognition,
                "use_ocr_for_image_block": use_ocr_for_image_block,
                "format_block_content": format_block_content,
            }

            normalized_engine = _normalize_engine(engine)
            if normalized_engine:
                init_kwargs["engine"] = normalized_engine

            normalized_device = _normalize_optional_text(device)
            if normalized_device:
                init_kwargs["device"] = normalized_device

            if normalized_engine == "transformers":
                _validate_transformers_runtime(normalized_device)

            normalized_layout_threshold = _normalize_optional_number(layout_threshold)
            if normalized_layout_threshold is not None:
                init_kwargs["layout_threshold"] = normalized_layout_threshold

            normalized_max_new_tokens = max_new_tokens if max_new_tokens > 0 else None
            if normalized_max_new_tokens is not None:
                init_kwargs["max_new_tokens"] = normalized_max_new_tokens

            server_url = _normalize_optional_text(vl_rec_server_url)
            if server_url:
                init_kwargs["vl_rec_server_url"] = server_url

            api_model_name = _normalize_optional_text(vl_rec_api_model_name)
            if api_model_name:
                init_kwargs["vl_rec_api_model_name"] = api_model_name

            if use_tensorrt:
                init_kwargs["use_tensorrt"] = True
                init_kwargs["precision"] = precision

            pipeline = self._get_pipeline(init_kwargs)
            cv_images = tensor_to_cv2_img(image)

            predict_kwargs = {
                "use_doc_orientation_classify": use_doc_orientation_classify,
                "use_doc_unwarping": use_doc_unwarping,
                "use_layout_detection": use_layout_detection,
                "use_chart_recognition": use_chart_recognition,
                "use_seal_recognition": use_seal_recognition,
                "use_ocr_for_image_block": use_ocr_for_image_block,
                "format_block_content": format_block_content,
            }
            if normalized_layout_threshold is not None:
                predict_kwargs["layout_threshold"] = normalized_layout_threshold
            if normalized_max_new_tokens is not None:
                predict_kwargs["max_new_tokens"] = normalized_max_new_tokens

            try:
                results = pipeline.predict(input=cv_images, **predict_kwargs)
            except TypeError as e:
                if "input" not in str(e):
                    raise
                results = pipeline.predict(cv_images, **predict_kwargs)

            markdown_pages = []
            text_pages = []
            json_pages = []

            for result in results:
                if isinstance(result, dict):
                    result_json = result
                    result_markdown = result.get("markdown")
                else:
                    result_json = getattr(result, "json", None)
                    result_markdown = getattr(result, "markdown", None)

                if callable(result_json):
                    result_json = result_json()
                if callable(result_markdown):
                    result_markdown = result_markdown()

                markdown_text = _extract_markdown_text(result_markdown)
                plain_text = _extract_plain_text_from_vl_json(result_json) or markdown_text

                markdown_pages.append(markdown_text)
                text_pages.append(plain_text)
                json_pages.append(_json_safe(result_json if result_json is not None else result))

            markdown_output = "\n\n".join(page for page in markdown_pages if page)
            text_output = "\n\n".join(page for page in text_pages if page)
            json_output = json.dumps(json_pages, ensure_ascii=False, indent=2)

            return (markdown_output, text_output, json_output)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"PaddleOCR-VL Failed: {e}")
