# __init__.py
from .nodes import PaddleOCR_Node, PaddleOCR_Unified_Node, PaddleOCR_VL_Node

NODE_CLASS_MAPPINGS = {
    "PaddleOCR_Node": PaddleOCR_Node,
    "PaddleOCR_Unified_Node": PaddleOCR_Unified_Node,
    "PaddleOCR_VL_Node": PaddleOCR_VL_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleOCR_Node": "PaddleOCR (Legacy)",
    "PaddleOCR_Unified_Node": "PaddleOCR Unified (PP-OCR)",
    "PaddleOCR_VL_Node": "PaddleOCR-VL Document Parser",
}




__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
