try:
    from paddleocr import PaddleOCRVL
    print("PaddleOCRVL found")
    print(PaddleOCRVL)
except ImportError as e:
    print("PaddleOCRVL NOT found")
    print('Install with: pip install -U "paddleocr[doc-parser]"')
    print(e)
except Exception as e:
    print(f"Error: {e}")
