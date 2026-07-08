try:
    from paddleocr import PaddleOCRVL
    print("PaddleOCRVL found")
    print(PaddleOCRVL)
except ImportError as e:
    print("PaddleOCRVL NOT found")
    print('Install with: pip install -U "paddleocr[doc-parser]>=3.7.0" transformers accelerate "opencv-python-headless<5" "numpy<2"')
    print(e)
except Exception as e:
    print(f"Error: {e}")
