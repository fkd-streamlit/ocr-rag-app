"""
設定ファイル
"""

import os
from pathlib import Path

# ベースディレクトリ
BASE_DIR = Path(__file__).parent

# データディレクトリ
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OCR_RESULTS_DIR = DATA_DIR / "ocr_results"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# ディレクトリ作成
for dir_path in [DATA_DIR, UPLOADS_DIR, OCR_RESULTS_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Tesseract設定
TESSERACT_LANG = 'jpn'  # 日本語
TESSERACT_PSM_DEFAULT = 6  # 単一の均一なテキストブロック
TESSERACT_OEM_DEFAULT = 3  # デフォルトエンジン

# PDF設定
PDF_DPI = 200  # PDF変換時の解像度（無料枠向けに下げる）
MAX_PDF_PAGES = 20  # 無料枠向けの最大ページ数

# 画像前処理のデフォルト値
DEFAULT_CONTRAST = 1.0
DEFAULT_BRIGHTNESS = 0
DEFAULT_THRESHOLD = 127
DEFAULT_USE_ADAPTIVE = False

# ベクトルDB設定
VECTOR_DB_COLLECTION_NAME = "technical_documents"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # 日本語対応
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 10

# Streamlit設定
PAGE_TITLE = "技術資料OCR・RAG検索"
PAGE_ICON = "📄"

