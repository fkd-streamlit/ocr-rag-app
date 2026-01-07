# app.py
"""
技術資料OCR・RAG検索アプリケーション
- PDF/画像ファイルをアップロードしてOCR処理
- 画像前処理（コントラスト、明度、閾値処理）をリアルタイムで調整可能
- OCR精度調整（PSM、OEM）
- OCR結果をベクトルDBに保存してRAG検索可能に
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

# PDF処理
try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# -------------------------
# 0) 設定（config.py があれば優先）
# -------------------------
try:
    from config import (
        DATA_DIR,
        UPLOADS_DIR,
        OCR_RESULTS_DIR,
        VECTOR_DB_DIR,
        TESSERACT_LANG,
        TESSERACT_PSM_DEFAULT,
        TESSERACT_OEM_DEFAULT,
        PDF_DPI,
        MAX_PDF_PAGES,
        DEFAULT_CONTRAST,
        DEFAULT_BRIGHTNESS,
        DEFAULT_THRESHOLD,
        DEFAULT_USE_ADAPTIVE,
        VECTOR_DB_COLLECTION_NAME,
        EMBEDDING_MODEL_NAME,
        DEFAULT_SEARCH_RESULTS,
        MAX_SEARCH_RESULTS,
        PAGE_TITLE,
        PAGE_ICON,
    )
except Exception:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"
    OCR_RESULTS_DIR = DATA_DIR / "ocr_results"
    VECTOR_DB_DIR = DATA_DIR / "chroma_db"
    TESSERACT_LANG = 'jpn'
    TESSERACT_PSM_DEFAULT = 6
    TESSERACT_OEM_DEFAULT = 3
    PDF_DPI = 200
    MAX_PDF_PAGES = 20
    DEFAULT_CONTRAST = 1.0
    DEFAULT_BRIGHTNESS = 0
    DEFAULT_THRESHOLD = 127
    DEFAULT_USE_ADAPTIVE = False
    VECTOR_DB_COLLECTION_NAME = "technical_documents"
    EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    DEFAULT_SEARCH_RESULTS = 5
    MAX_SEARCH_RESULTS = 10
    PAGE_TITLE = "技術資料OCR・RAG検索"
    PAGE_ICON = "📄"

# ディレクトリ作成
for dir_path in [DATA_DIR, UPLOADS_DIR, OCR_RESULTS_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1) Tesseract OCR設定
# -------------------------
def find_tesseract_cmd() -> Optional[str]:
    """Tesseractのパスを検出（Windows/Linux対応）"""
    # 環境変数から取得
    if os.environ.get("TESSERACT_CMD"):
        return os.environ.get("TESSERACT_CMD")
    
    # Windowsのデフォルトパス
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in windows_paths:
        if os.path.exists(path):
            return path
    
    # Linux/Mac: whichコマンドで検索
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        return tesseract_cmd
    
    return None

TESSERACT_CMD = find_tesseract_cmd()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# -------------------------
# 2) RAG依存の読み込み（あれば使う）
# -------------------------
CHROMADB_AVAILABLE = True
CHROMA_IMPORT_ERROR = ""

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except Exception as e:
    CHROMADB_AVAILABLE = False
    CHROMA_IMPORT_ERROR = str(e)

# -------------------------
# 3) UI設定
# -------------------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# 4) データ構造
# -------------------------
@dataclass
class Doc:
    id: str
    filename: str
    text: str
    uploaded_at: str
    meta: Dict[str, Any]

def now_id(prefix: str = "doc") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# -------------------------
# 5) 画像前処理
# -------------------------
def preprocess_image(
    image: np.ndarray,
    contrast: float = 1.0,
    brightness: int = 0,
    threshold: int = 127,
    use_adaptive: bool = False,
) -> np.ndarray:
    """画像前処理（コントラスト、明度、閾値処理）"""
    img = image.copy()
    
    # コントラスト調整
    if contrast != 1.0:
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
    
    # 明度調整
    if brightness != 0:
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness)
    
    # グレースケール変換
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 閾値処理
    if use_adaptive:
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    return img

# -------------------------
# 6) OCR実行
# -------------------------
def perform_ocr(
    image: np.ndarray,
    lang: str = TESSERACT_LANG,
    psm: int = TESSERACT_PSM_DEFAULT,
    oem: int = TESSERACT_OEM_DEFAULT,
) -> str:
    """Tesseract OCRでテキスト抽出"""
    if TESSERACT_CMD is None:
        raise RuntimeError(
            "Tesseract OCRが見つかりません。\n"
            "Windows: C:\\Program Files\\Tesseract-OCR\\tesseract.exe にインストールしてください。\n"
            "Linux/Mac: sudo apt-get install tesseract-ocr tesseract-ocr-jpn を実行してください。\n"
            "または環境変数 TESSERACT_CMD にパスを設定してください。"
        )
    
    config = f"--psm {psm} --oem {oem} -l {lang}"
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

# -------------------------
# 7) PDF処理
# -------------------------
def pdf_to_images(pdf_path: Path, dpi: int = PDF_DPI) -> List[Image.Image]:
    """PDFを画像に変換"""
    if not PDF_AVAILABLE:
        raise RuntimeError("pdf2imageがインストールされていません。pip install pdf2image poppler-utils")
    
    images = convert_from_path(str(pdf_path), dpi=dpi)
    return images

# -------------------------
# 8) テキスト分割（RAG用 chunking）
# -------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """シンプルな文字数ベース分割（日本語向けに安定）"""
    text = text.replace("\r\n", "\n")
    if len(text) <= chunk_size:
        return [text]
    
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks

# -------------------------
# 9) RAG（ChromaDB）
# -------------------------
@st.cache_resource
def get_embedding_model() -> Optional["SentenceTransformer"]:
    if not CHROMADB_AVAILABLE:
        return None
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_chroma_collection():
    if not CHROMADB_AVAILABLE:
        return None
    # 新しいChromaDBクライアント構築方法
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        col = client.get_collection(VECTOR_DB_COLLECTION_NAME)
    except Exception:
        col = client.create_collection(VECTOR_DB_COLLECTION_NAME)
    return col

def rag_add_doc(doc: Doc) -> Tuple[bool, str]:
    """1文書を chunk に分解してベクトルDBに追加"""
    if not CHROMADB_AVAILABLE:
        return False, "ChromaDB/SentenceTransformers が未導入です"
    
    model = get_embedding_model()
    col = get_chroma_collection()
    if model is None or col is None:
        return False, "RAG初期化に失敗しました"
    
    chunks = chunk_text(doc.text)
    ids = [f"{doc.id}__c{i:04d}" for i in range(len(chunks))]
    metadatas = []
    for i in range(len(chunks)):
        metadatas.append(
            {
                "doc_id": doc.id,
                "chunk_index": i,
                "filename": doc.filename,
                "uploaded_at": doc.uploaded_at,
            }
        )
    
    try:
        emb = model.encode(chunks).tolist()
        col.add(ids=ids, embeddings=emb, documents=chunks, metadatas=metadatas)
        return True, f"RAGに登録しました（chunks={len(chunks)}）"
    except Exception as e:
        return False, f"RAG登録エラー: {e}"

def rag_delete_doc(doc_id: str) -> Tuple[bool, str]:
    if not CHROMADB_AVAILABLE:
        return False, "ChromaDB未導入"
    col = get_chroma_collection()
    if col is None:
        return False, "コレクション取得失敗"
    
    try:
        all_ids = col.get(include=["metadatas"]).get("ids", [])
        all_metas = col.get(include=["metadatas"]).get("metadatas", [])
        del_ids = []
        for _id, m in zip(all_ids, all_metas):
            if isinstance(m, dict) and m.get("doc_id") == doc_id:
                del_ids.append(_id)
        if del_ids:
            col.delete(ids=del_ids)
        return True, f"RAGから削除しました（{len(del_ids)}件）"
    except Exception as e:
        return False, f"RAG削除エラー: {e}"

def rag_search(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    if not CHROMADB_AVAILABLE:
        return []
    
    model = get_embedding_model()
    col = get_chroma_collection()
    if model is None or col is None:
        return []
    
    try:
        qemb = model.encode([query]).tolist()
        res = col.query(query_embeddings=qemb, n_results=n_results, include=["documents", "metadatas", "distances", "ids"])
        out: List[Dict[str, Any]] = []
        if res and res.get("ids") and len(res["ids"][0]) > 0:
            for i in range(len(res["ids"][0])):
                out.append(
                    {
                        "id": res["ids"][0][i],
                        "text": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i],
                        "distance": res["distances"][0][i] if res.get("distances") else None,
                    }
                )
        return out
    except Exception:
        return []

# -------------------------
# 10) 簡易検索（RAGが無いとき）
# -------------------------
def simple_search(docs: List[Doc], query: str, limit: int = 5) -> List[Dict[str, Any]]:
    q = query.strip()
    if not q:
        return []
    hits = []
    for d in docs:
        idx = d.text.find(q)
        if idx >= 0:
            start = max(0, idx - 120)
            end = min(len(d.text), idx + 400)
            snippet = d.text[start:end]
            hits.append(
                {
                    "doc_id": d.id,
                    "filename": d.filename,
                    "snippet": snippet,
                    "pos": idx,
                }
            )
    hits.sort(key=lambda x: x["pos"])
    return hits[:limit]

# -------------------------
# 11) セッション初期化
# -------------------------
if "documents" not in st.session_state:
    st.session_state.documents: List[Doc] = []

if "processed_images" not in st.session_state:
    st.session_state.processed_images: Dict[str, np.ndarray] = {}

# -------------------------
# 12) サイドバー（画像前処理設定）
# -------------------------
with st.sidebar:
    st.header("⚙️ OCR設定")
    
    # Tesseract状態
    st.subheader("Tesseract OCR")
    if TESSERACT_CMD:
        st.success(f"✅ Tesseract検出: {TESSERACT_CMD}")
    else:
        st.error("❌ Tesseract OCRが見つかりません")
        st.caption("Windows: C:\\Program Files\\Tesseract-OCR\\tesseract.exe にインストール")
        st.caption("Linux/Mac: sudo apt-get install tesseract-ocr tesseract-ocr-jpn")
    
    st.markdown("---")
    
    # 画像前処理設定
    st.subheader("📸 画像前処理")
    contrast = st.slider("コントラスト", 0.5, 2.0, DEFAULT_CONTRAST, 0.1)
    brightness = st.slider("明度", -100, 100, DEFAULT_BRIGHTNESS, 5)
    threshold = st.slider("閾値", 0, 255, DEFAULT_THRESHOLD, 1)
    use_adaptive = st.checkbox("適応的閾値処理", DEFAULT_USE_ADAPTIVE)
    
    st.markdown("---")
    
    # OCR設定
    st.subheader("🔍 OCR設定")
    lang = st.selectbox("言語", ["jpn", "eng", "jpn+eng"], index=0)
    psm = st.slider("PSM (Page Segmentation Mode)", 0, 13, TESSERACT_PSM_DEFAULT, 1)
    st.caption("6: 単一の均一なテキストブロック（推奨）")
    oem = st.slider("OEM (OCR Engine Mode)", 0, 3, TESSERACT_OEM_DEFAULT, 1)
    st.caption("3: デフォルトエンジン（推奨）")
    
    st.markdown("---")
    
    # RAG状態
    st.subheader("RAG状態")
    if CHROMADB_AVAILABLE:
        st.success("✅ RAG（ChromaDB + Embedding）利用可能")
        st.caption(f"Embedding: {EMBEDDING_MODEL_NAME}")
    else:
        st.warning("⚠️ RAGは未使用（簡易検索で動作）")
        st.caption(f"理由: {CHROMA_IMPORT_ERROR}")
    
    st.markdown("---")
    
    # 保存データ
    st.subheader("保存データ")
    st.write(f"保存済み文書数: **{len(st.session_state.documents)}**")

# -------------------------
# 13) メインUI
# -------------------------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("---")

tab_upload, tab_search, tab_list = st.tabs(["📤 ファイルアップロード", "🔍 検索", "📚 文書一覧"])

# ========== Tab 1: ファイルアップロード ==========
with tab_upload:
    st.header("📤 PDF/画像ファイルのアップロード")
    
    uploaded_file = st.file_uploader(
        "PDFまたは画像ファイルを選択",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        help="PDFまたは画像ファイルをアップロードしてOCR処理を行います"
    )
    
    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        # ファイル保存
        save_path = UPLOADS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # PDF処理
        if file_ext == ".pdf":
            if not PDF_AVAILABLE:
                st.error("pdf2imageがインストールされていません。pip install pdf2image poppler-utils")
            else:
                try:
                    with st.spinner("PDFを画像に変換中..."):
                        images = pdf_to_images(save_path, dpi=PDF_DPI)
                    
                    total_pages = len(images)
                    
                    # ページ数制限チェック
                    if total_pages > MAX_PDF_PAGES:
                        st.warning(f"⚠️ PDFが{total_pages}ページあります。無料枠では最大{MAX_PDF_PAGES}ページまで処理します。")
                        st.info(f"最初の{MAX_PDF_PAGES}ページのみ処理します。")
                        images = images[:MAX_PDF_PAGES]
                    
                    st.success(f"✅ PDFを{len(images)}ページの画像に変換しました（全{total_pages}ページ中）")
                    
                    # 最初のページを表示
                    if images:
                        st.subheader("📄 プレビュー（最初のページ）")
                        # 画像をリサイズしてメモリ使用量を削減
                        preview_img = images[0].copy()
                        if preview_img.width > 1200:
                            preview_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                        st.image(preview_img, caption=f"Page 1/{len(images)}", use_container_width=True)
                        
                        # 画像をnumpy配列に変換
                        img_array = np.array(preview_img)
                        if len(img_array.shape) == 3:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # 前処理
                        processed = preprocess_image(
                            img_array,
                            contrast=contrast,
                            brightness=brightness,
                            threshold=threshold,
                            use_adaptive=use_adaptive,
                        )
                        
                        # 前処理後の画像を表示
                        st.subheader("🔧 前処理後の画像")
                        st.image(processed, caption="前処理後", use_container_width=True)
                        
                        # メモリ解放
                        del preview_img, img_array, processed
                        
                        # OCR実行
                        if st.button("🔍 OCR実行", type="primary"):
                            all_text = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            try:
                                for i, img in enumerate(images):
                                    status_text.text(f"ページ {i+1}/{len(images)} を処理中...")
                                    
                                    # 画像をリサイズしてメモリ使用量を削減
                                    if img.width > 1600 or img.height > 1600:
                                        img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                                    
                                    img_array = np.array(img)
                                    if len(img_array.shape) == 3:
                                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                    
                                    processed = preprocess_image(
                                        img_array,
                                        contrast=contrast,
                                        brightness=brightness,
                                        threshold=threshold,
                                        use_adaptive=use_adaptive,
                                    )
                                    
                                    try:
                                        text = perform_ocr(processed, lang=lang, psm=psm, oem=oem)
                                        if text:
                                            all_text.append(f"=== ページ {i+1} ===\n{text}\n")
                                    except Exception as e:
                                        st.error(f"ページ {i+1} のOCRエラー: {e}")
                                    
                                    # メモリ解放
                                    del img, img_array, processed
                                    
                                    progress_bar.progress((i + 1) / len(images))
                                    
                                    # 無料枠向け：処理を分割してメモリを解放
                                    if (i + 1) % 5 == 0:
                                        import gc
                                        gc.collect()
                                
                                if all_text:
                                    full_text = "\n".join(all_text)
                                    
                                    # ドキュメント作成
                                    doc_id = now_id()
                                    doc = Doc(
                                        id=doc_id,
                                        filename=uploaded_file.name,
                                        text=full_text,
                                        uploaded_at=datetime.now().isoformat(),
                                        meta={
                                            "id": doc_id,
                                            "filename": uploaded_file.name,
                                            "text": full_text,
                                            "uploaded_at": datetime.now().isoformat(),
                                            "ocr_settings": {
                                                "lang": lang,
                                                "psm": psm,
                                                "oem": oem,
                                                "contrast": contrast,
                                                "brightness": brightness,
                                                "threshold": threshold,
                                                "use_adaptive": use_adaptive,
                                            },
                                        },
                                    )
                                    
                                    # セッションに追加
                                    st.session_state.documents.append(doc)
                                    
                                    # JSON保存
                                    json_path = OCR_RESULTS_DIR / f"{doc_id}.json"
                                    with open(json_path, "w", encoding="utf-8") as f:
                                        json.dump(doc.meta, f, ensure_ascii=False, indent=2)
                                    
                                    # RAG登録はスキップ（無料枠向け）
                                    st.success("✅ OCR完了")
                                    
                                    # 結果表示（最初の5000文字のみ）
                                    st.subheader("📝 OCR結果")
                                    preview_text = full_text[:5000] + ("..." if len(full_text) > 5000 else "")
                                    st.text_area("抽出されたテキスト（プレビュー）", preview_text, height=400)
                                    st.info(f"全{len(full_text)}文字のテキストが抽出されました。")
                                    
                                    status_text.empty()
                                    progress_bar.empty()
                                    
                                    # メモリ解放
                                    del images, all_text, full_text
                                    import gc
                                    gc.collect()
                            except Exception as e:
                                st.error(f"OCR処理エラー: {e}")
                                status_text.empty()
                                progress_bar.empty()
                except Exception as e:
                    st.error(f"PDF処理エラー: {e}")
        
        # 画像処理
        else:
            try:
                # 画像読み込み
                img = Image.open(uploaded_file)
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                st.subheader("📄 元の画像")
                st.image(img, caption=uploaded_file.name, use_container_width=True)
                
                # 前処理
                processed = preprocess_image(
                    img_array,
                    contrast=contrast,
                    brightness=brightness,
                    threshold=threshold,
                    use_adaptive=use_adaptive,
                )
                
                # 前処理後の画像を表示
                st.subheader("🔧 前処理後の画像")
                st.image(processed, caption="前処理後", use_container_width=True)
                
                # OCR実行
                if st.button("🔍 OCR実行", type="primary"):
                    try:
                        text = perform_ocr(processed, lang=lang, psm=psm, oem=oem)
                        
                        if text:
                            # ドキュメント作成
                            doc_id = now_id()
                            doc = Doc(
                                id=doc_id,
                                filename=uploaded_file.name,
                                text=text,
                                uploaded_at=datetime.now().isoformat(),
                                meta={
                                    "id": doc_id,
                                    "filename": uploaded_file.name,
                                    "text": text,
                                    "uploaded_at": datetime.now().isoformat(),
                                    "ocr_settings": {
                                        "lang": lang,
                                        "psm": psm,
                                        "oem": oem,
                                        "contrast": contrast,
                                        "brightness": brightness,
                                        "threshold": threshold,
                                        "use_adaptive": use_adaptive,
                                    },
                                },
                            )
                            
                            # セッションに追加
                            st.session_state.documents.append(doc)
                            
                            # JSON保存
                            json_path = OCR_RESULTS_DIR / f"{doc_id}.json"
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(doc.meta, f, ensure_ascii=False, indent=2)
                            
                            # RAG登録
                            if CHROMADB_AVAILABLE:
                                ok, msg = rag_add_doc(doc)
                                if ok:
                                    st.success(f"✅ OCR完了: {msg}")
                                else:
                                    st.warning(f"⚠️ OCR完了（RAG登録失敗）: {msg}")
                            else:
                                st.success("✅ OCR完了")
                            
                            # 結果表示
                            st.subheader("📝 OCR結果")
                            st.text_area("抽出されたテキスト", text, height=400)
                        else:
                            st.warning("テキストが抽出されませんでした。前処理設定を調整してください。")
                    except Exception as e:
                        st.error(f"OCRエラー: {e}")
            except Exception as e:
                st.error(f"画像処理エラー: {e}")

# ========== Tab 2: 検索 ==========
with tab_search:
    st.header("🔍 検索（RAGまたは簡易検索）")
    
    if len(st.session_state.documents) == 0:
        st.info("📝 まず「ファイルアップロード」タブからPDF/画像をアップロードしてOCR処理を行ってください。")
    else:
        query = st.text_input("検索クエリ", placeholder="例：プラスチック、材料、加工、温度設定…")
        n_results = st.slider("検索結果数", 1, MAX_SEARCH_RESULTS, DEFAULT_SEARCH_RESULTS)
        
        if st.button("🔍 検索実行", type="primary") and query.strip():
            if CHROMADB_AVAILABLE:
                with st.spinner("RAG検索中..."):
                    results = rag_search(query, n_results=n_results)
                
                if results:
                    st.subheader(f"検索結果（RAG）: {len(results)}件")
                    for i, r in enumerate(results, 1):
                        meta = r.get("metadata") or {}
                        dist = r.get("distance")
                        score = None
                        if isinstance(dist, (int, float)):
                            score = 1.0 / (1.0 + dist)
                        
                        title = f"結果 {i}: {meta.get('filename','(unknown)')} / doc={meta.get('doc_id','')}"
                        if score is not None:
                            title += f" / score≈{score:.3f}"
                        
                        with st.expander(title):
                            st.write("**メタデータ**")
                            st.json(meta)
                            st.write("**該当テキスト（chunk）**")
                            st.text(r.get("text", ""))
                else:
                    st.info("検索結果が見つかりませんでした（RAG）")
            else:
                with st.spinner("簡易検索中..."):
                    hits = simple_search(st.session_state.documents, query, limit=n_results)
                
                if hits:
                    st.subheader(f"検索結果（簡易）: {len(hits)}件")
                    for i, h in enumerate(hits, 1):
                        with st.expander(f"結果 {i}: {h['filename']} / doc={h['doc_id']}"):
                            st.text(h["snippet"])
                else:
                    st.info("検索結果が見つかりませんでした（簡易）")

# ========== Tab 3: 文書一覧 ==========
with tab_list:
    st.header("📚 保存済み文書一覧")
    
    if len(st.session_state.documents) == 0:
        st.info("📝 文書がありません。ファイルアップロードを行ってください。")
    else:
        st.write(f"**文書数: {len(st.session_state.documents)}**")
        
        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc.filename}（{doc.id}）"):
                st.write(f"**uploaded_at:** {doc.uploaded_at}")
                st.write(f"**文字数:** {len(doc.text)}")
                
                # プレビュー
                preview = doc.text[:800] + ("..." if len(doc.text) > 800 else "")
                st.text(preview)
                
                c1, c2, c3 = st.columns([1, 1, 2])
                
                with c1:
                    # JSONダウンロード
                    st.download_button(
                        "⬇️ JSONをダウンロード",
                        data=json.dumps(doc.meta, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"{doc.id}.json",
                        mime="application/json",
                        key=f"dl_{doc.id}",
                    )
                
                with c2:
                    if CHROMADB_AVAILABLE:
                        if st.button("🧠 RAGに登録", key=f"rag_add_{doc.id}"):
                            ok, msg = rag_add_doc(doc)
                            (st.success if ok else st.warning)(msg)
                
                with c3:
                    if st.button("🗑️ 削除", key=f"del_{doc.id}"):
                        # セッションから削除
                        st.session_state.documents = [d for d in st.session_state.documents if d.id != doc.id]
                        
                        # ディスクJSON削除
                        json_path = OCR_RESULTS_DIR / f"{doc.id}.json"
                        if json_path.exists():
                            json_path.unlink()
                        
                        # RAG削除
                        if CHROMADB_AVAILABLE:
                            rag_delete_doc(doc.id)
                        
                        st.success("削除しました")
                        st.rerun()
