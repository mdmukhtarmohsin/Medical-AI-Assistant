import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the application"""
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./documents"))
    FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "./faiss_index"))
    LOGS_DIR = Path(os.getenv("LOGS_DIR", "./logs"))
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    RAGAS_FAITHFULNESS_THRESHOLD = float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.90"))
    RAGAS_CONTEXT_PRECISION_THRESHOLD = float(os.getenv("RAGAS_CONTEXT_PRECISION_THRESHOLD", "0.85"))


def setup_logging():
    """Setup logging configuration"""
    Config.LOGS_DIR.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOGS_DIR / 'app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def log_ragas_metrics(document_id: str, question: str, answer: str, metrics: Dict[str, float]):
    """Log RAGAS metrics to JSONL file"""
    metrics_log_path = Config.LOGS_DIR / "metrics_log.jsonl"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "document_id": document_id,
        "question": question,
        "answer": answer,
        "metrics": metrics
    }
    
    with open(metrics_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def generate_document_id(filename: str) -> str:
    """Generate unique document ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    return f"{timestamp}_{clean_filename}"


def ensure_directories():
    """Ensure all necessary directories exist"""
    Config.UPLOAD_DIR.mkdir(exist_ok=True)
    Config.FAISS_INDEX_DIR.mkdir(exist_ok=True)
    Config.LOGS_DIR.mkdir(exist_ok=True)


def check_ragas_thresholds(metrics: Dict[str, float]) -> tuple[bool, str]:
    """Check if RAGAS metrics meet safety thresholds"""
    warnings = []
    
    if metrics.get("faithfulness", 0) < Config.RAGAS_FAITHFULNESS_THRESHOLD:
        warnings.append(f"Low faithfulness score: {metrics['faithfulness']:.3f}")
    
    if metrics.get("context_precision", 0) < Config.RAGAS_CONTEXT_PRECISION_THRESHOLD:
        warnings.append(f"Low context precision: {metrics['context_precision']:.3f}")
    
    if warnings:
        warning_msg = "This answer may not be reliable. " + "; ".join(warnings)
        return False, warning_msg
    
    return True, "" 