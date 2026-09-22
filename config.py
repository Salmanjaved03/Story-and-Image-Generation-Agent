"""
Configuration for PROJECT MONTAGE - Phase 1: The Writer's Room
Replace placeholder API keys with your actual keys.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Configuration (Groq) ───────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_YOUR_GROQ_API_KEY_HERE")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── Image Generation (HuggingFace) ─────────────────────────────────────────
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "hf_YOUR_HUGGINGFACE_API_KEY_HERE")
HUGGINGFACE_MODEL = os.getenv(
    "HUGGINGFACE_IMAGE_MODEL",
    "stabilityai/stable-diffusion-2-1"
)
HUGGINGFACE_API_URL = f"https://router.huggingface.co/hf-inference/models/{HUGGINGFACE_MODEL}"

# Fallback models to try if primary model fails
HUGGINGFACE_FALLBACK_MODELS = [
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
]

# ─── Memory Configuration (ChromaDB) ────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./memory_store")
CHROMA_COLLECTION = "montage_phase1"

# ─── Output Paths ───────────────────────────────────────────────────────────
OUTPUT_DIR = "./outputs"
IMAGE_ASSETS_DIR = os.path.join(OUTPUT_DIR, "image_assets")
SCENE_MANIFEST_PATH = os.path.join(OUTPUT_DIR, "scene_manifest.json")
CHARACTER_DB_PATH = os.path.join(OUTPUT_DIR, "character_db.json")

# ─── MCP Server Configuration ───────────────────────────────────────────────
MCP_SERVER_HOST = "localhost"
MCP_SERVER_PORT = 5100
