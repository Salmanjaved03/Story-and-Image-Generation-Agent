"""
LangGraph Shared State Definition
===================================
Defines the shared state that flows through all nodes in the
LangGraph workflow. This is the central data structure agents
read from and write to.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class WriterRoomState(TypedDict, total=False):
    """
    Shared state for the Writer's Room LangGraph workflow.
    
    All agents interact via this shared state — no direct agent-to-agent calls.
    """

    # ─── Input ────────────────────────────────────────────────────────────
    mode: str                        # "manual" or "autonomous"
    user_input: str                  # Raw prompt or script text from user
    genre: str                       # Genre for script generation
    num_scenes: int                  # Number of scenes to generate

    # ─── Script Processing ────────────────────────────────────────────────
    raw_script: Optional[Dict[str, Any]]           # Parsed/generated script data
    validation_result: Optional[Dict[str, Any]]    # Validation output (manual mode)
    scene_manifest: Optional[Dict[str, Any]]       # Final structured screenplay

    # ─── Character Processing ─────────────────────────────────────────────
    character_db: Optional[Dict[str, Any]]         # Character identity store
    image_results: Optional[List[Dict[str, Any]]]  # Generated image info

    # ─── Human-in-the-Loop ────────────────────────────────────────────────
    human_approved: bool             # Whether human approved the script
    human_feedback: Optional[str]    # Feedback from human review

    # ─── Memory ───────────────────────────────────────────────────────────
    memory_committed: bool           # Whether data was committed to memory

    # ─── Status ───────────────────────────────────────────────────────────
    current_stage: str               # Current processing stage
    errors: List[str]                # Error messages
    status: str                      # "in_progress", "completed", "failed"
