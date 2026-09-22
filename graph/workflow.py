"""
LangGraph Workflow - The Writer's Room
========================================
Defines the complete StateGraph workflow for Phase 1.

Nodes:
  - mode_selector_node:   Routes to manual or autonomous path
  - validator_node:        Validates manual scripts
  - scriptwriter_node:     Generates scripts from prompts
  - hitl_node:             Human approval checkpoint
  - character_node:        Extracts character identities
  - image_node:            Generates character images
  - memory_commit_node:    Persists all data to memory

Flow:
  START → mode_selector_node
    ├── (manual) → validator_node → hitl_node → character_node → image_node → memory_commit_node → END
    └── (autonomous) → scriptwriter_node → hitl_node → character_node → image_node → memory_commit_node → END
"""

import json
import os
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from graph.state import WriterRoomState
from agents.scriptwriter import scriptwriter_node
from agents.validator import validator_node
from agents.hitl import hitl_node
from agents.character_designer import character_node
from agents.image_synthesizer import image_node
from mcp.server import get_mcp_registry
import config


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE: Mode Selector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mode_selector_node(state: WriterRoomState) -> Dict[str, Any]:
    """
    Mode Selector Node - Routes to manual or autonomous path.
    Determines the processing mode based on user input.
    """
    print(f"\n{'='*60}")
    print(f"  🔀 Mode Selector Node")
    print(f"{'='*60}")

    mode = state.get("mode", "autonomous")
    print(f"  Selected mode: {mode.upper()}")

    if mode == "manual":
        print(f"  → Routing to: Script Validator (manual pipeline)")
    else:
        print(f"  → Routing to: Scriptwriter (autonomous pipeline)")

    return {
        "mode": mode,
        "current_stage": "mode_selected",
        "status": "in_progress"
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE: Memory Commit (Final)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def memory_commit_node(state: WriterRoomState) -> Dict[str, Any]:
    """
    Memory Commit Node - Final node that persists all generated data.
    Saves scene_manifest.json, character_db.json, and commits to vector store.
    """
    print(f"\n{'='*60}")
    print(f"  💾 Memory Commit Node - Saving All Outputs")
    print(f"{'='*60}")

    mcp = get_mcp_registry()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    errors = []

    # 1. Save scene_manifest.json
    scene_manifest = state.get("scene_manifest")
    if scene_manifest:
        try:
            with open(config.SCENE_MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(scene_manifest, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved: {config.SCENE_MANIFEST_PATH}")

            # Commit to vector memory
            mcp.invoke_tool(
                "commit_memory",
                data=scene_manifest,
                data_type="script_history",
                metadata={"title": scene_manifest.get("title", "Unknown")}
            )
            print(f"  ✅ Script committed to vector memory")
        except Exception as e:
            errors.append(f"Failed to save scene_manifest: {e}")
            print(f"  ❌ Error: {e}")
    else:
        print(f"  ⚠ No scene manifest to save")

    # 2. Save character_db.json
    character_db = state.get("character_db")
    if character_db:
        try:
            with open(config.CHARACTER_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(character_db, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved: {config.CHARACTER_DB_PATH}")

            # Commit to vector memory
            mcp.invoke_tool(
                "commit_memory",
                data=character_db,
                data_type="character_metadata",
                metadata={"total": character_db.get("total_characters", 0)}
            )
            print(f"  ✅ Characters committed to vector memory")
        except Exception as e:
            errors.append(f"Failed to save character_db: {e}")
            print(f"  ❌ Error: {e}")
    else:
        print(f"  ⚠ No character database to save")

    # 3. Log image results
    image_results = state.get("image_results", [])
    if image_results:
        successful_images = [r for r in image_results if r.get("success")]
        print(f"  ✅ {len(successful_images)} character images generated in {config.IMAGE_ASSETS_DIR}")
        for img in successful_images:
            print(f"    → {img.get('character_name', 'Unknown')}: {img.get('image_path', 'N/A')}")
    else:
        print(f"  ⚠ No images generated")

    # Summary
    print(f"\n{'─'*60}")
    print(f"  📊 OUTPUT SUMMARY")
    print(f"{'─'*60}")
    print(f"  Scene Manifest: {'✅' if scene_manifest else '❌'} {config.SCENE_MANIFEST_PATH}")
    print(f"  Character DB:   {'✅' if character_db else '❌'} {config.CHARACTER_DB_PATH}")
    print(f"  Images:         {len(image_results)} generated in {config.IMAGE_ASSETS_DIR}")
    print(f"  Memory Store:   ✅ Committed to ChromaDB")
    print(f"{'─'*60}\n")

    return {
        "memory_committed": True,
        "current_stage": "completed",
        "status": "completed",
        "errors": state.get("errors", []) + errors
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTING FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_by_mode(state: WriterRoomState) -> str:
    """Route based on selected mode: manual or autonomous."""
    mode = state.get("mode", "autonomous")
    if mode == "manual":
        return "validator_node"
    return "scriptwriter_node"


def route_after_hitl(state: WriterRoomState) -> str:
    """Route after human review: continue or end."""
    if state.get("human_approved", False):
        return "character_node"
    return END


def route_after_validation(state: WriterRoomState) -> str:
    """Route after validation: continue if valid, end if failed."""
    if state.get("status") == "failed":
        return END
    return "hitl_node"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD WORKFLOW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_workflow() -> StateGraph:
    """
    Build the complete LangGraph StateGraph for The Writer's Room.
    
    Graph Structure:
      START → mode_selector_node
        ├── (manual) → validator_node → hitl_node → character_node → image_node → memory_commit_node → END
        └── (autonomous) → scriptwriter_node → hitl_node → character_node → image_node → memory_commit_node → END
    """
    print("\n╔══════════════════════════════════════════╗")
    print("║   Building LangGraph Workflow             ║")
    print("╚══════════════════════════════════════════╝\n")

    workflow = StateGraph(WriterRoomState)

    # ──── Add Nodes ────
    workflow.add_node("mode_selector_node", mode_selector_node)
    workflow.add_node("validator_node", validator_node)
    workflow.add_node("scriptwriter_node", scriptwriter_node)
    workflow.add_node("hitl_node", hitl_node)
    workflow.add_node("character_node", character_node)
    workflow.add_node("image_node", image_node)
    workflow.add_node("memory_commit_node", memory_commit_node)

    print("  [Graph] Added 7 nodes")

    # ──── Set Entry Point ────
    workflow.set_entry_point("mode_selector_node")
    print("  [Graph] Entry point: mode_selector_node")

    # ──── Add Conditional Edges ────
    # Mode selector routes to either validator or scriptwriter
    workflow.add_conditional_edges(
        "mode_selector_node",
        route_by_mode,
        {
            "validator_node": "validator_node",
            "scriptwriter_node": "scriptwriter_node"
        }
    )
    print("  [Graph] Edge: mode_selector → [validator | scriptwriter]")

    # Validator routes to HITL or END on failure
    workflow.add_conditional_edges(
        "validator_node",
        route_after_validation,
        {
            "hitl_node": "hitl_node",
            END: END
        }
    )
    print("  [Graph] Edge: validator → [hitl | END]")

    # Scriptwriter always goes to HITL
    workflow.add_edge("scriptwriter_node", "hitl_node")
    print("  [Graph] Edge: scriptwriter → hitl")

    # HITL routes to character node or END
    workflow.add_conditional_edges(
        "hitl_node",
        route_after_hitl,
        {
            "character_node": "character_node",
            END: END
        }
    )
    print("  [Graph] Edge: hitl → [character | END]")

    # Sequential: character → image → memory_commit → END
    workflow.add_edge("character_node", "image_node")
    workflow.add_edge("image_node", "memory_commit_node")
    workflow.add_edge("memory_commit_node", END)
    print("  [Graph] Edge: character → image → memory_commit → END")

    print("\n  [Graph] Workflow built successfully ✅\n")

    return workflow


def compile_workflow():
    """Build and compile the workflow for execution."""
    workflow = build_workflow()
    return workflow.compile()
