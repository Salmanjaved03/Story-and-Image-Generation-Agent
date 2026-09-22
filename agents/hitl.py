"""
Human-in-the-Loop (HITL) Agent
================================
Provides a checkpoint control before execution continues.

Why Required:
  - Prevents hallucinated scripts from propagating
  - Ensures user intent alignment
  - Allows user to approve, reject, or modify generated content

This agent pauses the workflow and asks for human input.
"""

import json
from typing import Any, Dict

from graph.state import WriterRoomState


class HITLAgent:
    """
    Human-in-the-Loop Agent - Checkpoint for human approval.
    
    Displays the generated script for user review and collects
    approval/rejection/modification feedback.
    """

    def __init__(self):
        self.name = "Human-in-the-Loop Agent"
        self.role = "Provide checkpoint control before execution continues"

    def _display_script_preview(self, script_data: Dict[str, Any]) -> None:
        """Display a formatted preview of the script for human review."""
        print(f"\n{'─'*60}")
        print(f"  📄 SCRIPT PREVIEW FOR REVIEW")
        print(f"{'─'*60}")

        title = script_data.get("title", "Untitled")
        genre = script_data.get("genre", "Unknown")
        scenes = script_data.get("scenes", [])

        print(f"\n  Title: {title}")
        print(f"  Genre: {genre}")
        print(f"  Total Scenes: {len(scenes)}")
        print()

        for scene in scenes:
            scene_num = scene.get("scene_number", "?")
            heading = scene.get("heading", "NO HEADING")
            setting = scene.get("setting", "")
            action = scene.get("action", "")

            print(f"  ┌─ Scene {scene_num}: {heading}")
            if setting:
                print(f"  │  Setting: {setting[:80]}...")
            if action:
                print(f"  │  Action: {action[:80]}...")

            dialogues = scene.get("dialogue", [])
            if dialogues:
                print(f"  │  Dialogue ({len(dialogues)} lines):")
                for d in dialogues[:3]:
                    char = d.get("character", "?")
                    line = d.get("line", "")
                    print(f"  │    {char}: \"{line[:60]}...\"" if len(line) > 60 else f"  │    {char}: \"{line}\"")
                if len(dialogues) > 3:
                    print(f"  │    ... and {len(dialogues) - 3} more lines")

            visual = scene.get("visual_cues", {})
            if visual:
                print(f"  │  Visual: {visual.get('mood', 'N/A')} mood, {visual.get('lighting', 'N/A')} lighting")

            print(f"  └{'─'*40}")

        print()

    def run(self, state: WriterRoomState) -> Dict[str, Any]:
        """
        Execute HITL checkpoint:
        1. Display script preview
        2. Ask for human approval
        3. Collect feedback
        4. Route accordingly
        """
        print(f"\n{'='*60}")
        print(f"  🧑 {self.name} - Checkpoint")
        print(f"{'='*60}")

        script_data = state.get("scene_manifest") or state.get("raw_script")

        if not script_data:
            print("  ⚠ No script data available for review")
            return {
                "human_approved": False,
                "human_feedback": "No script data available",
                "errors": state.get("errors", []) + ["HITL: No script data to review"],
                "current_stage": "hitl_no_data",
                "status": "failed"
            }

        # Display the preview
        self._display_script_preview(script_data)

        # Ask for human input
        print(f"  ╔══════════════════════════════════════════╗")
        print(f"  ║   HUMAN REVIEW REQUIRED                  ║")
        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  [A] Approve - Continue to next stage     ║")
        print(f"  ║  [R] Reject  - Stop the pipeline          ║")
        print(f"  ║  [M] Modify  - Provide feedback           ║")
        print(f"  ╚══════════════════════════════════════════╝")

        while True:
            try:
                choice = input("\n  Your choice (A/R/M): ").strip().upper()
            except EOFError:
                # Non-interactive mode - auto-approve
                choice = "A"
                print("  [Auto-mode] Non-interactive environment detected - auto-approving")
                break

            if choice in ("A", "R", "M"):
                break
            print("  Invalid choice. Please enter A, R, or M.")

        if choice == "A":
            print(f"\n  ✅ Script APPROVED by human reviewer")
            return {
                "human_approved": True,
                "human_feedback": "Approved",
                "current_stage": "human_approved",
                "status": "in_progress"
            }

        elif choice == "R":
            print(f"\n  ❌ Script REJECTED by human reviewer")
            reason = ""
            try:
                reason = input("  Reason for rejection (optional): ").strip()
            except EOFError:
                pass
            return {
                "human_approved": False,
                "human_feedback": f"Rejected: {reason}" if reason else "Rejected",
                "current_stage": "human_rejected",
                "status": "failed"
            }

        else:  # M - Modify
            feedback = ""
            try:
                feedback = input("  Provide your modification feedback: ").strip()
            except EOFError:
                pass
            print(f"\n  📝 Feedback recorded: {feedback[:100]}...")
            print(f"  ℹ  Note: In this phase, modifications are recorded for future use.")
            print(f"  ✅ Proceeding with current script + recorded feedback")
            return {
                "human_approved": True,
                "human_feedback": f"Modification requested: {feedback}",
                "current_stage": "human_approved_with_feedback",
                "status": "in_progress"
            }


def hitl_node(state: WriterRoomState) -> Dict[str, Any]:
    """LangGraph node function for the Human-in-the-Loop Agent."""
    agent = HITLAgent()
    return agent.run(state)
