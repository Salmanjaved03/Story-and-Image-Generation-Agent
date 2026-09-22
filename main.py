"""
PROJECT MONTAGE - Phase 1: The Writer's Room
==============================================
Main Entry Point

Autonomous Story and Image Generation Layer

This module operates as a multi-agent creative system where
specialized agents collaborate autonomously using:
  - MCP-based tool discovery
  - LangGraph stateful workflows
  - Shared persistent memory (ChromaDB)

Two modes:
  1. Manual:     User provides a script for validation
  2. Autonomous: User provides a prompt for LLM generation

Outputs:
  - outputs/scene_manifest.json  → Structured screenplay
  - outputs/character_db.json    → Character identity store
  - outputs/image_assets/        → AI-generated character visuals
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from mcp.tools import register_all_tools
from graph.workflow import compile_workflow


def print_banner():
    """Print the application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🎬  PROJECT MONTAGE - Phase 1                              ║
║   ─────────────────────────────                               ║
║   THE WRITER'S ROOM                                           ║
║   Autonomous Story and Image Generation Layer                 ║
║                                                               ║
║   Multi-Agent System:                                         ║
║     • Scriptwriter Agent      → Script generation             ║
║     • Script Validator Agent  → Structure validation          ║
║     • Human-in-the-Loop       → Approval checkpoint          ║
║     • Character Designer      → Identity extraction           ║
║     • Image Synthesizer       → Visual generation             ║
║                                                               ║
║   Architecture:                                               ║
║     • MCP-based tool discovery (no hardcoded APIs)            ║
║     • LangGraph stateful workflows                            ║
║     • ChromaDB persistent memory                              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")


def get_sample_scripts():
    """Return sample scripts for manual mode testing."""
    return {
        "1": {
            "name": "The Last Coffee Shop",
            "script": """INT. COFFEE SHOP - MORNING

A cozy, old-fashioned coffee shop. Morning light streams through dusty windows.

ELENA (30s, tired but determined)
sits at the counter, nursing a cold cup of coffee.

ELENA
(to herself)
This is the last day. After today, it's all gone.

The door CHIMES. MARCO (40s, well-dressed, carrying a briefcase) enters.

MARCO
Elena? I thought I'd find you here.

ELENA
(not looking up)
Where else would I be? This place is all I have left.

MARCO
That's exactly why I'm here. I have a proposition.

EXT. COFFEE SHOP - SAME TIME

Through the window, we see Elena and Marco talking. A FOR SALE sign hangs on the door.

A YOUNG WOMAN (20s) stops to look at the sign, then peers inside.

INT. COFFEE SHOP - CONTINUOUS

ELENA
You want to buy this place? You don't even drink coffee.

MARCO
(smiling)
No. I want to invest. There's a difference.

ELENA
(skeptical)
And what do you get out of it?

MARCO
A second chance. For both of us.

Elena looks at him for a long moment, then slowly pushes her cup aside.

ELENA
Alright. Tell me your plan.
"""
        },
        "2": {
            "name": "Midnight Signal",
            "script": """INT. RADIO STATION - NIGHT

A dimly lit, cluttered radio station. Vintage equipment everywhere.

DJ SARAH (late 20s, headphones around neck, punk aesthetic)
leans into the microphone.

SARAH
You're listening to The Midnight Signal, broadcasting to anyone still awake 
and anyone who needs to hear this. Tonight's topic: messages we never sent.

She presses a button. A CALLER's voice crackles through.

CALLER (V.O.)
Sarah? I... I've been listening every night for three months.

SARAH
Welcome to the show. What's your unsent message?

CALLER (V.O.)
(long pause)
I want to tell my father I forgive him. But he died last year.

Sarah removes her headphones slowly, affected by the words.

SARAH
(softly, off-mic)
Yeah. I know how that feels.

She composes herself, puts headphones back on.

SARAH
(on-mic)
Sometimes the most important messages are the ones we say 
out loud, even when no one's listening. Especially then.

EXT. CITY SKYLINE - NIGHT

The radio tower blinks red against the dark sky. Below, apartment windows 
glow one by one as people tune in.

INT. APARTMENT - NIGHT

A TEENAGER sits by the window, radio playing softly.

TEENAGER
(whispering)
Mom... I'm sorry I didn't say goodbye.
"""
        }
    }


def run_autonomous_mode():
    """Run the autonomous script generation mode."""
    print("\n  ═══ AUTONOMOUS MODE ═══")
    print("  The Scriptwriter Agent will generate a screenplay from your prompt.\n")

    # Get user prompt
    print("  Enter your story concept/prompt:")
    print("  (Or press Enter for a default prompt)\n")

    try:
        prompt = input("  📝 Your prompt: ").strip()
    except EOFError:
        prompt = ""

    if not prompt:
        prompt = (
            "A mysterious lighthouse keeper discovers that the light signals "
            "she sends every night are being answered by someone — or something — "
            "from across a sea that all maps say is empty. As she investigates, "
            "she uncovers a hidden truth about her own past."
        )
        print(f"\n  Using default prompt: {prompt[:80]}...")

    # Get genre
    try:
        genre_input = input("\n  🎭 Genre (drama/thriller/comedy/sci-fi/fantasy) [drama]: ").strip().lower()
    except EOFError:
        genre_input = ""
    genre = genre_input if genre_input else "drama"

    # Get number of scenes
    try:
        scenes_input = input("  🎬 Number of scenes (2-6) [3]: ").strip()
    except EOFError:
        scenes_input = ""
    try:
        num_scenes = int(scenes_input) if scenes_input else 3
        num_scenes = max(2, min(6, num_scenes))
    except ValueError:
        num_scenes = 3

    return {
        "mode": "autonomous",
        "user_input": prompt,
        "genre": genre,
        "num_scenes": num_scenes,
        "human_approved": False,
        "memory_committed": False,
        "errors": [],
        "status": "in_progress",
        "current_stage": "initialized"
    }


def run_manual_mode():
    """Run the manual script injection mode."""
    print("\n  ═══ MANUAL MODE ═══")
    print("  Provide a script for validation and processing.\n")

    samples = get_sample_scripts()

    print("  Available sample scripts:")
    for key, sample in samples.items():
        print(f"    [{key}] {sample['name']}")
    print(f"    [C] Enter custom script")

    try:
        choice = input("\n  Your choice: ").strip().upper()
    except EOFError:
        choice = "1"

    if choice in samples:
        script_text = samples[choice]["script"]
        print(f"\n  Using sample script: {samples[choice]['name']}")
    elif choice == "C":
        print("\n  Enter your script (type 'END' on a new line to finish):")
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
        except EOFError:
            pass
        script_text = "\n".join(lines)
    else:
        # Default to first sample
        script_text = samples["1"]["script"]
        print(f"\n  Using default sample: {samples['1']['name']}")

    return {
        "mode": "manual",
        "user_input": script_text,
        "genre": "drama",
        "num_scenes": 3,
        "human_approved": False,
        "memory_committed": False,
        "errors": [],
        "status": "in_progress",
        "current_stage": "initialized"
    }


def main():
    """Main entry point for PROJECT MONTAGE Phase 1."""
    print_banner()

    # ──── Step 1: Register MCP Tools ────
    print("  Step 1: Initializing MCP Tool Registry...")
    register_all_tools()

    # ──── Step 2: Build LangGraph Workflow ────
    print("  Step 2: Building LangGraph Workflow...")
    app = compile_workflow()

    # ──── Step 3: Select Mode ────
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   Select Processing Mode                 ║")
    print("  ╠══════════════════════════════════════════╣")
    print("  ║  [1] Autonomous - Generate from prompt   ║")
    print("  ║  [2] Manual     - Provide a script       ║")
    print("  ╚══════════════════════════════════════════╝")

    try:
        mode_choice = input("\n  Your choice (1/2) [1]: ").strip()
    except EOFError:
        mode_choice = "1"

    if mode_choice == "2":
        initial_state = run_manual_mode()
    else:
        initial_state = run_autonomous_mode()

    # ──── Step 4: Execute Workflow ────
    print(f"\n{'═'*60}")
    print(f"  🚀 EXECUTING LANGGRAPH WORKFLOW")
    print(f"  Mode: {initial_state['mode'].upper()}")
    print(f"{'═'*60}\n")

    try:
        # Run the compiled workflow
        final_state = app.invoke(initial_state)

        # ──── Step 5: Display Results ────
        print(f"\n{'═'*60}")
        print(f"  🏁 WORKFLOW COMPLETE")
        print(f"{'═'*60}")
        print(f"\n  Status: {final_state.get('status', 'unknown')}")
        print(f"  Stage:  {final_state.get('current_stage', 'unknown')}")

        if final_state.get("errors"):
            print(f"\n  Errors:")
            for err in final_state["errors"]:
                print(f"    ✗ {err}")

        # Check outputs
        if os.path.exists(config.SCENE_MANIFEST_PATH):
            print(f"\n  📄 Scene Manifest: {config.SCENE_MANIFEST_PATH}")
        if os.path.exists(config.CHARACTER_DB_PATH):
            print(f"  👤 Character DB:   {config.CHARACTER_DB_PATH}")
        if os.path.exists(config.IMAGE_ASSETS_DIR):
            images = os.listdir(config.IMAGE_ASSETS_DIR)
            print(f"  🖼️  Images:         {len(images)} files in {config.IMAGE_ASSETS_DIR}")

        print(f"\n  {'✅ Pipeline completed successfully!' if final_state.get('status') == 'completed' else '⚠ Pipeline ended with issues.'}")

    except KeyboardInterrupt:
        print("\n\n  ⚠ Workflow interrupted by user.")
    except Exception as e:
        print(f"\n  ❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
