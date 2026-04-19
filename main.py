#!/usr/bin/env python3
"""
Village of Minds 
Controls:
    WASD        — Move player
    E           — Talk to nearby NPC
    F           — Push nearby NPC
    TAB         — Toggle spectator mode
    ESC         — Pause / Close chat
    +/-         — Zoom (spectator mode)
    1-5         — Time speed (spectator mode)
    R           — Start raid (spectator mode)
    Mouse Click — Follow NPC (spectator mode)
    Left Click  — Sword slash (play mode)
    ENTER       — Send chat message
    T           — Plead for mercy when detained
    1/2/3       — Present your point during trial
    SPACE       — Sword slash (defend against raids)

Core Systems Used:
    - K-Means Clustering: Groups NPCs by behavior vectors
    - DBSCAN: Detects spatial gatherings
    - Rule-based sentiment: Local chat affect inference
    - Fuzzy controller: Explainable action suggestions
    - Self-Organizing routine adaptation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.engine import GameEngine


def main():
    """Launch the Village of Minds simulation."""
    print("=" * 55)
    print("  Village of Minds — NPC Simulation")
    print("=" * 55)
    print()
    print("  Controls:")
    print("    WASD        — Move player")
    print("    E           — Talk to nearby NPC")
    print("    F           — Push nearby NPC")
    print("    TAB         — Toggle spectator mode")
    print("    ESC         — Pause / Close chat")
    print("    +/-         — Zoom (spectator mode)")
    print("    1-5         — Time speed (spectator mode)")
    print("    R           — Start raid (spectator mode)")
    print("    Mouse Click — Follow NPC (spectator mode)")
    print("    Left Click  — Sword slash (play mode)")
    print("    ENTER       — Send chat message")
    print("    T           — Plead for mercy when detained")
    print("    1/2/3       — Present your point during trial")
    print("    SPACE       — Sword slash (defend against raids)")
    print()
    print("  Core Systems:")
    print("    • K-Means Clustering (behavior groups)")
    print("    • DBSCAN (spatial gatherings)")
    print("    • Local Sentiment Rules (chat affect)")
    print("    • Fuzzy Action Controller (explainable decisions)")
    print("    • Routine Adaptation (schedule learning)")
    print()
    print("  Starting simulation...")
    print()

    engine = GameEngine()
    engine.run()


if __name__ == "__main__":
    main()
