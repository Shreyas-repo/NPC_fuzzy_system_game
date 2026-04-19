# NPC-FUZZY-SYSTEM

## Summary

NPC-FUZZY-SYSTEM is a simple game framework for creating non-player characters (NPCs) that behave in a realistic, flexible way. It uses a fuzzy logic-style system and AI-style rules to let NPCs make decisions based on their needs, emotions, memories, and environment.

## How it works

- NPCs have internal states like mood, needs, and social memory.
- The game checks what each NPC wants or needs at each moment.
- NPC behavior is chosen from multiple options instead of one fixed action, so NPCs can react differently depending on the situation.
- NPCs also learn from interactions and adjust over time, making their behavior feel more natural.

## Workflow

1. The game starts and loads NPC data, world state, and behavior rules.
2. Each update loop, NPCs evaluate their current needs and what is happening around them.
3. The system scores possible actions using fuzzy-style logic and selects the best action.
4. NPCs perform actions, update their internal state, and record important events.
5. The game uses those updates to shape future decisions, so NPCs can adapt.

This makes NPC-FUZZY-SYSTEM a good fit for games where NPCs should feel alive and unpredictable instead of following rigid scripts.