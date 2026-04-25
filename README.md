A hybrid, adaptive AI framework designed to create dynamic, realistic, and evolving interactions between players and NPCs. This system fuses Natural Language Processing (NLP), Fuzzy Logic, Neural Networks, Markov Chains, and Evolutionary Algorithms to move beyond static dialogue trees.

🚀 Key Features
Hybrid AI Architecture: Combines symbolic and sub-symbolic AI for robust decision-making.

Adaptive Personalities: NPCs adjust their behavior based on sentiment, trust, and context.

Memory-Driven Dialogue: Multi-turn conversation awareness with non-repetitive phrasing.

Dynamic Evolution: Uses Evolutionary Optimization to fine-tune behavioral policies over time.

Social Simulation: NPCs influence one another, creating emergent group behaviors.

🏗️ High-Level Architecture
The system follows a sequential pipeline to process player input and generate context-aware responses:

Player Message

Sentiment Analysis (NLP)

Topic Detection (Fuzzy NLP)

Neural Network Behaviour Bias

Fuzzy Logic Decision System

Dialogue Generation (Dictionary + Markov)

NPC Emotional State Update

Social Interaction Learning (Clustering)

Research Metrics Evaluation

Evolutionary Optimization

🧠 Core Components
1. Sentiment Analysis Engine
Analyzes the player's emotional tone and intensity to influence NPC mood.

Tools: TextBlob & keyword-based fallback.

Outputs: Sentiment score (-1 to 1), emotional intensity, and danger metrics.

2. Topic Detection (Fuzzy NLP)
Maps player input to specific domains (e.g., topic_work, topic_family, topic_economy) using fuzzy matching to guide dialogue focus.

3. Neural Dialogue Behaviour Model
A predictive network that processes current context (trust, needs, mood) to determine behavioral biases like warmth and action preference.

4. Fuzzy Logic Behaviour Controller
Converts complex human-like emotional inputs into concrete NPC actions.

Logic: Uses fuzzy rules (e.g., IF hunger high THEN eat) to score possible actions such as flee, work, or socialize.

5. Dialogue Generation Engine
A three-layered approach for realistic speech:

Templates: Class-based dialogue sets (Merchant, Noble, Guard, etc.).

Markov Chains: Dynamically varies phrasing to prevent repetitive, robotic responses.

Memory System: Stores interaction history to facilitate context-aware, multi-turn conversations.

6. NPC Emotional & Social State
Emotional Update: Trust and mood fluctuate based on interaction quality.

Clustering: Unsupervised learning identifies behavioral clusters; happy NPCs can boost group morale, while fearful clusters increase defensive posture.

7. Evolutionary Policy Optimization
Uses Genetic Algorithms to continuously improve the NPC "brain."

Process: A population of fuzzy weight sets is tested; the best performers are selected, mutated, and crossed over to ensure the simulation becomes more realistic over time.

📊 Research & Metrics
The system logs performance metrics to research_metrics.jsonl, tracking:

Conflict Rates & Social Stability

Average Mood & Trust

Chat Latency
