***

# Intelligent NPC Interaction System

This project implements an intelligent NPC interaction system combining NLP, Fuzzy Logic, Neural Networks, Markov Chains, Evolutionary Optimization, and Social Simulation to create dynamic, adaptive and realistic conversations between players and NPC characters.

The system allows NPCs to:
* understand player sentiment
* adapt tone and behaviour
* generate context-aware dialogue
* learn from previous conversations
* evolve behaviour over time
* influence each other socially

---

## 🏗️ High Level Architecture

The system follows a hybrid AI pipeline:


    A[Player Message] --> B[Sentiment Analysis NLP]
    B --> C[Topic Detection Fuzzy NLP]
    C --> D[Neural Network Behaviour Bias]
    D --> E[Fuzzy Logic Decision System]
    E --> F[Dialogue Generation Dictionary + Markov Variation]
    F --> G[NPC Emotional State Update]
    G --> H[Social Interaction Learning Clustering]
    H --> I[Research Metrics Evaluation]
    I --> J[Evolutionary Optimization of Fuzzy Weights]

Each component contributes to making NPC behaviour adaptive and realistic.

---

## 🧠 Core Components

### 1. Sentiment Analysis Engine
The player’s message is analyzed using:
* TextBlob sentiment model (if available)
* fallback keyword-based sentiment scoring

Output features include:
* sentiment score (-1 to 1)
* emotional intensity
* danger score
* positive/negative keyword hits

> **Example:**
> "You are useless"
> → sentiment = -0.8
> → intensity = 0.25
> → danger_score = 0.2
> These values influence NPC mood and response tone.

### 2. Topic Detection using Fuzzy NLP
Keywords are mapped to conversation topics:
* "work" → topic_work
* "family" → topic_family
* "village" → topic_village
* "trade" → topic_economy

> **Example:**
> Player: "How is work today?"
> topic_work = 0.9
> Topic scores guide dialogue selection and behaviour decisions.

### 3. Neural Dialogue Behaviour Model
A neural network predicts behavioural biases using features such as:
* sentiment
* danger level
* topic score
* trust
* mood
* NPC needs

Output includes:
* warmth bias
* sentiment adjustment
* action preference bias

> **Example:**
> negative player tone → reduced warmth
> friendly player tone → increased warmth
> This allows NPC personalities to adapt dynamically.

### 4. Fuzzy Logic Behaviour Controller
Fuzzy logic converts numeric inputs into behavioural decisions.
Inputs include:
* hunger, energy, social_need, trust, mood, danger, topic score, sentiment

> **Example fuzzy rules:**
> IF hunger high → eat
> IF energy low → sleep
> IF trust high → socialize
> IF danger high → flee
> IF topic_work high → discuss work

Each possible action receives a score (eat, sleep, socialize, work, flee, guard). The highest scoring action determines NPC behaviour.

### 5. Dialogue Generation Engine
Dialogue responses are generated using a hybrid approach:

**A. Dialogue Templates (Primary Source)**
Each NPC class has dialogue sets (Royal, Merchant, Guard, Traveller, Labourer, Peasant, Blacksmith, Noble). Each class contains:
* positive responses
* neutral responses
* negative responses
* low trust responses

**B. Markov Chain Language Variation**
A Markov model learns word transitions from previous conversations. It introduces variation in phrasing to prevent repetitive dialogue.

**C. Dialogue Memory System**
NPCs store conversation history (`npc.dialogue_history`). Memory enables:
* context-aware responses, reduced repetition, multi-turn conversations, adaptive trust changes

**D. Dynamic Response Composition**
Final response is constructed using layered components:
* Base response + tone clause + optional history clause + optional trust/mood clause.

### 6. NPC Emotional State Update
Each interaction updates NPC internal state:
* trust increases after positive interaction
* trust decreases after rude interaction
* mood changes depending on conversation tone
* NPC personality evolves over time.

### 7. Social Interaction Learning (Unsupervised Learning)
NPC behaviour vectors are embedded and clustered using similarity measures.
* **Example behaviour vector:** [mood, trust, energy, hunger, social_need]
* Similar NPCs influence each other: happy NPC clusters increase group mood; fearful NPC clusters increase defensive behaviour. This simulates social dynamics.

### 8. Research Metrics System
The system tracks performance metrics (average mood, average trust, conflict rate, social stability, chat latency). Metrics are stored for analysis in `research_metrics.jsonl`.

### 9. Evolutionary Policy Optimization
Fuzzy controller weights are optimized using a genetic algorithm.
* **Weights control importance of actions:** eat weight, sleep weight, socialize weight, work weight, flee weight, guard weight.
* **Evolution process:** generate population of weight sets → evaluate using research metrics → select best performing weights → apply crossover and mutation → produce improved policy.

---

## 🔄 End-to-End Example Flow
**Player input:** "You are terrible at your job"

**Processing steps:**
1. sentiment analysis detects negative tone
2. topic detection identifies "work"
3. neural network reduces warmth bias
4. fuzzy logic prioritizes defensive behaviour
5. dialogue generator selects negative merchant response
6. Markov model adds wording variation
7. tone clause added
8. trust decreases
9. conversation stored in memory
10. behaviour model updated
11. evolutionary optimizer adjusts policy

**Final response:**
"I won't tolerate rude customers.
That was harsh.
Let's keep this civil."

---

## ✨ Key Features
* hybrid AI architecture
* adaptive NPC personalities
* memory-driven dialogue
* emotion-aware responses
* dynamic behaviour evolution
* realistic social simulation
* non-repetitive conversations
