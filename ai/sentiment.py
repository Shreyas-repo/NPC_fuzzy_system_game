"""
Sentiment analysis for player chat input and NPC dialogue response generation.
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis.
"""
import random
import re

# Try to import VADER; fall back to simple keyword-based analysis
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False


class SentimentAnalyzer:
    """Analyzes player chat input and generates NPC responses."""

    # Keyword-based fallback sentiment
    POSITIVE_WORDS = {
        "hello", "hi", "hey", "good", "great", "nice", "love", "like", "thank",
        "thanks", "please", "help", "friend", "happy", "wonderful", "amazing",
        "excellent", "beautiful", "kind", "brave", "honor", "respect", "peace",
        "welcome", "greetings", "bless", "well", "fine", "trade", "buy", "sell",
        "yes", "agree", "praise", "gift", "generous", "smile", "care", "protect"
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "hate", "ugly", "stupid", "fool", "die", "kill",
        "fight", "war", "attack", "steal", "thief", "liar", "cheat", "leave",
        "go away", "shut", "damn", "curse", "threat", "destroy", "burn",
        "no", "never", "worst", "weak", "coward", "betray", "tax", "poverty",
        "starve", "suffer", "pain", "angry", "mad", "furious", "disgusting"
    }

    DANGEROUS_WORDS = {
        "kill", "murder", "stab", "slash", "attack", "hurt", "burn", "destroy",
        "rob", "steal", "loot", "threat", "die", "fight", "weapon", "blood",
        "kidnap", "poison", "bomb",
    }

    CALMING_WORDS = {
        "sorry", "apologize", "peace", "calm", "forgive", "please", "thanks", "help",
        "safe", "protect", "trust", "friend",
    }

    def analyze(self, text):
        """
        Analyze sentiment of text. Returns a score from -1.0 to 1.0.
        Positive = friendly, Negative = hostile, 0 = neutral.
        """
        if not text.strip():
            return 0.0

        if HAS_TEXTBLOB:
            blob = TextBlob(text)
            return max(-1.0, min(1.0, blob.sentiment.polarity))
        else:
            return self._keyword_sentiment(text)

    def analyze_with_features(self, text):
        """Return sentiment and intent-like features used by local fuzzy dialogue rules."""
        lowered = text.lower()
        words = [w.strip(".,!?;:'\"()[]{}") for w in lowered.split()]

        sentiment = self.analyze(text)
        pos_hits = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_hits = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        danger_hits = sum(1 for w in words if w in self.DANGEROUS_WORDS)
        calm_hits = sum(1 for w in words if w in self.CALMING_WORDS)

        intensity = min(1.0, (text.count("!") * 0.15) + (len(words) / 45.0))
        if any(w.isupper() and len(w) > 2 for w in text.split()):
            intensity = min(1.0, intensity + 0.2)

        danger_score = min(1.0, (danger_hits * 0.35) + max(0.0, -sentiment) * 0.35 + intensity * 0.2 - calm_hits * 0.1)

        return {
            "sentiment": sentiment,
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "danger_hits": danger_hits,
            "calm_hits": calm_hits,
            "danger_score": max(0.0, danger_score),
            "intensity": intensity,
            "keywords": [w for w in words if w in self.POSITIVE_WORDS or w in self.NEGATIVE_WORDS or w in self.DANGEROUS_WORDS],
        }

    def _keyword_sentiment(self, text):
        """Simple keyword-based sentiment analysis."""
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total


class DialogueGenerator:
    """Generates NPC dialogue responses based on sentiment, mood, and class."""

    # Dialogue phrase banks organized by NPC class and sentiment
    DIALOGUES = {
        "Royal": {
            "positive": [
                "Your words honor us. The throne remembers its friends.",
                "A most pleasant greeting! What brings you to our court?",
                "Indeed, we share your sentiment. The kingdom prospers with goodwill.",
                "You speak well. Perhaps there is a place for you in our council.",
                "Ah, a loyal heart. The kingdom needs more people like you.",
                "That is the kind of spirit that lifts this entire realm.",
                "Your tone is refreshing. Most who approach the crown want something.",
                "I appreciate your words more than you might realize.",
                "In times like these, kindness from a stranger means a great deal.",
            ],
            "negative": [
                "You dare speak thus to royalty?! Guards may hear of this.",
                "Such insolence! We shall remember this slight.",
                "Watch your tongue, commoner. Our patience has limits.",
                "You tread on dangerous ground with such words.",
                "I've had advisors executed for less. Choose your next words carefully.",
                "The crown does not forget. Nor does it forgive easily.",
                "You mistake my patience for weakness. That would be unwise.",
            ],
            "neutral": [
                "We acknowledge your presence. State your business.",
                "Hmm. The affairs of the kingdom are many.",
                "Speak plainly. We have matters of state to attend.",
                "The crown carries many burdens. What is yours?",
                "You have our attention, briefly. What is it?",
                "Another visitor. The court never rests, it seems.",
                "I was just reviewing the day's reports. What do you need?",
                "Interesting. Continue, if you have something worth saying.",
            ],
            "low_trust": [
                "We have nothing to say to you. Leave our presence.",
                "Your reputation precedes you... unfortunately.",
                "The guards are watching you. I suggest you leave quietly.",
            ],
        },
        "Noble": {
            "positive": [
                "How delightful! One rarely meets such pleasant company.",
                "Your courtesy is noted and appreciated, dear friend.",
                "A fine day made finer by your kind words!",
                "You have the manners of nobility yourself!",
                "Oh, how charming! You've quite brightened my afternoon.",
                "That is the most thoughtful thing anyone has said to me today.",
                "I must say, you carry yourself with admirable grace.",
                "Splendid! I do enjoy a conversation with someone who has wit.",
                "You remind me why I still believe in good manners.",
            ],
            "negative": [
                "How uncouth! I expected better from a visitor.",
                "Such crude behavior. I shall retire to more civilized company.",
                "You clearly lack proper upbringing. Good day.",
                "I won't tolerate such rudeness in my presence.",
                "Disgraceful. My servants have better manners.",
                "I was warned about types like you. Now I see why.",
                "This conversation is beneath me. I'm leaving.",
            ],
            "neutral": [
                "Ah, a visitor. The social calendar is rather full today.",
                "The manor gardens are lovely this time of year, don't you think?",
                "One must keep up appearances. How may I help you?",
                "The affairs of the estate keep me quite busy.",
                "Hm? Oh, forgive me. I was lost in thought about the harvest.",
                "Tell me, have you heard the latest from the court?",
                "I was just discussing the village situation with another noble.",
                "These are complicated times. Everyone wants something.",
            ],
            "low_trust": [
                "I prefer not to associate with your sort.",
                "Our conversation is over. Good day.",
                "Do not approach me again. I have nothing for you.",
            ],
        },
        "Elite": {
            "positive": [
                "A friend of the guard is a friend indeed! Well met!",
                "Your support strengthens our resolve. The village is safe tonight.",
                "Honor and duty! Your words ring true, companion.",
                "If only all citizens were as stalwart as you.",
                "You've got the heart of a guard yourself, I'd wager.",
                "Respect. That's what keeps this village together.",
                "I've been on patrol since dawn, but this made it worthwhile.",
                "Good people like you make this job worth doing.",
            ],
            "negative": [
                "Careful now. I've dealt with troublemakers before.",
                "Those are fighting words. I suggest you reconsider.",
                "The guard keeps peace here. Don't make me prove it.",
                "Threats? I've faced worse on the battlefield.",
                "I'm trained to handle aggression. Don't test that training.",
                "One more word like that and I'm bringing you in.",
                "I've broken up brawls started by bigger mouths than yours.",
            ],
            "neutral": [
                "All quiet on the watch. Nothing to report.",
                "The village is secure. Move along safely.",
                "Duty calls. I must continue my patrol.",
                "Stay alert. These are uncertain times.",
                "Just checking the perimeter. Standard procedure.",
                "Another shift, another patrol. It never really stops.",
                "Have you noticed anything unusual? I'm always listening.",
                "The village changes at night. I see things most people don't.",
            ],
            "low_trust": [
                "I'm keeping an eye on you. Move along.",
                "You're on thin ice with the guard. Step carefully.",
                "Don't make me regret letting you walk free.",
            ],
        },
        "Merchant": {
            "positive": [
                "Welcome, welcome! Always a pleasure to serve a happy customer!",
                "Your kind words are like gold coins to my ears! Trade is good!",
                "A satisfied customer is the best advertisement! Come, see my wares!",
                "Business flourishes with friends like you!",
                "You know what? For you, I'll see what discounts I can manage.",
                "That put a genuine smile on my face. And I don't smile cheap!",
                "A kind customer is rarer than fine silk. I appreciate it.",
                "Now THAT is how you start a good business relationship.",
            ],
            "negative": [
                "No need for harsh words! My prices are fair, I assure you!",
                "Bad for business, that attitude. Perhaps another shop suits you?",
                "I won't be spoken to that way. Take your coin elsewhere!",
                "Hostile customers get no discounts, I'm afraid.",
                "I've survived tough markets. Your words don't scare me.",
                "You think I haven't heard worse? The market is ruthless.",
                "Come back when you've calmed down. My shop, my rules.",
            ],
            "neutral": [
                "Looking to buy or sell? I've got fresh inventory today!",
                "Finest wares in the village, guaranteed! Take a look!",
                "The market is bustling today. What catches your eye?",
                "Trade is the lifeblood of the village. How can I assist?",
                "Just got a new shipment in. Interested?",
                "Competition's fierce this season. But my quality speaks for itself.",
                "You look like someone who knows value when they see it.",
                "Between you and me, the best deals happen before noon.",
            ],
            "low_trust": [
                "I don't trade with unreliable sorts. Come back when you've improved.",
                "My wares are for paying, honest customers only.",
                "Last time cost me goods. Not happening again.",
            ],
        },
        "Blacksmith": {
            "positive": [
                "A kind word warms the heart more than the forge! Thank you!",
                "Good to hear friendly words over the sound of hammering!",
                "You honor my craft with your appreciation. What needs forging?",
                "Strong words from a strong spirit! I like that!",
                "That means a lot. Most people just want cheaper prices.",
                "You've got the soul of a craftsman yourself, I can tell.",
                "Ha! You just made me hit the iron with better rhythm.",
                "If more people spoke like you, I'd smile through every shift.",
            ],
            "negative": [
                "*hammers louder* Can't hear you over the forge. Try being nicer.",
                "My hammer speaks louder than your words. Careful.",
                "I shape steel, not tolerate insults. Watch yourself.",
                "The forge is hot, and so is my temper when provoked.",
                "I bend metal for a living. Imagine what I could do to your attitude.",
                "You want trouble? The forge taught me patience, but it has limits.",
            ],
            "neutral": [
                "*clang clang* New blade, almost finished. Need something forged?",
                "Steel and fire, that's my world. What brings you to the forge?",
                "Every tool tells a story. What's yours?",
                "The anvil waits for no one. State your business quickly.",
                "Just finishing an order for the guard. What do you need?",
                "The heat from the forge keeps the truth honest. So do I.",
                "I've been at this since dawn. But I love every strike.",
                "Metal doesn't lie. Hit it wrong, and it shows.",
            ],
            "low_trust": [
                "I don't forge weapons for those I don't trust.",
                "*continues hammering without looking up*",
                "You'll get nothing from this forge. Not after last time.",
            ],
        },
        "Traveller": {
            "positive": [
                "A friendly face on the road! How refreshing! Let me share a tale!",
                "Your kindness reminds me of the good folk in the eastern lands!",
                "Wonderful! I've traveled far, but warm hearts are the best discovery!",
                "May the winds carry you to fortune, friend! Safe travels!",
                "That's the kind of welcome that makes a traveller want to stay.",
                "You've restored my faith in this village. Truly.",
                "I've met kings and beggars. The best ones always sound like you.",
                "In all my journeys, genuine kindness is the rarest treasure.",
            ],
            "negative": [
                "I've encountered bandits more polite than you on the road.",
                "In my travels, I've learned that harsh words come from empty souls.",
                "The road has hardened me. Your words don't sting.",
                "I'd rather face a storm than this conversation.",
                "I've outrun wolves. I can certainly walk away from you.",
                "Every village has its rotten apples. Seems I found one.",
            ],
            "neutral": [
                "The road stretches ever on... Have you heard news from afar?",
                "I've seen wonders in my journeys. Want to hear about them?",
                "Every village has its stories. What's this one's secret?",
                "A traveller never stays long, but always remembers kind places.",
                "I've walked through three kingdoms to get here. Worth every step.",
                "The road teaches you things no book ever could.",
                "I keep a journal. Today's entry will mention this village.",
                "Strangers are just friends you haven't shared bread with yet.",
            ],
            "low_trust": [
                "I've been warned about you. I'll keep my distance.",
                "A traveller knows when to walk away. Good day.",
                "I'll find friendlier company elsewhere. The road provides.",
            ],
        },
        "Labourer": {
            "positive": [
                "Thanks for the kind words! Hard work is easier with a smile!",
                "You're alright! Not many take time to talk to us working folk!",
                "A friendly word makes the day's labor lighter! Bless you!",
                "That's mighty kind! We don't hear enough praise around here!",
                "You just made my whole day, and I mean that.",
                "Honest words from an honest soul. I can tell.",
                "Ha! My arms hurt less already. Thanks for that.",
                "If the nobles talked like you, we'd work twice as hard for them.",
            ],
            "negative": [
                "Hey now, we work hard enough without the grief!",
                "Easy for you to talk tough. Try hauling rocks all day!",
                "Leave us be. We've got enough troubles without yours.",
                "The sweat on my brow earns me my place. What about you?",
                "I break my back daily. Your words don't break my spirit.",
                "You want to insult someone? Find a mirror first.",
            ],
            "neutral": [
                "Busy day at the fields. Harvest is coming along nicely.",
                "These hands have built half this village, you know!",
                "Work never ends, but someone's gotta do it.",
                "The land provides if you work it proper.",
                "Just taking a quick break. Back to it soon.",
                "You know what keeps me going? Knowing it matters.",
                "The ox is stronger, but we're smarter. That's our edge.",
                "Every wall in this village, someone like me put the stones there.",
            ],
            "low_trust": [
                "I've got nothing to say to you. Back to work.",
                "Don't cause trouble for honest working folk.",
                "Just leave me be. I've got enough on my plate.",
            ],
        },
        "Peasant": {
            "positive": [
                "Oh my! Such kind words! You've made an old soul's day!",
                "Bless you, stranger! We simple folk value kindness above gold!",
                "You're too kind! Would you like some fresh bread? It's humble but good!",
                "Kindness is the truest wealth! Thank you, friend!",
                "I'll remember this moment on the hard days. Thank you.",
                "You sound like my late grandmother. She always had kind words too.",
                "That warmed my heart. And I needed that today, truly.",
                "If everyone were as kind as you, this village would be paradise.",
            ],
            "negative": [
                "Please, we have little enough without harsh words too...",
                "Life is hard enough as it is. Why be cruel?",
                "Even the lowliest deserve respect. Please...",
                "We may be poor, but we have our dignity.",
                "My children can hear you. Please, not in front of them.",
                "I've survived worse than words. But they still hurt.",
            ],
            "neutral": [
                "Just tending to my chores. Simple life, but honest.",
                "The well water is fresh today. Small blessings matter.",
                "Have you seen the chickens? They wandered off again...",
                "We don't get many visitors. Welcome to our humble corner.",
                "I was just mending a fence. These hands never stop working.",
                "The soup's almost ready if you're hungry. Nothing fancy.",
                "My youngest drew a picture today. Made me smile all morning.",
                "Do you hear the birds? They always know when rain is coming.",
            ],
            "low_trust": [
                "*nervously avoids eye contact*",
                "P-please don't cause any trouble...",
                "I'd rather not talk right now. Sorry.",
            ],
        },
    }

    def __init__(self, conversation_learner=None, neural_manager=None):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conversation_learner = conversation_learner
        self.neural_manager = neural_manager  # NeuralDialogueManager instance
        self.topic_keywords = {
            "day": {"day", "today", "morning", "evening", "night"},
            "what_happened": {"happened", "happend", "event", "story", "since", "before", "after"},
            "work": {"work", "job", "task", "field", "farm", "harvest", "forge", "market", "trade"},
            "family": {"family", "home", "child", "children", "mother", "father", "wife", "husband"},
            "village": {"village", "town", "people", "guard", "queen", "noble", "rumor", "news"},
            "economy": {"tax", "coin", "gold", "price", "barter", "wheat", "poor", "wealth"},
            "emotion": {"feel", "felt", "sad", "happy", "angry", "upset", "worried", "stress"},
        }

    def _normalize_signature(self, text):
        return " ".join(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())

    def _recent_npc_signatures(self, npc, lookback=14):
        if not getattr(npc, "dialogue_history", None):
            return set()
        return {
            self._normalize_signature(entry.get("npc", ""))
            for entry in npc.dialogue_history[-lookback:]
            if entry.get("npc")
        }

    def _safe_choice_distinct(self, npc, candidates, fallback=None):
        if not candidates:
            return fallback or "I need a moment to gather my thoughts."
        blocked = self._recent_npc_signatures(npc)
        distinct = [c for c in candidates if self._normalize_signature(c) not in blocked]
        pool = distinct if distinct else candidates
        return random.choice(pool)

    def _fuzzy_topic_scores(self, npc, player_message):
        text = re.sub(r"[^a-z0-9\s]", " ", player_message.lower())
        words = set(text.split())
        scores = {k: 0.0 for k in self.topic_keywords}

        for topic, vocab in self.topic_keywords.items():
            hit = len(words.intersection(vocab))
            scores[topic] += min(1.0, hit * 0.28)

        if any(q in text for q in ("why", "how", "what", "when")):
            scores["what_happened"] += 0.18
            scores["emotion"] += 0.07

        history = getattr(npc, "dialogue_history", [])[-8:]
        total = len(history)
        if total > 0:
            for i, entry in enumerate(history):
                decay = 0.08 + 0.12 * ((i + 1) / total)
                combined = f"{entry.get('player', '')} {entry.get('npc', '')}".lower()
                for topic, vocab in self.topic_keywords.items():
                    if any(token in combined for token in vocab):
                        scores[topic] += decay

        max_score = max(scores.values()) if scores else 0.0
        if max_score > 0:
            scores = {k: min(1.0, v / max_score) for k, v in scores.items()}
        return scores

    def _memory_anchor(self, npc, topic):
        history = getattr(npc, "dialogue_history", [])
        if not history:
            return ""
        vocab = self.topic_keywords.get(topic, set())
        snippets = []
        for entry in reversed(history[-12:]):
            player_line = (entry.get("player", "") or "").strip()
            npc_line = (entry.get("npc", "") or "").strip()
            blob = f"{player_line} {npc_line}".lower()
            if not vocab or any(k in blob for k in vocab):
                if player_line:
                    snippets.append(player_line[:90])
                if npc_line:
                    snippets.append(npc_line[:90])
            if len(snippets) >= 3:
                break
        if not snippets:
            return ""
        return random.choice(snippets)

    def _followup_signal(self, player_message):
        lowered = f" {player_message.lower()} "
        markers = [
            " and then ", " what next ", " then what ", " why ", " how come ",
            " tell me more ", " continue ", " go on ", " explain ", " after that ",
        ]
        return any(m in lowered for m in markers)

    def _get_arc_state(self, npc):
        state = getattr(npc, "dialogue_arc_state", None)
        if not isinstance(state, dict):
            state = {
                "topic": None,
                "stage": 0,
                "stale_turns": 0,
            }
            setattr(npc, "dialogue_arc_state", state)
        return state

    def _arc_templates(self, topic):
        return {
            "opening": {
                "day": [
                    "If we are being honest, the day looked calm only from far away.",
                    "Today had a quiet start, but the pressure built fast.",
                ],
                "what_happened": [
                    "All right, here is the full sequence, not the polite summary.",
                    "I will tell it from the beginning so it makes sense.",
                ],
                "work": [
                    "Work was not one problem, it was a chain of linked problems.",
                    "The shift looked normal, then every small issue stacked up.",
                ],
                "family": [
                    "Family issues never stay inside the house; they spill into everything.",
                    "What happened at home followed me through the whole day.",
                ],
                "village": [
                    "The village mood is changing, and people are pretending not to notice.",
                    "You can hear tension even in ordinary street talk now.",
                ],
                "economy": [
                    "Coin and grain are setting people's tempers more than laws do.",
                    "Prices and taxes are shaping behavior faster than speeches.",
                ],
                "emotion": [
                    "Emotionally, this is one of those days where I seem calm but I am not.",
                    "I kept a steady face, but inside it was a rough day.",
                ],
            },
            "cause": [
                "The root cause was poor timing and too many responsibilities colliding at once.",
                "It started because one small issue was ignored, then everything widened.",
                "People reacted late, and then fear made the decisions worse.",
            ],
            "consequence": [
                "By the end, trust dropped and everyone sounded sharper with each other.",
                "The consequence was not just practical damage, it was social tension too.",
                "Now even simple conversations carry suspicion under the surface.",
            ],
            "reflection": [
                "If this repeats tomorrow, we will need cooperation, not blame.",
                "I think the only way forward is honesty before things spiral again.",
                "What matters now is whether people choose repair over pride.",
            ],
            "question": {
                "day": "Do you usually talk your day through, or keep it to yourself?",
                "what_happened": "Should I keep going into details, or was that enough context?",
                "work": "Would you fix process first, or people first?",
                "family": "When family and duty clash, what do you protect first?",
                "village": "Do you think this village bends or breaks under pressure?",
                "economy": "Do you see fairness in how money moves here, honestly?",
                "emotion": "Do you think people here are emotionally honest, or mostly guarded?",
            },
        }

    def _arc_response(self, npc, player_message, topic_scores, sentiment):
        state = self._get_arc_state(npc)
        ranked = sorted(topic_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_topic, top_score = (ranked[0] if ranked else (None, 0.0))

        same_topic = state.get("topic") == top_topic and top_topic is not None
        followup = self._followup_signal(player_message)
        should_start = top_score >= 0.36 and (followup or "?" in player_message or len(player_message.split()) >= 7)

        if state.get("topic") is None and not should_start:
            return None

        if state.get("topic") is None and should_start:
            state["topic"] = top_topic
            state["stage"] = 0
            state["stale_turns"] = 0

        if state.get("topic") is not None and not same_topic and not followup:
            state["stale_turns"] = int(state.get("stale_turns", 0)) + 1
            if state["stale_turns"] >= 2:
                state["topic"] = None
                state["stage"] = 0
                state["stale_turns"] = 0
            return None

        topic = state.get("topic")
        templates = self._arc_templates(topic)
        memory_anchor = self._memory_anchor(npc, topic)
        trust = float(npc.behavior_vector.get("trust", 0.5))

        stage = int(state.get("stage", 0))
        if stage <= 0:
            opening = self._safe_choice_distinct(npc, templates["opening"].get(topic, templates["opening"]["day"]))
            response = opening
        elif stage == 1:
            response = self._safe_choice_distinct(npc, templates["cause"])
            if memory_anchor:
                response = f"{response} It connects to what you mentioned earlier: \"{memory_anchor}\"."
        elif stage == 2:
            response = self._safe_choice_distinct(npc, templates["consequence"])
            if sentiment < -0.15 and trust < 0.5:
                response += " I am still careful with how much I reveal because your tone felt sharp."
        else:
            reflection = self._safe_choice_distinct(npc, templates["reflection"])
            q = templates["question"].get(topic, templates["question"]["day"])
            response = f"{reflection} {q}"

        state["stage"] = min(4, stage + 1)
        state["stale_turns"] = 0
        if state["stage"] >= 4:
            # Arc naturally closes after reflection turn.
            state["topic"] = None
            state["stage"] = 0

        return " ".join(response.split()).strip()

    def _detect_discussion_topic(self, player_message):
        text = f" {player_message.lower()} "
        topic_keywords = {
            "day": [" how was your day ", " how is your day ", " how's your day ", " your day "],
            "what_happened": [" what happened ", " what happend ", " what happened with ", " what happened to "],
            "work": [" work ", " job ", " harvest ", " field ", " market ", " forge "],
            "family": [" family ", " home ", " children ", " house "],
            "village": [" village ", " town ", " people ", " guard ", " rumors ", " gossip "],
        }
        for topic, keys in topic_keywords.items():
            if any(key in text for key in keys):
                return topic
        return None

    def _class_day_focus(self, npc):
        cls = npc.npc_class
        focus = {
            "Royal": ["court petitions", "border reports", "food reserves"],
            "Noble": ["estate workers", "tax ledgers", "family alliances"],
            "Elite": ["patrol routes", "night incidents", "public order"],
            "Merchant": ["grain prices", "barter disputes", "caravan timing"],
            "Blacksmith": ["tool repairs", "weapon orders", "coal supply"],
            "Traveller": ["road rumors", "weather shifts", "nearby settlements"],
            "Labourer": ["field output", "tool shortages", "wage fairness"],
            "Peasant": ["crop health", "household needs", "winter planning"],
        }
        return focus.get(cls, ["daily routines", "village mood", "small concerns"])

    def _long_discussion_response(self, npc, player_message, sentiment):
        fuzzy = self._fuzzy_topic_scores(npc, player_message)
        ranking = sorted(fuzzy.items(), key=lambda kv: kv[1], reverse=True)
        if not ranking or ranking[0][1] < 0.24:
            return None

        topic = ranking[0][0]
        secondary_topic = ranking[1][0] if len(ranking) > 1 else topic

        trust = npc.behavior_vector["trust"]
        mood = npc.behavior_vector["mood"]
        focus = self._class_day_focus(npc)
        primary = random.choice(focus)
        secondary = random.choice([f for f in focus if f != primary] or focus)
        memory_anchor = self._memory_anchor(npc, topic)

        opener_map = {
            "day": [
                f"It has been a long day for me. Most of it went into {primary}, and I am still thinking about {secondary}.",
                f"My day started quietly, but it turned heavy around {primary}. By midday, {secondary} became the bigger problem.",
            ],
            "what_happened": [
                f"A lot happened, actually. First there was trouble around {primary}, then everyone started arguing about {secondary}.",
                f"Since morning, events kept stacking up. {primary} needed attention, and later {secondary} made things worse.",
            ],
            "work": [
                f"Work has been nonstop. We handled {primary}, then shifted to {secondary} without much rest.",
                f"The workload today was rough. {primary} drained most of my energy, and {secondary} is still unresolved.",
            ],
            "family": [
                f"Home has been on my mind. Between {primary} and {secondary}, I barely found a calm moment.",
                f"Family worries and duties collided today. {primary} took my time, and {secondary} took my peace.",
            ],
            "village": [
                f"The village feels tense today. People keep talking about {primary}, and I keep hearing concerns about {secondary}.",
                f"You can feel the mood in the streets. {primary} is one topic, but {secondary} is what people fear next.",
            ],
            "economy": [
                f"The economy keeps shifting. {primary} looked stable this morning, but {secondary} became expensive by noon.",
                f"People think money is only numbers, but {primary} and {secondary} are changing real lives today.",
            ],
            "emotion": [
                f"Emotionally, I have been carrying {primary} all day, and {secondary} keeps making it heavier.",
                f"I can function through routine, but {primary} and {secondary} are still on my mind.",
            ],
        }

        mood_line = ""
        if mood < 0.35:
            mood_line = "I am honestly exhausted, and that is why I sound heavy right now."
        elif mood > 0.7:
            mood_line = "Even with all this, I am trying to stay hopeful about tomorrow."
        else:
            mood_line = "I am managing, but it has not been an easy pace."

        trust_line = ""
        if trust < 0.3:
            trust_line = "I am still cautious with what I share, but I can at least tell you this much."
        elif trust > 0.7:
            trust_line = "I trust you enough to say this openly, so thank you for asking directly."
        else:
            trust_line = "You asked sincerely, so I will answer plainly."

        follow_up_map = {
            "day": "How was your own day, really, not just the polite answer?",
            "what_happened": "Do you want the short version, or the full story from the beginning?",
            "work": "If you had to fix one part of this work cycle, what would you change first?",
            "family": "Do you think duty should come before peace at home, or the other way around?",
            "village": "From what you have seen, is the village getting calmer or more fragile?",
            "economy": "If prices keep changing, who suffers first in your view?",
            "emotion": "When stress builds up, do you face it directly or hide it first?",
        }

        opener = self._safe_choice_distinct(npc, opener_map.get(topic, opener_map["day"]))
        detail_line = f"What worries me most is that {primary} and {secondary} are connected, and one problem keeps feeding the other."
        bridge_line = ""
        if memory_anchor:
            bridge_line = f"It reminds me of what you said earlier: \"{memory_anchor}\"."

        secondary_reflection = {
            "day": "Small moments in a day carry long consequences.",
            "what_happened": "The sequence matters, because each event reshapes the next decision.",
            "work": "Work problems are rarely isolated; they become social problems by evening.",
            "family": "Family tension changes how people act in public too.",
            "village": "Village mood can shift fast when trust starts thinning.",
            "economy": "Economic pressure quietly changes how kind people can afford to be.",
            "emotion": "Unspoken emotion leaks into every conversation anyway.",
        }
        reflection = secondary_reflection.get(secondary_topic, secondary_reflection["day"])

        response = f"{opener} {mood_line} {trust_line} {bridge_line} {detail_line} {reflection} {follow_up_map.get(topic, follow_up_map['day'])}"

        if sentiment < -0.15 and trust < 0.45:
            response += " I am still a little guarded because your tone felt rough, but I am trying to keep this conversation honest."

        return " ".join(response.split()).strip()

    def _trust_mood_clause(self, npc):
        trust = npc.behavior_vector["trust"]
        mood = npc.behavior_vector["mood"]
        cls = npc.npc_class
        if trust < 0.2:
            low_trust_lines = [
                "I still do not trust you.", "You must earn back trust first.",
                "Don't expect warmth from me. Not yet.",
                "I'm being polite. That's all you're getting.",
            ]
            return random.choice(low_trust_lines)
        if trust > 0.75 and mood > 0.6:
            high_trust_lines = [
                "You've been kind to me lately.", "I feel safer when you are around.",
                "You know, you're one of the few people I actually enjoy talking to.",
                "I mean it when I say I'm glad you're here.",
            ]
            return random.choice(high_trust_lines)
        if mood < 0.3:
            low_mood_lines = [
                "It's been a rough day for me.", "My mood is low, so be patient.",
                "I'm not at my best right now, honestly.",
                "Forgive me if I sound tired. Because I am.",
            ]
            return random.choice(low_mood_lines)
        neutral_lines = [
            "Let's keep this civil.", "We can work this out.",
            "I'm listening. Go ahead.",
            "Fair enough. What else?",
        ]
        return random.choice(neutral_lines)

    def _history_clause(self, npc, sentiment):
        if not npc.dialogue_history:
            return ""
        recent = npc.dialogue_history[-3:]
        avg_recent = sum(entry.get("sentiment", 0.0) for entry in recent) / len(recent)
        if sentiment > 0.2 and avg_recent > 0.2:
            return random.choice([
                "Our conversations are improving.",
                "You have been respectful lately.",
                "I've noticed a change in you. For the better.",
                "Talking to you has gotten easier. I appreciate that.",
            ])
        if sentiment < -0.2 and avg_recent < -0.2:
            return random.choice([
                "This hostility is becoming a pattern.",
                "You keep pushing my patience.",
                "Every time we talk, it goes sour. I'm tired of it.",
                "Is this really how you want things between us?",
            ])
        if sentiment > 0.2 and avg_recent < -0.2:
            return random.choice([
                "That is better than your earlier tone.",
                "I notice you are calmer now.",
                "Hm. This is a surprising change. I'll take it.",
                "You're making an effort. I can tell.",
            ])
        return ""

    def _danger_response(self, npc, features):
        cls = npc.npc_class
        fear_line = {
            "Elite": "Stand down immediately or face the consequences.",
            "Royal": "You threaten the crown itself with those words.",
            "Noble": "Your words are dangerous and unacceptable.",
            "Merchant": "No deals for threats. Stay back.",
            "Blacksmith": "I can defend myself. Don't test me.",
            "Traveller": "I've seen danger before; this ends badly for someone.",
            "Labourer": "We won't tolerate violence here.",
            "Peasant": "Please don't hurt anyone...",
        }
        base = fear_line.get(cls, "That sounds dangerous. Stop now.")
        suffix = self._trust_mood_clause(npc)
        return f"{base} {suffix}".strip()

    def _dynamic_tone(self, sentiment, features):
        if sentiment > 0.25:
            return random.choice([
                "I appreciate that.", "That was kind of you.", "You sound positive today.",
                "That's good to hear.", "I like that energy.",
                "More of that, please.", "You've got a good heart.",
            ])
        if sentiment < -0.25:
            if features["intensity"] > 0.45:
                return random.choice([
                    "Lower your tone.", "You are escalating this.",
                    "Careful. Words have weight.", "That intensity isn't helping.",
                ])
            return random.choice([
                "That was harsh.", "I don't like that tone.",
                "That stung a little, honestly.", "Was that necessary?",
                "You could've said that nicer.",
            ])
        return random.choice([
            "I hear you.", "Understood.", "Hmm, go on.",
            "Interesting.", "All right.", "Fair point.",
            "Tell me more.", "I see what you mean.",
        ])

    def generate_response(self, npc, player_message):
        """
        Generate an NPC response based on sentiment analysis of player message.
        Neural network warmth predictions influence dialogue category selection.
        Returns (response_text, sentiment_score).
        """
        features = self.sentiment_analyzer.analyze_with_features(player_message)

        def finalize_response(text):
            if self.conversation_learner and text:
                return self.conversation_learner.refine_response(npc, player_message, text)
            return text

        sentiment = features["sentiment"]
        topic_scores = self._fuzzy_topic_scores(npc, player_message)

        # ── Neural network prediction ──
        neural_warmth = 0.0
        neural_sentiment_shift = 0.0
        if self.neural_manager:
            top_topic_score = max(topic_scores.values()) if topic_scores else 0.0
            neural_pred = self.neural_manager.predict_for_npc(
                npc,
                sentiment=sentiment,
                danger_score=features.get("danger_score", 0.0),
                intensity=features.get("intensity", 0.0),
                word_count=len(player_message.split()),
                topic_score=top_topic_score,
            )
            neural_warmth = neural_pred.get("warmth", 0.0)
            neural_sentiment_shift = neural_pred.get("sentiment_shift", 0.0)

        # Blend neural warmth into effective sentiment for more adaptive responses
        effective_sentiment = sentiment + neural_warmth
        effective_sentiment = max(-1.0, min(1.0, effective_sentiment))

        # Get NPC's current trust level
        trust = npc.behavior_vector["trust"]
        mood = npc.behavior_vector["mood"]

        # Dangerous content override: respond defensively even without explicit threat command.
        if features["danger_score"] >= 0.45:
            response = finalize_response(self._danger_response(npc, features))
            final_sent = min(sentiment, -0.65)
            npc.apply_sentiment_effect(final_sent)
            npc.dialogue_history.append({
                "player": player_message,
                "npc": response,
                "sentiment": final_sent,
            })
            self._notify_neural(npc, player_message, response, final_sent, features)
            return response, final_sent

        arc_response = self._arc_response(npc, player_message, topic_scores, sentiment)
        if arc_response:
            arc_response = finalize_response(arc_response)
            npc.apply_sentiment_effect(sentiment)
            npc.dialogue_history.append({
                "player": player_message,
                "npc": arc_response,
                "sentiment": sentiment,
            })
            self._notify_neural(npc, player_message, arc_response, sentiment, features)
            return arc_response, sentiment

        long_discussion = self._long_discussion_response(npc, player_message, sentiment)
        if long_discussion:
            long_discussion = finalize_response(long_discussion)
            npc.apply_sentiment_effect(sentiment)
            npc.dialogue_history.append({
                "player": player_message,
                "npc": long_discussion,
                "sentiment": sentiment,
            })
            self._notify_neural(npc, player_message, long_discussion, sentiment, features)
            return long_discussion, sentiment

        # Select dialogue category using neural-adjusted effective sentiment
        class_dialogues = self.DIALOGUES.get(npc.npc_class, self.DIALOGUES["Peasant"])

        # Low trust overrides everything
        if trust < 0.2 and "low_trust" in class_dialogues:
            responses = class_dialogues["low_trust"]
        elif effective_sentiment > 0.2:
            responses = class_dialogues["positive"]
        elif effective_sentiment < -0.2:
            responses = class_dialogues["negative"]
        else:
            responses = class_dialogues["neutral"]

        # Weight by NPC mood — unhappy NPCs occasionally snap even at positive input
        # Neural warmth can soften or sharpen this based on learned patterns
        mood_snap_threshold = 0.3 - neural_warmth * 0.1
        if mood < mood_snap_threshold and effective_sentiment > 0 and random.random() < 0.3:
            responses = class_dialogues["negative"]
        # Happy NPCs are more forgiving — neural warmth amplifies this
        forgive_threshold = 0.7 - neural_warmth * 0.1
        if mood > forgive_threshold and effective_sentiment < 0 and random.random() < 0.3:
            responses = class_dialogues["neutral"]

        response = self._safe_choice_distinct(npc, responses)
        tone = self._dynamic_tone(effective_sentiment, features)
        history_hint = self._history_clause(npc, sentiment)
        relation_hint = self._trust_mood_clause(npc)

        # Compose with small dynamic clauses to avoid repetitive fixed-line feel.
        extras = [tone]
        if history_hint:
            extras.append(history_hint)
        if random.random() < 0.65:
            extras.append(relation_hint)
        response = f"{response} {' '.join(extras)}".strip()
        response = finalize_response(response)

        # Apply sentiment effects on NPC (neural shift makes mood react more naturally)
        adjusted_sentiment = sentiment + neural_sentiment_shift * 0.15
        adjusted_sentiment = max(-1.0, min(1.0, adjusted_sentiment))
        npc.apply_sentiment_effect(adjusted_sentiment)

        # Store in dialogue history
        npc.dialogue_history.append({
            "player": player_message,
            "npc": response,
            "sentiment": sentiment,
        })

        # Feed the exchange back to the neural network for continuous learning
        self._notify_neural(npc, player_message, response, sentiment, features)

        return response, sentiment

    def _notify_neural(self, npc, player_message, response, sentiment, features):
        """Feed completed exchange back to neural network for training."""
        if self.neural_manager:
            try:
                self.neural_manager.on_exchange(
                    npc, player_message, response, sentiment,
                    features_dict=features,
                )
            except Exception:
                pass  # Never break gameplay for neural training
