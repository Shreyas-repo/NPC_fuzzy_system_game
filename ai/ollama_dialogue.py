"""
Ollama-powered dynamic dialogue system.
Uses local LLM (deepseek-r1:7b) to generate contextual NPC responses
based on their personality, mood, trust, class, and conversation history.
"""
import json
import threading
import queue
import re
import time
from collections import OrderedDict
from ai.ollama_client import OllamaClient
from config import (
    OLLAMA_USE_LLM_EMOTION_ANALYSIS,
    OLLAMA_CHAT_TIMEOUT,
    OLLAMA_CHAT_NUM_PREDICT,
    OLLAMA_MAX_HISTORY_EXCHANGES,
    OLLAMA_ENABLE_RESPONSE_CACHE,
    OLLAMA_CACHE_MAX_ENTRIES,
    OLLAMA_SLOW_RESPONSE_THRESHOLD,
    OLLAMA_SLOW_RESPONSE_COOLDOWN,
)


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"


class OllamaDialogue:
    """
    Generates dynamic NPC dialogue using Ollama local LLM.
    Runs inference in a background thread to avoid blocking the game loop.
    """

    def __init__(self, client=None, conversation_learner=None):
        self.client = client or OllamaClient(base_url="http://localhost:11434", chat_model=MODEL)
        self.conversation_learner = conversation_learner
        self.response_queue = queue.Queue()
        self.pending = False
        self.last_response = None
        self.last_emotion = None
        self.available = self._check_ollama()
        self.use_llm_emotion_analysis = OLLAMA_USE_LLM_EMOTION_ANALYSIS
        self.enable_cache = OLLAMA_ENABLE_RESPONSE_CACHE
        self.cache_max_entries = OLLAMA_CACHE_MAX_ENTRIES
        self.response_cache = OrderedDict()
        self.slow_response_threshold = OLLAMA_SLOW_RESPONSE_THRESHOLD
        self.slow_response_cooldown = OLLAMA_SLOW_RESPONSE_COOLDOWN
        self.cooldown_until = 0.0

        if self.available:
            self._warmup_async()

    def _check_ollama(self):
        """Check if Ollama is running."""
        return self.client.is_available()

    def generate_response_async(self, npc, player_message, conversation_history=None):
        """
        Start generating a response in a background thread.
        Call poll_response() to check if it's ready.
        """
        if not self.available:
            return False

        if self.pending:
            return False  # Already generating

        if time.time() < self.cooldown_until:
            return False

        cache_key = self._cache_key(npc, player_message)
        if self.enable_cache and cache_key in self.response_cache:
            cached_response, cached_emotion = self.response_cache[cache_key]
            self.response_cache.move_to_end(cache_key)
            self.response_queue.put({
                "success": True,
                "response": cached_response,
                "emotion": dict(cached_emotion),
                "npc_name": npc.name,
                "npc_class": npc.npc_class,
                "cached": True,
            })
            return True

        self.pending = True
        thread = threading.Thread(
            target=self._generate_worker,
            args=(npc, player_message, conversation_history, cache_key),
            daemon=True
        )
        thread.start()
        return True

    def _generate_worker(self, npc, player_message, conversation_history, cache_key):
        """Background worker that calls Ollama API."""
        try:
            start_time = time.perf_counter()
            prompt = self._build_prompt(npc, player_message, conversation_history)
            raw_text = self.client.generate(
                prompt,
                model=MODEL,
                options={
                    "temperature": 0.55,
                    "top_p": 0.9,
                    "num_predict": OLLAMA_CHAT_NUM_PREDICT,
                    "stop": ["\n\n", "Player:", "Human:", "["],
                },
                timeout=OLLAMA_CHAT_TIMEOUT,
            )

            # Clean up the response — remove thinking tags and artifacts
            clean_text = self._clean_response(raw_text)
            if self.conversation_learner:
                clean_text = self.conversation_learner.refine_response(
                    npc,
                    player_message,
                    clean_text,
                )

            # Extract emotion data from the response
            emotion_data = self._extract_emotion(clean_text, player_message)

            elapsed = time.perf_counter() - start_time
            if elapsed > self.slow_response_threshold:
                self.cooldown_until = time.time() + self.slow_response_cooldown

            if self.enable_cache:
                self.response_cache[cache_key] = (clean_text, emotion_data)
                self.response_cache.move_to_end(cache_key)
                while len(self.response_cache) > self.cache_max_entries:
                    self.response_cache.popitem(last=False)

            self.response_queue.put({
                "success": True,
                "response": clean_text,
                "emotion": emotion_data,
                "npc_name": npc.name,
                "npc_class": npc.npc_class,
                "latency": elapsed,
            })

        except Exception as e:
            self.response_queue.put({
                "success": False,
                "error": str(e)
            })
            self.cooldown_until = time.time() + self.slow_response_cooldown

        self.pending = False

    def _warmup_async(self):
        """Warm up model load so first player interaction is faster."""
        thread = threading.Thread(target=self._warmup_worker, daemon=True)
        thread.start()

    def _warmup_worker(self):
        try:
            self.client.generate(
                "Reply with one medieval word.",
                model=MODEL,
                options={"temperature": 0.0, "num_predict": 8},
                timeout=min(8.0, OLLAMA_CHAT_TIMEOUT),
            )
        except Exception:
            pass

    def _build_prompt(self, npc, player_message, conversation_history=None):
        """Build a character-aware prompt for the LLM."""
        mood_desc = self._mood_to_text(npc.behavior_vector["mood"])
        trust_desc = self._trust_to_text(npc.behavior_vector["trust"])
        energy_desc = self._energy_to_text(npc.needs["energy"])
        hunger_desc = "hungry" if npc.needs["hunger"] > 0.6 else "satisfied"

        # Personality traits
        friendly = "friendly" if npc.personality["friendliness"] > 0.6 else "reserved" if npc.personality["friendliness"] > 0.3 else "cold"
        social = "sociable" if npc.personality["sociability"] > 0.6 else "introverted"

        # Build conversation context
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            recent = conversation_history[-OLLAMA_MAX_HISTORY_EXCHANGES:]
            for entry in recent:
                history_text += f"Player: {entry['player']}\n{npc.name}: {entry['npc']}\n"

        social_rank_names = {0: "peasant", 1: "commoner", 2: "traveller",
                            3: "middle class", 4: "upper class", 5: "royalty"}
        rank = social_rank_names.get(getattr(npc, 'social_rank', 1), "commoner")

        style_hint = ""
        if self.conversation_learner:
            style_hint = self.conversation_learner.build_generation_hint(npc)

        prompt = (
            f"You are {npc.name}, a medieval {npc.npc_class}. "
            f"Persona: {friendly}, {social}, rank {rank}. "
            f"State: {mood_desc}, {trust_desc} toward player, {energy_desc}, {hunger_desc}, activity {npc.state}.\n"
            f"Reply as {npc.name} in 1 short sentence, in-character only. "
            f"Do not repeat your recent wording.\n"
            f"{history_text}Player: {player_message}\n"
            f"{npc.name}:"
        )

        if style_hint:
            prompt = f"{prompt}\n{style_hint}\n"

        return prompt

    def _cache_key(self, npc, player_message):
        mood_bucket = int(npc.behavior_vector.get("mood", 0.5) * 4)
        trust_bucket = int(npc.behavior_vector.get("trust", 0.5) * 4)
        normalized = " ".join(player_message.lower().split())[:120]
        return f"{npc.npc_class}|{npc.state}|m{mood_bucket}|t{trust_bucket}|{normalized}"

    def _clean_response(self, text):
        """Clean up LLM response — remove thinking tags, trim, etc."""
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any remaining XML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove asterisk actions like *smiles*
        text = re.sub(r'\*[^*]+\*', '', text)
        # Remove quotes around the whole response
        text = text.strip().strip('"').strip("'")
        # Take only the first 1-2 sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 2:
            text = ' '.join(sentences[:2])
        # Final cleanup
        text = text.strip()
        if not text:
            text = "..."
        return text

    def _extract_emotion(self, response_text, player_message):
        """
        Extract an emotion vector from the interaction.
        Returns a dict with emotion dimensions used for the vector database.
        """
        if self.use_llm_emotion_analysis:
            llm_analysis = self.client.analyze_text(f"Player: {player_message}\nNPC: {response_text}")
            if llm_analysis:
                normalized = {
                    "joy": float(llm_analysis.get("joy", 0.0)),
                    "anger": float(llm_analysis.get("anger", 0.0)),
                    "fear": float(llm_analysis.get("fear", 0.0)),
                    "sadness": float(llm_analysis.get("sadness", 0.0)),
                    "trust": float(llm_analysis.get("trust", 0.0)),
                    "surprise": float(llm_analysis.get("surprise", 0.0)),
                    "disgust": float(llm_analysis.get("disgust", 0.0)),
                    "curiosity": float(llm_analysis.get("curiosity", 0.0)),
                }
                return {k: max(0.0, min(1.0, v)) for k, v in normalized.items()}

        combined = (player_message + " " + response_text).lower()

        # Emotion dimensions (0.0 to 1.0)
        emotions = {
            "joy":       0.0,
            "anger":     0.0,
            "fear":      0.0,
            "sadness":   0.0,
            "trust":     0.0,
            "surprise":  0.0,
            "disgust":   0.0,
            "curiosity": 0.0,
        }

        # Keyword-based emotion extraction
        joy_words = {"happy", "glad", "pleased", "joy", "wonderful", "great", "good",
                     "smile", "laugh", "delight", "welcome", "friend", "kind", "thank",
                     "bless", "honor", "love", "beautiful", "fine", "well", "prosper"}
        anger_words = {"angry", "furious", "rage", "hate", "fight", "attack", "insolence",
                       "dare", "threat", "war", "destroy", "fool", "stupid", "shut",
                       "curse", "damn", "trouble", "hostile", "rude"}
        fear_words = {"afraid", "scared", "fear", "danger", "careful", "beware", "warn",
                      "nervous", "trouble", "threat", "dark", "uncertain", "worry"}
        sad_words = {"sad", "sorry", "poor", "suffer", "pain", "lonely", "miss",
                     "mourn", "weep", "cry", "loss", "humble", "unfortunate"}
        trust_words = {"trust", "reliable", "honest", "loyal", "faith", "believe",
                       "friend", "companion", "ally", "protect", "safe", "honor"}
        surprise_words = {"surprise", "unexpected", "amazing", "wonder", "incredible",
                          "really", "truly", "indeed", "remarkable", "astonish"}
        disgust_words = {"disgust", "vile", "filth", "ugly", "repulsive", "wretched",
                         "crude", "uncouth", "horrible", "terrible", "awful"}
        curiosity_words = {"curious", "wonder", "question", "tell", "explain", "how",
                           "why", "what", "interest", "story", "tale", "know", "learn"}

        words = set(re.findall(r'\b\w+\b', combined))

        emotions["joy"] = min(1.0, len(words & joy_words) * 0.2)
        emotions["anger"] = min(1.0, len(words & anger_words) * 0.2)
        emotions["fear"] = min(1.0, len(words & fear_words) * 0.25)
        emotions["sadness"] = min(1.0, len(words & sad_words) * 0.25)
        emotions["trust"] = min(1.0, len(words & trust_words) * 0.2)
        emotions["surprise"] = min(1.0, len(words & surprise_words) * 0.25)
        emotions["disgust"] = min(1.0, len(words & disgust_words) * 0.25)
        emotions["curiosity"] = min(1.0, len(words & curiosity_words) * 0.2)

        # Normalize so at least one dimension has a value
        max_val = max(emotions.values())
        if max_val == 0:
            emotions["curiosity"] = 0.3  # Default: mild curiosity

        return emotions

    def poll_response(self):
        """
        Check if a response is ready.
        Returns the response dict or None if still generating.
        """
        try:
            result = self.response_queue.get_nowait()
            self.last_response = result
            if result.get("success"):
                self.last_emotion = result.get("emotion")
            return result
        except queue.Empty:
            return None

    def _mood_to_text(self, mood):
        if mood > 0.8: return "very happy and cheerful"
        if mood > 0.6: return "in a good mood"
        if mood > 0.4: return "feeling neutral"
        if mood > 0.2: return "feeling down and irritable"
        return "very unhappy and hostile"

    def _trust_to_text(self, trust):
        if trust > 0.8: return "very trusting"
        if trust > 0.6: return "somewhat trusting"
        if trust > 0.4: return "cautious"
        if trust > 0.2: return "suspicious"
        return "deeply distrustful"

    def _energy_to_text(self, energy):
        if energy > 0.7: return "energetic"
        if energy > 0.4: return "a bit tired"
        return "exhausted"
