"""
Chat box UI for player-NPC interaction.
Supports local rule-based dialogue and optional LLM backends.
"""
import random
import pygame
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    COLORS,
    CHAT_MAX_HISTORY,
    OLLAMA_RESPONSE_MAX_WAIT,
)


class ChatBox:
    """Chat interface for talking to NPCs with local-first dialogue."""

    def __init__(self):
        self.active = False
        self.target_npc = None
        self.input_text = ""
        self.history = []  # list of (sender, text, color)
        self.font = None
        self.small_font = None
        self.cursor_visible = True
        self.cursor_timer = 0.0
        self.scroll_offset = 0

        # Optional LLM integration (disabled in local-only mode)
        self.ollama_dialogue = None
        self.dialogue_generator = None  # fallback
        self.emotion_database = None
        self.conversation_learner = None
        self.waiting_for_response = False
        self.loading_dots = 0
        self.loading_timer = 0.0
        self.response_wait_timer = 0.0
        self.npcs = []
        self.last_response_latency = 0.0
        self.crime_reports = []
        self.social_incidents = []
        self.idle_open_timer = 0.0
        self.idle_prompt_count = 0
        self.session_player_messages = 0

        # Layout
        self.box_width = 520
        self.box_height = 320
        self.box_x = SCREEN_WIDTH // 2 - self.box_width // 2
        self.box_y = SCREEN_HEIGHT - self.box_height - 10
        self.input_height = 32
        self.padding = 10

    def init_fonts(self):
        """Initialize fonts (must be called after pygame.init)."""
        self.font = pygame.font.SysFont("Arial", 14)
        self.small_font = pygame.font.SysFont("Arial", 12)
        self.title_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.emoji_font = pygame.font.SysFont("Segoe UI Emoji", 14)

    def set_ollama(self, ollama_dialogue, emotion_database, conversation_learner=None):
        """Set optional LLM dialogue system and emotion database."""
        self.ollama_dialogue = ollama_dialogue
        self.emotion_database = emotion_database
        self.conversation_learner = conversation_learner

    def set_npcs(self, npcs):
        """Provide current world NPC list for reaction propagation."""
        self.npcs = npcs or []

    def consume_crime_report(self):
        """Pop one reported suspicious/criminal interaction, if any."""
        if not self.crime_reports:
            return None
        return self.crime_reports.pop(0)

    def consume_social_incident(self):
        """Pop one player social incident (awkward/annoying interaction), if any."""
        if not self.social_incidents:
            return None
        return self.social_incidents.pop(0)

    def open(self, npc, dialogue_generator):
        """Open chat with an NPC."""
        self.active = True
        self.target_npc = npc
        self.dialogue_generator = dialogue_generator
        self.input_text = ""
        self.history = []
        self.scroll_offset = 0
        self.waiting_for_response = False
        self.idle_open_timer = 0.0
        self.idle_prompt_count = 0
        self.session_player_messages = 0

        # Load conversation history from NPC
        for entry in npc.dialogue_history[-8:]:
            self.history.append(("You", entry["player"], COLORS["ui_chat_player"]))
            self.history.append((npc.name, entry["npc"], COLORS["ui_chat_npc"]))

        # Set NPC state
        npc.state = "talking"

        # Show prediction from emotion database
        if self.emotion_database:
            prediction = self.emotion_database.get_npc_prediction(npc.name)
            if prediction and prediction.get("interaction_count", 0) > 0:
                tendency = prediction.get("tendency", "neutral")
                self.history.append(("System",
                    f"[{npc.name} remembers you as {tendency} (from {prediction['interaction_count']} past chats)]",
                    COLORS["ui_text_dim"]))

    def close(self):
        """Close the chat box."""
        if self.target_npc:
            self._handle_no_talk_close_reaction()
            # Preserve explicit reaction states (e.g., fleeing/fighting) set during chat.
            if self.target_npc.state == "talking":
                self.target_npc.state = "idle"
                self.target_npc.state_timer = 0
        self.active = False
        self.target_npc = None
        self.input_text = ""
        self.waiting_for_response = False

    def handle_event(self, event):
        """Handle keyboard input for the chat box."""
        if not self.active:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.close()
                return True
            elif event.key == pygame.K_RETURN:
                if not self.waiting_for_response:
                    self._send_message()
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
                return True
            elif event.key == pygame.K_UP:
                self.scroll_offset += 1
                return True
            elif event.key == pygame.K_DOWN:
                self.scroll_offset = max(0, self.scroll_offset - 1)
                return True
            elif event.key == pygame.K_PAGEUP:
                self.scroll_offset += 5
                return True
            elif event.key == pygame.K_PAGEDOWN:
                self.scroll_offset = max(0, self.scroll_offset - 5)
                return True
            elif event.unicode and len(self.input_text) < 200 and not self.waiting_for_response:
                self.input_text += event.unicode
                return True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Wheel up
                self.scroll_offset += 2
                return True
            elif event.button == 5:  # Wheel down
                self.scroll_offset = max(0, self.scroll_offset - 2)
                return True

        return False

    def _build_wrapped_history_rows(self, max_text_width):
        """Flatten message history into wrapped render rows for accurate scrolling."""
        rows = []
        for sender, text, color in self.history:
            sender_label = f"{sender}: "
            sender_width = self.small_font.render(sender_label, True, color).get_width()
            first_line_width = max(30, max_text_width - sender_width)
            continuation_width = max_text_width

            words = (text or "").split()
            if not words:
                rows.append((sender, "", color))
                continue

            current = ""
            is_first = True
            for word in words:
                candidate = f"{current}{word} "
                limit = first_line_width if is_first else continuation_width
                if self.font.render(candidate, True, COLORS["ui_text"]).get_width() > limit and current:
                    rows.append((sender if is_first else "", current.strip(), color))
                    is_first = False
                    current = f"{word} "
                else:
                    current = candidate

            if current:
                rows.append((sender if is_first else "", current.strip(), color))

        return rows

    def _send_message(self):
        """Send the current input text as a chat message."""
        if not self.input_text.strip() or not self.target_npc:
            return

        player_msg = self.input_text.strip()
        self.input_text = ""

        # Add player message to history
        self.history.append(("You", player_msg, COLORS["ui_chat_player"]))
        self.session_player_messages += 1
        self.idle_open_timer = 0.0

        intent = self._detect_intent(player_msg)
        if intent["type"] in ("threat_robbery", "threat_violence", "suspicious"):
            self._handle_threat_intent(player_msg, intent)
            return

        # Try optional LLM backend first when available
        if self.ollama_dialogue and self.ollama_dialogue.available:
            success = self.ollama_dialogue.generate_response_async(
                self.target_npc, player_msg,
                self.target_npc.dialogue_history
            )
            if success:
                self.waiting_for_response = True
                self.loading_dots = 0
                self.response_wait_timer = 0.0
                self._pending_player_msg = player_msg
                return

        # Local rule-based dialogue
        self._use_fallback_dialogue(player_msg)

    def _use_fallback_dialogue(self, player_msg):
        """Use the local rule-based dialogue as fallback."""
        if self.dialogue_generator:
            response, sentiment = self.dialogue_generator.generate_response(
                self.target_npc, player_msg
            )

            self.history.append((self.target_npc.name, response, COLORS["ui_chat_npc"]))

            # Record in emotion database even in local-only mode.
            emotion = self._estimate_local_emotion(sentiment)
            if self.emotion_database:
                self.emotion_database.add_interaction(
                    self.target_npc, player_msg, response,
                    emotion, sentiment
                )
            if self.conversation_learner:
                self.conversation_learner.log_exchange(
                    self.target_npc,
                    player_msg,
                    response,
                    sentiment=sentiment,
                    emotion=emotion,
                    source="local",
                )
            self.last_response_latency = 0.0

        self._trim_history()
        self.scroll_offset = 0

    def _detect_intent(self, message):
        text = " " + message.lower() + " "
        robbery_terms = [
            " rob ", " steal ", " mug ", " loot ", " plunder ", " take your gold ",
            " take your money ", " extort ", " pickpocket ",
        ]
        violence_terms = [
            " kill ", " attack ", " hurt ", " stab ", " slash ", " beat you ", " fight you ",
            " burn ", " destroy ", " blood ", " weapon ", " murder ", " poison ", " bomb ",
        ]
        suspicious_terms = [
            " smuggle ", " contraband ", " illegal ", " black market ", " bribe ",
            " scam ", " fraud ", " fake papers ", " poison the well ", " assassinate ",
        ]
        rob_hits = sum(1 for t in robbery_terms if t in text)
        vio_hits = sum(1 for t in violence_terms if t in text)
        sus_hits = sum(1 for t in suspicious_terms if t in text)
        if rob_hits > 0:
            return {"type": "threat_robbery", "severity": min(1.0, 0.7 + 0.1 * rob_hits)}
        if vio_hits > 0:
            return {"type": "threat_violence", "severity": min(1.0, 0.75 + 0.1 * vio_hits)}
        if sus_hits > 0:
            return {"type": "suspicious", "severity": min(1.0, 0.45 + 0.1 * sus_hits)}
        return {"type": "neutral", "severity": 0.0}

    def _class_reaction(self, npc):
        cls = npc.npc_class
        if cls == "Elite":
            return "fight", "Drop your weapon. You answer to the guard!"
        if cls in ("Royal", "Noble"):
            return "call_for_help", "Guards! Seize this criminal at once!"
        if cls in ("Merchant", "Traveller"):
            return "flee_to_guard", "Thief! Help, someone call the guard!"
        if cls == "Blacksmith":
            if npc.personality.get("aggression", 0.0) > 0.35:
                return "fight", "You'll find iron answers faster than fear."
            return "call_for_help", "Back off, or I'll call the guard now!"
        if cls == "Labourer":
            return "flee_to_guard", "No! Help! Guard!"
        return "helpless", "P-please don't rob me..."

    def _handle_threat_intent(self, player_msg, intent):
        npc = self.target_npc
        if not npc:
            return

        reaction_type, response_text = self._class_reaction(npc)
        npc.react_to_threat(reaction_type, self.npcs)
        self.history.append((npc.name, response_text, COLORS["ui_chat_npc"]))

        emotion = {
            "joy": 0.0,
            "anger": 0.7 if reaction_type in ("fight", "call_for_help") else 0.2,
            "fear": 0.8 if reaction_type in ("flee_to_guard", "helpless") else 0.3,
            "sadness": 0.2,
            "trust": 0.0,
            "surprise": 0.4,
            "disgust": 0.35,
            "curiosity": 0.0,
        }
        sentiment = -0.9

        npc.apply_sentiment_effect(sentiment)
        npc.dialogue_history.append({
            "player": player_msg,
            "npc": response_text,
            "sentiment": sentiment,
        })

        if self.emotion_database:
            self.emotion_database.add_interaction(
                npc,
                player_msg,
                response_text,
                emotion,
                sentiment,
            )

        if self.conversation_learner:
            self.conversation_learner.log_exchange(
                npc,
                player_msg,
                response_text,
                sentiment=sentiment,
                emotion=emotion,
                source="rule",
            )

        self.crime_reports.append({
            "npc_name": npc.name,
            "npc_class": npc.npc_class,
            "intent_type": intent.get("type", "suspicious"),
            "severity": float(intent.get("severity", 0.5)),
            "player_message": player_msg,
        })

        self._trim_history()
        self.scroll_offset = 0

    def update(self, dt):
        """Update chat box — check for Ollama responses."""
        if not self.active:
            return

        self._update_idle_reaction(dt)

        self.cursor_timer += dt
        if self.cursor_timer >= 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

        # Loading animation
        if self.waiting_for_response:
            self.response_wait_timer += dt
            self.loading_timer += dt
            if self.loading_timer >= 0.4:
                self.loading_dots = (self.loading_dots + 1) % 4
                self.loading_timer = 0

            if self.response_wait_timer >= OLLAMA_RESPONSE_MAX_WAIT:
                self.waiting_for_response = False
                self.history.append(("System",
                    "[Response timed out, using local reply]",
                    COLORS["ui_text_dim"]))
                self._use_fallback_dialogue(self._pending_player_msg)
                self._trim_history()
                self.scroll_offset = 0
                return

            # Poll for Ollama response
            if self.ollama_dialogue:
                result = self.ollama_dialogue.poll_response()
                if result:
                    self.waiting_for_response = False
                    self.response_wait_timer = 0.0
                    if result.get("success"):
                        response_text = result["response"]
                        emotion = result.get("emotion", {})
                        self.last_response_latency = float(result.get("latency", 0.0) or 0.0)

                        self.history.append((self.target_npc.name, response_text, COLORS["ui_chat_npc"]))

                        # Compute sentiment from emotion vector
                        sentiment = (emotion.get("joy", 0) + emotion.get("trust", 0)
                                   - emotion.get("anger", 0) - emotion.get("disgust", 0)
                                   - emotion.get("fear", 0))
                        sentiment = max(-1.0, min(1.0, sentiment))

                        # Apply effects on NPC
                        self.target_npc.apply_sentiment_effect(sentiment)

                        # Store in NPC dialogue history
                        self.target_npc.dialogue_history.append({
                            "player": self._pending_player_msg,
                            "npc": response_text,
                            "sentiment": sentiment,
                        })

                        # Record in emotion database
                        if self.emotion_database:
                            self.emotion_database.add_interaction(
                                self.target_npc,
                                self._pending_player_msg,
                                response_text,
                                emotion,
                                sentiment
                            )

                        if self.conversation_learner:
                            self.conversation_learner.log_exchange(
                                self.target_npc,
                                self._pending_player_msg,
                                response_text,
                                sentiment=sentiment,
                                emotion=emotion,
                                source="ollama",
                            )

                        # Show emotion reaction
                        dominant_emotion = max(emotion, key=emotion.get) if emotion else "curiosity"
                        emotion_emojis = {
                            "joy": "😊", "anger": "😠", "fear": "😨",
                            "sadness": "😢", "trust": "🤝", "surprise": "😲",
                            "disgust": "🤢", "curiosity": "🤔",
                        }
                        self.target_npc.show_emotion(
                            emotion_emojis.get(dominant_emotion, "🤔"), 3.0
                        )
                    else:
                        # Ollama failed — fallback
                        self.last_response_latency = 0.0
                        self.history.append(("System",
                            "[AI unavailable, using local reply]",
                            COLORS["ui_text_dim"]))
                        self._use_fallback_dialogue(self._pending_player_msg)

                    self._trim_history()
                    self.scroll_offset = 0

    def _idle_reaction_profile(self, npc):
        friendliness = float(npc.personality.get("friendliness", 0.5))
        aggression = float(npc.personality.get("aggression", 0.2))
        sociability = float(npc.personality.get("sociability", 0.5))
        trust = float(npc.behavior_vector.get("trust", 0.5))

        if friendliness > 0.72 and trust > 0.5:
            return "caring"
        if aggression > 0.55 and trust < 0.6:
            return "stern"
        if sociability < 0.35 and trust < 0.55:
            return "impatient"
        if friendliness < 0.4 and aggression > 0.35:
            return "sarcastic"
        if trust > 0.7:
            return "playful"
        return "neutral"

    def _idle_reaction_line(self, npc):
        cls = npc.npc_class
        profile = self._idle_reaction_profile(npc)

        by_profile = {
            "caring": [
                "Hey, is everything okay? You can take your time.",
                "You look troubled. Want to talk about what happened?",
                "No pressure, but I can listen if you need it.",
            ],
            "stern": [
                "Speak clearly. I do not have patience for games.",
                "If you called me over, say your point now.",
                "Stand straight and talk. What is the issue?",
            ],
            "impatient": [
                "Talk or let me get back to my work.",
                "I cannot stand here forever. Say it.",
                "If you have nothing to say, do not waste my time.",
            ],
            "sarcastic": [
                "You pressed me to chat just to stare? Bold strategy.",
                "What now, speech vanished?",
                "Are you drunk, or just collecting awkward silences today?",
            ],
            "playful": [
                "You look like you forgot your line in a play.",
                "I am waiting for the dramatic part.",
                "Take a breath. Start from the beginning.",
            ],
            "neutral": [
                "You alright? You look like something is off.",
                "Is something wrong? You can just say it.",
                "Are you okay... or just very tired today?",
            ],
        }

        class_flair = {
            "Royal": "Choose your words with care.",
            "Noble": "Compose yourself before you speak.",
            "Elite": "Focus. What do you need?",
            "Merchant": "Time is coin, friend. Speak up.",
            "Blacksmith": "I can wait, but not forever.",
            "Traveller": "Road dust in your head, maybe?",
            "Labourer": "I have work waiting. Talk or move.",
            "Peasant": "I will listen, but please say something.",
        }
        base = random.choice(by_profile.get(profile, by_profile["neutral"]))
        return f"{base} {class_flair.get(cls, '')}".strip(), profile

    def _update_idle_reaction(self, dt):
        if not self.target_npc or self.waiting_for_response or self.session_player_messages > 0:
            return

        self.idle_open_timer += dt
        threshold = 3.2 + (self.idle_prompt_count * 2.6)
        if self.idle_prompt_count < 2 and self.idle_open_timer >= threshold:
            line, profile = self._idle_reaction_line(self.target_npc)
            self.history.append((self.target_npc.name, line, COLORS["ui_chat_npc"]))
            self.idle_prompt_count += 1
            self.idle_open_timer = 0.0
            sentiment_hit = {
                "caring": -0.03,
                "playful": -0.04,
                "neutral": -0.06,
                "impatient": -0.08,
                "sarcastic": -0.09,
                "stern": -0.1,
            }.get(profile, -0.06)
            self.target_npc.apply_sentiment_effect(sentiment_hit)
            self.social_incidents.append({
                "type": "silent_prompt",
                "npc_name": self.target_npc.name,
                "npc_class": self.target_npc.npc_class,
                "style": profile,
                "severity": min(0.2, abs(sentiment_hit) + 0.04),
                "text": line,
            })
            self._trim_history()
            self.scroll_offset = 0

    def _handle_no_talk_close_reaction(self):
        npc = self.target_npc
        if not npc or self.session_player_messages > 0:
            return

        profile = self._idle_reaction_profile(npc)
        close_lines = {
            "caring": [
                "You left before saying anything. I hope you are okay.",
                "If something is wrong, you can tell me next time.",
            ],
            "playful": [
                "You started a conversation with silence and vanished. Impressive.",
                "That was the shortest conversation in village history.",
            ],
            "neutral": [
                "You walked up, said nothing, and left. That felt strange.",
                "Next time just say what is on your mind.",
            ],
            "impatient": [
                "Do not pull me into chat and disappear again.",
                "If you keep wasting my time, I will ignore you.",
            ],
            "sarcastic": [
                "Wonderful conversation. Truly deep. Zero words, full confusion.",
                "You sure you are not drunk? That made no sense.",
            ],
            "stern": [
                "If you summon me, speak. Do not repeat this.",
                "Enough games. Next time, be direct.",
            ],
        }

        escalation = min(2, self.idle_prompt_count)
        penalties = {
            "caring": (-0.04, -0.06, -0.08),
            "playful": (-0.05, -0.07, -0.1),
            "neutral": (-0.08, -0.1, -0.12),
            "impatient": (-0.1, -0.12, -0.15),
            "sarcastic": (-0.1, -0.13, -0.16),
            "stern": (-0.11, -0.14, -0.18),
        }
        line = random.choice(close_lines.get(profile, close_lines["neutral"]))
        penalty = penalties.get(profile, penalties["neutral"])[escalation]

        self.history.append((npc.name, line, COLORS["ui_chat_npc"]))
        npc.apply_sentiment_effect(penalty)
        npc.dialogue_history.append({
            "player": "[Opened chat, said nothing]",
            "npc": line,
            "sentiment": penalty,
        })
        if self.conversation_learner:
            self.conversation_learner.log_exchange(
                npc,
                "[Opened chat, said nothing]",
                line,
                sentiment=penalty,
                emotion=self._estimate_local_emotion(penalty),
                source="rule",
            )
        self.social_incidents.append({
            "type": "silent_close",
            "npc_name": npc.name,
            "npc_class": npc.npc_class,
            "style": profile,
            "severity": min(0.28, abs(penalty) + 0.05),
            "text": line,
        })

    def _trim_history(self):
        """Trim chat history to prevent memory issues."""
        if len(self.history) > CHAT_MAX_HISTORY * 2:
            self.history = self.history[-CHAT_MAX_HISTORY * 2:]

    def _estimate_local_emotion(self, sentiment):
        """Map scalar sentiment to a compact emotion vector for local research logs."""
        s = max(-1.0, min(1.0, float(sentiment)))
        joy = max(0.0, s)
        trust = max(0.0, s * 0.8)
        anger = max(0.0, -s * 0.7)
        fear = max(0.0, -s * 0.5)
        sadness = max(0.0, -s * 0.4)
        curiosity = 0.25 + (0.15 if abs(s) < 0.3 else 0.0)
        surprise = 0.15 + abs(s) * 0.2
        disgust = max(0.0, -s * 0.3)
        return {
            "joy": joy,
            "anger": anger,
            "fear": fear,
            "sadness": sadness,
            "trust": trust,
            "surprise": min(1.0, surprise),
            "disgust": disgust,
            "curiosity": min(1.0, curiosity),
        }

    def render(self, surface):
        """Render the chat box."""
        if not self.active or not self.font:
            return

        # Semi-transparent background
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 100), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        surface.blit(overlay, (0, 0))

        # Chat box background
        box_surf = pygame.Surface((self.box_width, self.box_height), pygame.SRCALPHA)
        pygame.draw.rect(box_surf, (25, 25, 40, 235),
                         (0, 0, self.box_width, self.box_height), border_radius=8)
        pygame.draw.rect(box_surf, COLORS["ui_border"],
                         (0, 0, self.box_width, self.box_height), 2, border_radius=8)

        # Title bar
        if self.target_npc:
            npc = self.target_npc
            title = f"Talking to {npc.name} ({npc.npc_class})"
            title_surf = self.title_font.render(title, True, COLORS["ui_accent"])
            box_surf.blit(title_surf, (self.padding, 8))

            # NPC mood/trust indicators
            mood_stars = int(npc.behavior_vector['mood'] * 5)
            trust_stars = int(npc.behavior_vector['trust'] * 5)
            mood_text = f"Mood: {'★' * mood_stars}{'☆' * (5 - mood_stars)}"
            trust_text = f"Trust: {'★' * trust_stars}{'☆' * (5 - trust_stars)}"
            mood_color = COLORS["ui_positive"] if npc.behavior_vector['mood'] > 0.5 else COLORS["ui_negative"]
            trust_color = COLORS["ui_positive"] if npc.behavior_vector['trust'] > 0.5 else COLORS["ui_warning"]
            mood_surf = self.small_font.render(mood_text, True, mood_color)
            trust_surf = self.small_font.render(trust_text, True, trust_color)
            box_surf.blit(mood_surf, (self.padding, 28))
            box_surf.blit(trust_surf, (self.padding + 160, 28))

        # Divider
        pygame.draw.line(box_surf, COLORS["ui_border"],
                         (self.padding, 48), (self.box_width - self.padding, 48))

        # Chat history area
        history_y = 52
        history_height = self.box_height - 105
        row_height = 16
        visible_lines = max(1, history_height // row_height)
        max_row_width = self.box_width - (2 * self.padding) - 6
        rows = self._build_wrapped_history_rows(max_row_width)

        max_scroll = max(0, len(rows) - visible_lines)
        self.scroll_offset = min(self.scroll_offset, max_scroll)

        end_idx = len(rows) - self.scroll_offset
        start_idx = max(0, end_idx - visible_lines)

        y = history_y
        for sender, text, color in rows[start_idx:end_idx]:
            if sender:
                name_surf = self.small_font.render(f"{sender}: ", True, color)
                box_surf.blit(name_surf, (self.padding, y))
                msg_x = self.padding + name_surf.get_width()
            else:
                msg_x = self.padding + 4

            line_surf = self.font.render(text, True, COLORS["ui_text"])
            box_surf.blit(line_surf, (msg_x, y))
            y += row_height

        # Loading indicator when waiting for Ollama
        if self.waiting_for_response:
            loading_text = f"{self.target_npc.name} is thinking{'.' * self.loading_dots}"
            loading_surf = self.font.render(loading_text, True, COLORS["ui_accent"])
            box_surf.blit(loading_surf, (self.padding, y))

        # Input area
        input_y = self.box_height - self.input_height - self.padding - 14
        pygame.draw.rect(box_surf, (40, 40, 55),
                         (self.padding, input_y, self.box_width - 2 * self.padding, self.input_height),
                         border_radius=4)

        border_color = COLORS["ui_text_dim"] if self.waiting_for_response else COLORS["ui_accent"]
        pygame.draw.rect(box_surf, border_color,
                         (self.padding, input_y, self.box_width - 2 * self.padding, self.input_height),
                         1, border_radius=4)

        # Input text
        if self.waiting_for_response:
            display_text = "Waiting for response..."
            text_color = COLORS["ui_text_dim"]
        else:
            display_text = self.input_text
            if self.cursor_visible:
                display_text += "|"
            text_color = COLORS["ui_text"]

        input_surf = self.font.render(display_text, True, text_color)
        box_surf.blit(input_surf, (self.padding + 6, input_y + 8))

        # Hint text
        hint = "ENTER: Send | ESC: Close | ↑↓/PgUp/PgDn/Wheel: Scroll"
        hint_surf = self.small_font.render(hint, True, COLORS["ui_text_dim"])
        box_surf.blit(hint_surf, (self.padding, self.box_height - self.padding + 1))

        # Emotion database hint
        if self.emotion_database:
            stats = self.emotion_database.get_stats_for_ui()
            db_hint = f"Interactions: {stats['total_interactions']} | Player: {stats['player_tendency']}"
            db_surf = self.small_font.render(db_hint, True, COLORS["ui_text_dim"])
            box_surf.blit(db_surf, (self.box_width - db_surf.get_width() - self.padding,
                                     self.box_height - self.padding + 1))

        surface.blit(box_surf, (self.box_x, self.box_y))
