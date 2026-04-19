"""
Online conversation logging and lightweight language learning.

This module stores every conversation exchange in a CSV file and incrementally
trains a per-cluster Markov language model from NPC responses.
"""
import csv
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime

from config import (
    CONVERSATION_CSV_PATH,
    CONVERSATION_FINE_TUNE_DIR,
    CONVERSATION_LEARNING_ENABLED,
    CONVERSATION_EXPORT_EVERY_N_ROWS,
    CONVERSATION_EXPORT_MIN_ROWS,
    CONVERSATION_MODEL_MIN_WORDS,
    CONVERSATION_MODEL_MAX_WORDS,
    CONVERSATION_RECENT_MEMORY,
    CONVERSATION_TRAIN_SPLIT,
    CONVERSATION_CSV_REFRESH_INTERVAL,
    CONVERSATION_ADAPT_EVERY_N_ROWS,
    CONVERSATION_REPETITION_THRESHOLD_INITIAL,
    CONVERSATION_REPETITION_THRESHOLD_MIN,
    CONVERSATION_REPETITION_THRESHOLD_MAX,
    CONVERSATION_PATTERN_MIX_PROB,
    ADAPTIVE_PROFILE_PRESETS,
)


class ConversationLearningModel:
    """Logs chat exchanges and trains a lightweight online response model."""

    FIELDNAMES = [
        "timestamp",
        "npc_name",
        "npc_class",
        "cluster_id",
        "source",
        "player_message",
        "npc_response",
        "sentiment",
        "emotion_json",
    ]
    PRETTY_FIELDNAMES = [
        "timestamp",
        "source",
        "speaker",
        "speaker_class",
        "target",
        "target_class",
        "tone",
        "cluster_key",
        "cluster_label",
        "sentiment",
        "text",
    ]

    def __init__(self, csv_path=CONVERSATION_CSV_PATH):
        self.enabled = bool(CONVERSATION_LEARNING_ENABLED)
        self.csv_path = os.path.abspath(csv_path)
        root, ext = os.path.splitext(self.csv_path)
        self.pretty_csv_path = f"{root}_pretty{ext or '.csv'}"
        self.min_words = max(3, int(CONVERSATION_MODEL_MIN_WORDS))
        self.max_words = max(self.min_words + 2, int(CONVERSATION_MODEL_MAX_WORDS))
        self.recent_memory = max(16, int(CONVERSATION_RECENT_MEMORY))
        self.fine_tune_dir = os.path.abspath(CONVERSATION_FINE_TUNE_DIR)
        self.export_every_n_rows = max(1, int(CONVERSATION_EXPORT_EVERY_N_ROWS))
        self.export_min_rows = max(1, int(CONVERSATION_EXPORT_MIN_ROWS))
        self.train_split = min(0.98, max(0.5, float(CONVERSATION_TRAIN_SPLIT)))

        self._chain = defaultdict(lambda: defaultdict(Counter))
        self._starters = defaultdict(Counter)
        self._pattern_chain = defaultdict(lambda: defaultdict(Counter))
        self._pattern_starters = defaultdict(Counter)
        self._recent_by_npc = defaultdict(lambda: deque(maxlen=self.recent_memory))
        self._recent_by_cluster = defaultdict(lambda: deque(maxlen=self.recent_memory * 3))
        self.total_rows = 0
        self._export_lock = threading.Lock()
        self._export_in_progress = False
        self.repetition_threshold = float(CONVERSATION_REPETITION_THRESHOLD_INITIAL)
        self.repetition_threshold_min = float(CONVERSATION_REPETITION_THRESHOLD_MIN)
        self.repetition_threshold_max = float(CONVERSATION_REPETITION_THRESHOLD_MAX)
        self.csv_refresh_interval = float(CONVERSATION_CSV_REFRESH_INTERVAL)
        self.adapt_every_n_rows = int(CONVERSATION_ADAPT_EVERY_N_ROWS)
        self.pattern_mix_prob = float(CONVERSATION_PATTERN_MIX_PROB)
        self.runtime_profile = "custom"
        self._last_adapt_rows = 0
        self._last_csv_refresh_ts = 0.0
        self._file_offsets = {}

        self._ensure_csv()
        self._load_existing_csv()
        self._load_existing_pretty_csv()

    def set_adaptation_profile(self, profile_name):
        preset = dict((ADAPTIVE_PROFILE_PRESETS or {}).get(profile_name, {}))
        if not preset:
            return

        self.runtime_profile = str(profile_name)
        self.csv_refresh_interval = float(preset.get("csv_refresh_interval", self.csv_refresh_interval))
        self.adapt_every_n_rows = max(1, int(preset.get("adapt_every_n_rows", self.adapt_every_n_rows)))
        self.pattern_mix_prob = max(0.0, min(1.0, float(preset.get("pattern_mix_prob", self.pattern_mix_prob))))

        self.repetition_threshold_min = float(preset.get("repetition_threshold_min", self.repetition_threshold_min))
        self.repetition_threshold_max = float(preset.get("repetition_threshold_max", self.repetition_threshold_max))
        target_rep = float(preset.get("repetition_threshold_initial", self.repetition_threshold))
        self.repetition_threshold = max(self.repetition_threshold_min, min(self.repetition_threshold_max, target_rep))

    def _ensure_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
        if not os.path.exists(self.pretty_csv_path):
            with open(self.pretty_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.PRETTY_FIELDNAMES)
                writer.writeheader()

    def _append_pretty_row(
        self,
        source,
        speaker,
        speaker_class,
        target,
        target_class,
        tone,
        cluster_key,
        cluster_label,
        sentiment,
        text,
    ):
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source": str(source or "local"),
            "speaker": str(speaker or "NPC"),
            "speaker_class": str(speaker_class or "Unknown"),
            "target": str(target or ""),
            "target_class": str(target_class or ""),
            "tone": str(tone or ""),
            "cluster_key": str(cluster_key or ""),
            "cluster_label": str(cluster_label or ""),
            "sentiment": f"{float(sentiment):.4f}",
            "text": str(text or ""),
        }
        try:
            with open(self.pretty_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.PRETTY_FIELDNAMES)
                writer.writerow(row)
            self._file_offsets[self.pretty_csv_path] = int(self._file_offsets.get(self.pretty_csv_path, 0)) + 1
        except Exception:
            return

    def log_social_line(
        self,
        speaker,
        text,
        tone,
        sentiment=0.0,
        target=None,
        cluster_key=None,
        cluster_label=None,
        source="npc_social",
    ):
        """Append one NPC social line to the pretty live CSV."""
        if not self.enabled or speaker is None:
            return
        self._append_pretty_row(
            source=source,
            speaker=getattr(speaker, "name", "NPC"),
            speaker_class=getattr(speaker, "npc_class", "Unknown"),
            target=getattr(target, "name", "") if target is not None else "",
            target_class=getattr(target, "npc_class", "") if target is not None else "",
            tone=tone,
            cluster_key=cluster_key,
            cluster_label=cluster_label,
            sentiment=sentiment,
            text=text,
        )

    def _tokenize(self, text):
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    def _behavior_key(self, behavior_pattern):
        key = str(behavior_pattern or "").strip().lower()
        return key if key else "balanced"

    def _signature(self, text):
        return " ".join(self._tokenize(text))

    def _jaccard(self, a_sig, b_sig):
        a = set(a_sig.split())
        b = set(b_sig.split())
        if not a or not b:
            return 0.0
        return len(a & b) / float(len(a | b))

    def _cluster_id(self, npc):
        return int(getattr(npc, "cluster_id", -1) or -1)

    def _train_sentence(self, cluster_id, sentence, behavior_pattern=None):
        tokens = self._tokenize(sentence)
        if len(tokens) < 2:
            return

        starter = (tokens[0], tokens[1])
        self._starters[cluster_id][starter] += 1
        pattern_key = self._behavior_key(behavior_pattern)
        self._pattern_starters[pattern_key][starter] += 1

        for i in range(2, len(tokens) + 1):
            prev = (tokens[i - 2], tokens[i - 1])
            nxt = tokens[i] if i < len(tokens) else "<e>"
            self._chain[cluster_id][prev][nxt] += 1
            self._pattern_chain[pattern_key][prev][nxt] += 1

    def _record_recent(self, npc_name, cluster_id, text):
        sig = self._signature(text)
        if not sig:
            return
        self._recent_by_npc[npc_name].append(sig)
        self._recent_by_cluster[cluster_id].append(sig)

    def _load_existing_csv(self):
        if not os.path.exists(self.csv_path):
            return

        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    response = row.get("npc_response", "")
                    npc_name = row.get("npc_name", "Unknown")
                    behavior_pattern = row.get("behavior_pattern", "")
                    try:
                        cluster_id = int(row.get("cluster_id", -1))
                    except (TypeError, ValueError):
                        cluster_id = -1

                    self._train_sentence(cluster_id, response, behavior_pattern=behavior_pattern)
                    self._train_sentence(-1, response, behavior_pattern=behavior_pattern)
                    self._record_recent(npc_name, cluster_id, response)
                    self.total_rows += 1
                self._file_offsets[self.csv_path] = self.total_rows
        except Exception:
            # Keep gameplay resilient if CSV data is malformed.
            self.total_rows = 0

    def _load_existing_pretty_csv(self):
        if not os.path.exists(self.pretty_csv_path):
            return
        try:
            loaded = 0
            with open(self.pretty_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = (row.get("text") or "").strip()
                    if not text:
                        continue
                    self._train_sentence(-1, text, behavior_pattern="balanced")
                    loaded += 1
            self._file_offsets[self.pretty_csv_path] = loaded
        except Exception:
            return

    def _adapt_model_knobs(self):
        """Self-adjust generation constraints from recent dialogue quality signals."""
        sample = list(self._recent_by_cluster.get(-1, []))[-120:]
        if len(sample) < 20:
            return

        lens = [len(sig.split()) for sig in sample if sig]
        if not lens:
            return

        avg_len = sum(lens) / float(len(lens))
        unique_ratio = len(set(sample)) / float(max(1, len(sample)))

        target_min = int(max(4, min(14, round(avg_len * 0.55))))
        target_max = int(max(target_min + 2, min(30, round(avg_len * 1.7))))
        self.min_words = int(max(3, min(20, (self.min_words * 0.65) + (target_min * 0.35))))
        self.max_words = int(max(self.min_words + 2, min(34, (self.max_words * 0.65) + (target_max * 0.35))))

        # More repetition in recent logs => stricter near-duplicate filtering.
        if unique_ratio < 0.62:
            self.repetition_threshold = max(self.repetition_threshold_min, self.repetition_threshold - 0.03)
        elif unique_ratio > 0.82:
            self.repetition_threshold = min(self.repetition_threshold_max, self.repetition_threshold + 0.02)

    def refresh_from_live_csv(self):
        """Ingest rows appended by external tools without restarting runtime."""
        now = time.time()
        if (now - self._last_csv_refresh_ts) < self.csv_refresh_interval:
            return
        self._last_csv_refresh_ts = now

        for path in (self.csv_path, self.pretty_csv_path):
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    start = int(self._file_offsets.get(path, 0))
                    end_offset = start
                    for idx, row in enumerate(reader):
                        end_offset = idx + 1
                        if idx < start:
                            continue
                        if path == self.csv_path:
                            response = (row.get("npc_response") or "").strip()
                            if not response:
                                continue
                            npc_name = row.get("npc_name", "Unknown")
                            behavior_pattern = row.get("behavior_pattern", "")
                            try:
                                cluster_id = int(row.get("cluster_id", -1))
                            except (TypeError, ValueError):
                                cluster_id = -1
                            self._train_sentence(cluster_id, response, behavior_pattern=behavior_pattern)
                            self._train_sentence(-1, response, behavior_pattern=behavior_pattern)
                            self._record_recent(npc_name, cluster_id, response)
                            self.total_rows += 1
                        else:
                            text = (row.get("text") or "").strip()
                            if text:
                                self._train_sentence(-1, text, behavior_pattern="balanced")
                    self._file_offsets[path] = end_offset
            except Exception:
                continue

        if (self.total_rows - self._last_adapt_rows) >= self.adapt_every_n_rows:
            self._last_adapt_rows = self.total_rows
            self._adapt_model_knobs()

    def _weighted_choice(self, counter):
        if not counter:
            return None
        items = list(counter.items())
        keys = [k for k, _ in items]
        weights = [w for _, w in items]
        return random.choices(keys, weights=weights, k=1)[0]

    def generate_cluster_sentence(self, npc, avoid_signatures=None):
        if not self.enabled:
            return None

        cluster_id = self._cluster_id(npc)
        behavior_pattern = self._behavior_key(getattr(npc, "behavior_pattern", ""))
        cluster_starters = self._starters.get(cluster_id) or self._starters.get(-1)
        cluster_chain = self._chain.get(cluster_id) or self._chain.get(-1)
        pattern_starters = self._pattern_starters.get(behavior_pattern)
        pattern_chain = self._pattern_chain.get(behavior_pattern)

        use_pattern = bool(pattern_starters and pattern_chain and random.random() < self.pattern_mix_prob)
        starters = pattern_starters if use_pattern else cluster_starters
        chain = pattern_chain if use_pattern else cluster_chain

        if not starters or not chain:
            return None

        avoid = set(avoid_signatures or [])

        for _ in range(6):
            starter = self._weighted_choice(starters)
            if not starter:
                return None

            words = [starter[0], starter[1]]
            while len(words) < self.max_words:
                prev = (words[-2], words[-1])
                next_counter = chain.get(prev)
                if not next_counter:
                    break
                nxt = self._weighted_choice(next_counter)
                if not nxt:
                    break
                if nxt == "<e>":
                    if len(words) >= self.min_words:
                        break
                    non_end = Counter({k: v for k, v in next_counter.items() if k != "<e>"})
                    nxt = self._weighted_choice(non_end)
                    if not nxt:
                        break
                words.append(nxt)

            sentence = " ".join(words).strip()
            if not sentence:
                continue

            sentence = sentence[0].upper() + sentence[1:]
            if sentence[-1] not in ".!?":
                sentence += "."

            sig = self._signature(sentence)
            if sig and sig not in avoid:
                return sentence

        return None

    def is_repetitive(self, npc, candidate_text, threshold=None):
        if threshold is None:
            threshold = float(self.repetition_threshold)
        sig = self._signature(candidate_text)
        if not sig:
            return False

        recent_sigs = list(self._recent_by_npc.get(npc.name, []))
        recent_sigs.extend(list(self._recent_by_cluster.get(self._cluster_id(npc), [])))

        for entry in getattr(npc, "dialogue_history", [])[-10:]:
            hist_sig = self._signature(entry.get("npc", ""))
            if hist_sig:
                recent_sigs.append(hist_sig)

        for seen in recent_sigs:
            if sig == seen:
                return True
            if self._jaccard(sig, seen) >= threshold:
                return True
        return False

    def refine_response(self, npc, player_message, response_text):
        _ = player_message
        if not self.enabled or not response_text:
            return response_text

        self.refresh_from_live_csv()

        if not self.is_repetitive(npc, response_text):
            return response_text

        avoid = set(self._recent_by_npc.get(npc.name, []))
        replacement = self.generate_cluster_sentence(npc, avoid_signatures=avoid)
        return replacement or response_text

    def build_generation_hint(self, npc):
        if not self.enabled:
            return ""

        self.refresh_from_live_csv()

        avoid = set(self._recent_by_npc.get(npc.name, []))
        examples = []
        for _ in range(2):
            sample = self.generate_cluster_sentence(npc, avoid_signatures=avoid)
            if not sample:
                continue
            sig = self._signature(sample)
            avoid.add(sig)
            examples.append(sample)

        if not examples:
            return ""

        joined = " | ".join(examples)
        return (
            "Style memory from this NPC cluster (use as tone guidance, do not copy exactly): "
            f"{joined}"
        )

    def _iter_log_rows(self):
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    def _example_from_row(self, row):
        player = (row.get("player_message") or "").strip()
        npc = (row.get("npc_response") or "").strip()
        npc_name = (row.get("npc_name") or "NPC").strip()
        npc_class = (row.get("npc_class") or "Villager").strip()
        source = (row.get("source") or "local").strip()

        if not player or not npc:
            return None
        if player.startswith("[") and player.endswith("]"):
            return None

        system_text = (
            f"You are {npc_name}, a {npc_class} NPC in a medieval village sim. "
            "Reply naturally in one short sentence without repeating prior wording."
        )

        prompt = (
            f"NPC: {npc_name} ({npc_class})\n"
            f"Source: {source}\n"
            f"Player: {player}\n"
            "NPC:"
        )

        signature = f"{self._signature(player)}||{self._signature(npc)}"
        if not signature.strip("|"):
            return None

        return {
            "signature": signature,
            "prompt_completion": {
                "prompt": prompt,
                "completion": f" {npc}",
            },
            "chat": {
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": player},
                    {"role": "assistant", "content": npc},
                ]
            },
        }

    def export_fine_tune_datasets(self, force=False):
        if not self.enabled:
            return {"exported": False, "reason": "disabled"}

        rows = list(self._iter_log_rows())
        if not force and len(rows) < self.export_min_rows:
            return {"exported": False, "reason": "not_enough_rows", "rows": len(rows)}

        examples = []
        seen = set()
        for row in rows:
            ex = self._example_from_row(row)
            if not ex:
                continue
            sig = ex["signature"]
            if sig in seen:
                continue
            seen.add(sig)
            examples.append(ex)

        if len(examples) < 10:
            return {"exported": False, "reason": "not_enough_examples", "rows": len(rows)}

        rng = random.Random(42)
        rng.shuffle(examples)

        split_idx = int(len(examples) * self.train_split)
        split_idx = max(1, min(len(examples) - 1, split_idx))
        train = examples[:split_idx]
        val = examples[split_idx:]

        os.makedirs(self.fine_tune_dir, exist_ok=True)
        out_train_pc = os.path.join(self.fine_tune_dir, "finetune_train_prompt_completion.jsonl")
        out_val_pc = os.path.join(self.fine_tune_dir, "finetune_val_prompt_completion.jsonl")
        out_train_chat = os.path.join(self.fine_tune_dir, "finetune_train_chat.jsonl")
        out_val_chat = os.path.join(self.fine_tune_dir, "finetune_val_chat.jsonl")
        out_stats = os.path.join(self.fine_tune_dir, "finetune_stats.json")

        with open(out_train_pc, "w", encoding="utf-8") as f:
            for ex in train:
                f.write(json.dumps(ex["prompt_completion"], ensure_ascii=False) + "\n")

        with open(out_val_pc, "w", encoding="utf-8") as f:
            for ex in val:
                f.write(json.dumps(ex["prompt_completion"], ensure_ascii=False) + "\n")

        with open(out_train_chat, "w", encoding="utf-8") as f:
            for ex in train:
                f.write(json.dumps(ex["chat"], ensure_ascii=False) + "\n")

        with open(out_val_chat, "w", encoding="utf-8") as f:
            for ex in val:
                f.write(json.dumps(ex["chat"], ensure_ascii=False) + "\n")

        stats = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "csv_rows": len(rows),
            "examples_total": len(examples),
            "train_examples": len(train),
            "val_examples": len(val),
            "train_split": self.train_split,
            "files": {
                "train_prompt_completion": out_train_pc,
                "val_prompt_completion": out_val_pc,
                "train_chat": out_train_chat,
                "val_chat": out_val_chat,
            },
        }
        with open(out_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return {"exported": True, **stats}

    def _schedule_export(self):
        if self._export_in_progress:
            return

        def worker():
            with self._export_lock:
                self._export_in_progress = True
                try:
                    self.export_fine_tune_datasets(force=False)
                finally:
                    self._export_in_progress = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def log_exchange(self, npc, player_message, npc_response, sentiment=0.0, emotion=None, source="local"):
        if not self.enabled:
            return

        cluster_id = self._cluster_id(npc)
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "npc_name": npc.name,
            "npc_class": npc.npc_class,
            "cluster_id": cluster_id,
            "source": source,
            "player_message": player_message,
            "npc_response": npc_response,
            "sentiment": f"{float(sentiment):.4f}",
            "emotion_json": json.dumps(emotion or {}, separators=(",", ":")),
        }

        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)
        except Exception:
            return

        self.total_rows += 1
        behavior_pattern = getattr(npc, "behavior_pattern", "balanced")
        self._train_sentence(cluster_id, npc_response, behavior_pattern=behavior_pattern)
        self._train_sentence(-1, npc_response, behavior_pattern=behavior_pattern)
        self._record_recent(npc.name, cluster_id, npc_response)
        self._file_offsets[self.csv_path] = int(self._file_offsets.get(self.csv_path, 0)) + 1

        if (self.total_rows - self._last_adapt_rows) >= self.adapt_every_n_rows:
            self._last_adapt_rows = self.total_rows
            self._adapt_model_knobs()

        self._append_pretty_row(
            source=source,
            speaker=npc.name,
            speaker_class=npc.npc_class,
            target="Player" if source != "npc_social" else "",
            target_class="Human" if source != "npc_social" else "",
            tone="",
            cluster_key="",
            cluster_label="",
            sentiment=sentiment,
            text=npc_response,
        )

        if self.total_rows >= self.export_min_rows and self.total_rows % self.export_every_n_rows == 0:
            self._schedule_export()
