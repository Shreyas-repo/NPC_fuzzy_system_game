"""Village economy simulation: farms, barter, production, and fuzzy taxation."""
import random
from config import TILE_SIZE, ZONES


class EconomySystem:
    """Tracks local economy signals and class-driven production."""

    def __init__(self):
        self.farm_zones = sorted([z for z in ZONES if z.startswith("wheat_farm_")])
        self.farm_owners = {}  # zone -> {owner_name, owner_class}
        self.accounts = {}  # npc_id -> balances
        self.tax_collectors = {}  # collector_name -> {corruption, strictness}

        self.timers = {
            "production": 0.0,
            "barter": 0.0,
            "tax": 0.0,
            "climate": 0.0,
        }
        self.intervals = {
            "production": 3.0,
            "barter": 4.0,
            "tax": 12.0,
            "climate": 9.0,
        }

        self.ledger = {
            "treasury": 0.0,
            "grain_output": 0.0,
            "market_turnover": 0.0,
            "barter_count": 0,
            "last_tax_rate": 0.0,
            "tax_collected_last_round": 0.0,
            "corruption_skim_last_round": 0.0,
            "corrupt_collections": 0,
            "tax_collector_count": 0,
        }

        self.recent_events = []
        self.max_events = 40
        self.last_climate_bucket = "steady"

    def update(self, dt, npcs, soft_controller=None, social_system=None):
        if not npcs:
            return

        self._ensure_owners(npcs)
        self._ensure_accounts(npcs)
        self._ensure_tax_collectors(npcs)

        for key in self.timers:
            self.timers[key] += dt

        if self.timers["production"] >= self.intervals["production"]:
            self.timers["production"] = 0.0
            self._run_production(npcs)

        if self.timers["barter"] >= self.intervals["barter"]:
            self.timers["barter"] = 0.0
            self._run_barter(npcs)

        if self.timers["tax"] >= self.intervals["tax"]:
            self.timers["tax"] = 0.0
            self._collect_tax(npcs, soft_controller, social_system)

        if self.timers["climate"] >= self.intervals["climate"]:
            self.timers["climate"] = 0.0
            self._apply_economic_climate_effects(npcs, social_system)

    def _economic_climate(self):
        grain = max(0.0, min(1.0, self.ledger.get("grain_output", 0.0) / 18.0))
        turnover = max(0.0, min(1.0, self.ledger.get("market_turnover", 0.0) / 16.0))
        tax = max(0.0, min(1.0, self.ledger.get("last_tax_rate", 0.0) / 0.26))

        prosperity = 0.55 * grain + 0.45 * turnover
        stress = 0.45 * (1.0 - grain) + 0.25 * (1.0 - turnover) + 0.30 * tax
        return {
            "prosperity": max(0.0, min(1.0, prosperity)),
            "stress": max(0.0, min(1.0, stress)),
        }

    def _apply_economic_climate_effects(self, npcs, social_system):
        climate = self._economic_climate()
        prosperity = climate["prosperity"]
        stress = climate["stress"]

        if stress >= 0.62:
            bucket = "stress"
        elif prosperity >= 0.58:
            bucket = "prosperity"
        else:
            bucket = "steady"

        for npc in npcs:
            class_sensitivity = {
                "Peasant": 1.15,
                "Labourer": 1.1,
                "Traveller": 0.95,
                "Merchant": 0.85,
                "Blacksmith": 0.9,
                "Noble": 0.7,
                "Elite": 0.6,
                "Royal": 0.5,
            }.get(npc.npc_class, 1.0)

            mood_delta = (prosperity - stress) * 0.028 * class_sensitivity
            trust_delta = (prosperity - stress) * 0.022 * class_sensitivity
            npc.behavior_vector["mood"] = max(0.0, min(1.0, npc.behavior_vector.get("mood", 0.5) + mood_delta))
            npc.behavior_vector["trust"] = max(0.0, min(1.0, npc.behavior_vector.get("trust", 0.5) + trust_delta))

            # Economy shifts how NPCs treat each other in social interactions.
            if bucket == "stress":
                npc.personality["aggression"] = min(1.0, npc.personality.get("aggression", 0.2) + 0.004 * class_sensitivity)
                npc.personality["friendliness"] = max(0.0, npc.personality.get("friendliness", 0.5) - 0.003 * class_sensitivity)
                npc.personality["sociability"] = max(0.0, npc.personality.get("sociability", 0.5) - 0.0025 * class_sensitivity)
            elif bucket == "prosperity":
                npc.personality["aggression"] = max(0.0, npc.personality.get("aggression", 0.2) - 0.003)
                npc.personality["friendliness"] = min(1.0, npc.personality.get("friendliness", 0.5) + 0.003)
                npc.personality["sociability"] = min(1.0, npc.personality.get("sociability", 0.5) + 0.002)

        if social_system is not None and bucket != self.last_climate_bucket:
            if bucket == "stress":
                social_system.remember_group_event(
                    npcs,
                    "Economic hardship raised tensions and fear",
                    trust_shift=-0.012,
                    mood_shift=-0.018,
                    negative=True,
                )
                self._event("Economy stress: shortages and high pressure are spreading")
            elif bucket == "prosperity":
                social_system.remember_group_event(
                    npcs,
                    "Strong harvest and trade improved confidence",
                    trust_shift=0.012,
                    mood_shift=0.015,
                )
                self._event("Economy growth: stable grain and trade improved morale")
            else:
                self._event("Economy steady: market and grain remain balanced")

        self.last_climate_bucket = bucket

    def _ensure_owners(self, npcs):
        if self.farm_owners or not self.farm_zones:
            return

        owners = [n for n in npcs if n.npc_class in ("Noble", "Elite")]
        if not owners:
            owners = [n for n in npcs if n.npc_class in ("Royal", "Merchant")]
        if not owners:
            return

        owners = sorted(owners, key=lambda n: (n.npc_class, n.name))
        for idx, farm in enumerate(self.farm_zones):
            owner = owners[idx % len(owners)]
            self.farm_owners[farm] = {
                "owner_name": owner.name,
                "owner_class": owner.npc_class,
            }
            short_farm = farm.replace("wheat_farm_", "farm-")
            self._event(f"{short_farm} owned by {owner.name} ({owner.npc_class})")

    def _ensure_accounts(self, npcs):
        for npc in npcs:
            key = id(npc)
            if key in self.accounts:
                continue
            wealth = float(npc.behavior_vector.get("wealth", 0.5))
            self.accounts[key] = {
                "coin": 8.0 + wealth * 18.0,
                "grain": 1.5 + max(0.0, 0.6 - wealth) * 6.0,
                "tools": 0.5 + wealth * 1.8,
                "metal": 0.2 + (1.2 if npc.npc_class == "Blacksmith" else wealth * 0.5),
            }

    def _npc_zone(self, npc):
        tx = int(npc.x // TILE_SIZE)
        ty = int(npc.y // TILE_SIZE)
        for zone, (zx, zy, zw, zh) in ZONES.items():
            if zx <= tx < zx + zw and zy <= ty < zy + zh:
                return zone
        return None

    def _event(self, text):
        self.recent_events.append(text)
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]

    def _add_owner_rent(self, farm_zone, amount, npcs):
        owner_meta = self.farm_owners.get(farm_zone)
        if not owner_meta:
            return
        owner_name = owner_meta["owner_name"]
        owner = next((n for n in npcs if n.name == owner_name), None)
        if not owner:
            return
        self.accounts[id(owner)]["coin"] += amount

    def _run_production(self, npcs):
        produced = 0.0
        turnover = 0.0

        for npc in npcs:
            acc = self.accounts[id(npc)]
            energy = float(npc.needs.get("energy", 0.8))
            work_ethic = float(npc.personality.get("work_ethic", 0.6))

            if npc.npc_class == "Blacksmith":
                metal_out = max(0.0, 0.6 + work_ethic * 0.9 + random.uniform(-0.15, 0.25))
                acc["metal"] += metal_out
                craft = min(acc["metal"] * 0.45, 1.1)
                if craft > 0.0:
                    acc["metal"] -= craft
                    acc["tools"] += craft
                    acc["coin"] += craft * 1.5
                    turnover += craft * 1.5

            if npc.npc_class in ("Merchant", "Traveller"):
                sold_grain = min(acc["grain"] * 0.28, 1.8)
                sold_tools = min(acc["tools"] * 0.22, 1.2)
                if sold_grain > 0:
                    acc["grain"] -= sold_grain
                    acc["coin"] += sold_grain * 1.1
                    turnover += sold_grain * 1.1
                if sold_tools > 0:
                    acc["tools"] -= sold_tools
                    acc["coin"] += sold_tools * 2.2
                    turnover += sold_tools * 2.2

        self.ledger["grain_output"] = 0.95 * self.ledger["grain_output"] + 0.05 * produced
        self.ledger["market_turnover"] = 0.8 * self.ledger["market_turnover"] + 0.2 * turnover

        if produced > 0.3 and random.random() < 0.35:
            self._event(f"Farm yield +{produced:.1f} wheat this cycle")

    def _trade_pair(self, a, b):
        acc_a = self.accounts[id(a)]
        acc_b = self.accounts[id(b)]

        # Grain-for-tool barter
        if acc_a["grain"] > 2.2 and acc_b["tools"] > 0.4:
            grain_qty = min(1.2, acc_a["grain"] - 1.5)
            tool_qty = min(0.5, acc_b["tools"])
            if grain_qty > 0.2 and tool_qty > 0.05:
                acc_a["grain"] -= grain_qty
                acc_b["grain"] += grain_qty
                acc_b["tools"] -= tool_qty
                acc_a["tools"] += tool_qty
                self.ledger["barter_count"] += 1
                self._event(f"Barter: {a.name} traded wheat with {b.name}")
                a.show_speech("Wheat for tools? Fair trade.", 2.8)
                b.show_speech("Deal. Keep the grain coming.", 2.8)
                return True

        if acc_b["grain"] > 2.2 and acc_a["tools"] > 0.4:
            grain_qty = min(1.2, acc_b["grain"] - 1.5)
            tool_qty = min(0.5, acc_a["tools"])
            if grain_qty > 0.2 and tool_qty > 0.05:
                acc_b["grain"] -= grain_qty
                acc_a["grain"] += grain_qty
                acc_a["tools"] -= tool_qty
                acc_b["tools"] += tool_qty
                self.ledger["barter_count"] += 1
                self._event(f"Barter: {b.name} traded wheat with {a.name}")
                b.show_speech("Wheat for tools? Fair trade.", 2.8)
                a.show_speech("Deal. Keep the grain coming.", 2.8)
                return True

        return False

    def _run_barter(self, npcs):
        for i, npc_a in enumerate(npcs):
            if npc_a.npc_class not in ("Peasant", "Labourer", "Merchant", "Traveller", "Blacksmith"):
                continue
            for j in range(i + 1, len(npcs)):
                npc_b = npcs[j]
                if npc_b.npc_class not in ("Peasant", "Labourer", "Merchant", "Traveller", "Blacksmith"):
                    continue

                dx = npc_a.x - npc_b.x
                dy = npc_a.y - npc_b.y
                if dx * dx + dy * dy > (70 * 70):
                    continue

                if self._trade_pair(npc_a, npc_b):
                    return

    def _is_tax_exempt(self, npc):
        # Guards are represented by Elite class; queen is exempt by name.
        if npc.npc_class == "Elite":
            return True
        if npc.npc_class == "Royal" and "queen" in npc.name.lower():
            return True
        return False

    def _ensure_tax_collectors(self, npcs):
        elites = sorted([n for n in npcs if n.npc_class == "Elite"], key=lambda n: n.name)
        if not elites:
            self.tax_collectors = {}
            self.ledger["tax_collector_count"] = 0
            return

        desired = max(1, len(elites) // 2)
        existing = {name: data for name, data in self.tax_collectors.items() if any(e.name == name for e in elites)}
        if len(existing) >= desired:
            self.tax_collectors = existing
            self.ledger["tax_collector_count"] = len(self.tax_collectors)
            return

        used = set(existing.keys())
        for elite in elites:
            if len(existing) >= desired:
                break
            if elite.name in used:
                continue
            corruption = random.uniform(0.08, 0.52)
            strictness = random.uniform(0.4, 0.95)
            existing[elite.name] = {
                "corruption": corruption,
                "strictness": strictness,
            }
            if corruption > 0.35:
                self._event(f"Duty assigned: {elite.name} appointed tax collector (rumors of corruption)")
            else:
                self._event(f"Duty assigned: {elite.name} appointed tax collector")

        self.tax_collectors = existing
        self.ledger["tax_collector_count"] = len(self.tax_collectors)

    def _tax_due_for(self, npc, acc, rate):
        class_mult = {
            "Royal": 1.25,      # king remains taxable unless queen
            "Noble": 1.15,
            "Merchant": 1.1,
            "Blacksmith": 1.0,
            "Traveller": 0.92,
            "Labourer": 0.72,
            "Peasant": 0.55,
        }.get(npc.npc_class, 0.9)
        base_due = acc["coin"] * rate * class_mult
        floor = 0.18 if npc.npc_class in ("Labourer", "Peasant") else 0.3
        cap = 6.5 if npc.npc_class in ("Royal", "Noble", "Merchant") else 3.2
        due = min(cap, max(floor, base_due))
        return min(due, max(0.0, acc["coin"] - 0.08))

    def _fuzzy_tax_rate(self, npcs, soft_controller):
        avg_mood = sum(float(n.behavior_vector.get("mood", 0.5)) for n in npcs) / len(npcs)
        avg_trust = sum(float(n.behavior_vector.get("trust", 0.5)) for n in npcs) / len(npcs)
        commerce = max(0.0, min(1.0, self.ledger["market_turnover"] / 12.0))

        social_stability = avg_mood * 0.55 + avg_trust * 0.45
        tension = 1.0 - social_stability

        # If fuzzy controller is steering guard/flee heavily, tax pressure should rise.
        guard_pressure = 0.0
        if soft_controller is not None:
            weights = soft_controller.get_weights()
            guard_pressure = max(0.0, min(1.0, 0.5 * float(weights.get("guard", 1.0) - 1.0) + 0.5 * float(weights.get("flee", 1.0) - 1.0)))

        rate = 0.04 + 0.13 * tension + 0.06 * (1.0 - commerce) + 0.05 * guard_pressure
        return max(0.03, min(0.26, rate))

    def _collect_tax(self, npcs, soft_controller, social_system=None):
        rate = self._fuzzy_tax_rate(npcs, soft_controller)
        collectors = [n for n in npcs if n.name in self.tax_collectors and n.npc_class == "Elite"]
        taxable = [n for n in npcs if not self._is_tax_exempt(n)]
        taxable = [n for n in taxable if n.name not in self.tax_collectors]
        if not taxable or not collectors:
            return

        treasury_take = 0.0
        gross_collected = 0.0
        corruption_skim = 0.0
        corrupt_cases = 0

        random.shuffle(taxable)
        for npc in taxable:
            acc = self.accounts[id(npc)]
            due = self._tax_due_for(npc, acc, rate)
            if due <= 0.02:
                continue

            # Use nearest collector to create believable local duty behavior.
            collector = min(
                collectors,
                key=lambda c: (c.x - npc.x) * (c.x - npc.x) + (c.y - npc.y) * (c.y - npc.y),
            )
            cmeta = self.tax_collectors.get(collector.name, {})
            corruption = float(cmeta.get("corruption", 0.1))
            strictness = float(cmeta.get("strictness", 0.6))

            due *= 0.9 + strictness * 0.25
            due = min(due, max(0.0, acc["coin"] - 0.08))
            if due <= 0.02:
                continue

            acc["coin"] -= due
            gross_collected += due

            skim = 0.0
            if due > 0.25 and random.random() < corruption * 0.62:
                skim_frac = random.uniform(0.08, 0.28) + corruption * 0.2
                skim = min(due * 0.7, due * skim_frac)
                corruption_skim += skim
                corrupt_cases += 1
                self.accounts[id(collector)]["coin"] += skim
                if random.random() < 0.35:
                    collector.show_speech("Tax duty complete. Move along.", 2.4)
                    npc.show_speech("That felt heavier than usual...", 2.4)
                if social_system is not None and random.random() < 0.4:
                    social_system.remember_event(
                        npc,
                        f"felt extorted by collector {collector.name}",
                        impact=-0.02,
                        trust_shift=-0.03,
                        negative=True,
                    )
            else:
                if random.random() < 0.28:
                    collector.show_speech("By decree, tax is due.", 2.2)

            treasury_take += max(0.0, due - skim)

        self.ledger["treasury"] += treasury_take
        self.ledger["last_tax_rate"] = rate
        self.ledger["tax_collected_last_round"] = gross_collected
        self.ledger["corruption_skim_last_round"] = corruption_skim
        self.ledger["corrupt_collections"] = corrupt_cases
        self.ledger["tax_collector_count"] = len(collectors)

        # Legitimate stipend for collectors on top of treasury routing.
        if collectors and treasury_take > 0.0:
            stipend = (treasury_take * 0.12) / len(collectors)
            for c in collectors:
                self.accounts[id(c)]["coin"] += stipend

        if gross_collected > 0.05:
            if corrupt_cases > 0:
                self._event(
                    f"Tax round: {gross_collected:.1f} coin at {int(rate * 100)}% | treasury {treasury_take:.1f}, skimmed {corruption_skim:.1f}"
                )
            else:
                self._event(f"Tax round: {gross_collected:.1f} coin at {int(rate * 100)}% | treasury {treasury_take:.1f}")

            lead = random.choice(collectors)
            if self.tax_collectors.get(lead.name, {}).get("corruption", 0.0) > 0.35:
                lead.show_speech("Taxes keep the peace. Pay promptly.", 2.8)
            else:
                lead.show_speech("Village tax collection complete.", 2.8)

    def get_stats_for_ui(self):
        return {
            "treasury": self.ledger["treasury"],
            "grain_output": self.ledger["grain_output"],
            "market_turnover": self.ledger["market_turnover"],
            "barter_count": self.ledger["barter_count"],
            "last_tax_rate": self.ledger["last_tax_rate"],
            "tax_collected_last_round": self.ledger["tax_collected_last_round"],
            "corruption_skim_last_round": self.ledger["corruption_skim_last_round"],
            "corrupt_collections": self.ledger["corrupt_collections"],
            "tax_collector_count": self.ledger["tax_collector_count"],
            "farm_count": len(self.farm_zones),
            "owned_farms": len(self.farm_owners),
        }

    def get_recent_events(self, limit=4):
        if limit <= 0:
            return []
        return self.recent_events[-limit:]

    def get_farm_owner_lines(self):
        lines = []
        for zone in self.farm_zones:
            owner = self.farm_owners.get(zone)
            if not owner:
                continue
            short_zone = zone.replace("wheat_farm_", "farm-")
            lines.append(f"{short_zone}: {owner['owner_name']} ({owner['owner_class']})")
        return lines

    def get_npc_economic_status(self, npc):
        """Return a compact economy-driven status label for one NPC."""
        if npc is None:
            return {"label": "unknown", "color": "neutral"}

        acc = self.accounts.get(id(npc), {})
        coin = float(acc.get("coin", 0.0))
        grain = float(acc.get("grain", 0.0))
        tax = float(self.ledger.get("last_tax_rate", 0.0))
        climate = self._economic_climate()

        stress = climate["stress"]
        prosperity = climate["prosperity"]
        personal_pressure = 0.45 * (1.0 if coin < 7.0 else 0.0) + 0.35 * (1.0 if grain < 1.1 else 0.0) + 0.2 * (tax / 0.26)

        if personal_pressure + stress > 1.05:
            return {"label": "tax-stressed", "color": "negative"}
        if coin < 5.0 and grain < 0.8:
            return {"label": "resource-starved", "color": "negative"}
        if prosperity > 0.6 and coin > 12.0:
            return {"label": "harvest-buoyant", "color": "positive"}
        if coin > 20.0:
            return {"label": "wealth-secure", "color": "positive"}
        if tax > 0.18:
            return {"label": "under-heavy-tax", "color": "warning"}
        return {"label": "economy-steady", "color": "neutral"}

    def get_climate_status(self):
        """Expose economy climate for other systems (night crime, social policy, etc.)."""
        return self._economic_climate()
