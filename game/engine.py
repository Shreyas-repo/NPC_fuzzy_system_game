"""
Game Engine — orchestrates all systems: world, entities, AI, UI.
Now running local soft-computing systems without external LLM dependency.
"""
import pygame
import sys
import math
import random
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TITLE,
    STATE_PLAYING, STATE_SPECTATOR, STATE_CHATTING, STATE_PAUSED,
    TILE_SIZE, WORLD_WIDTH, WORLD_HEIGHT, ZONES,
    AUTO_ENTER_DOOR_DISTANCE, AUTO_ENTER_COOLDOWN, COLORS
)
from utils.sprite_assets import SpriteAssets
from game.world import World
from game.camera import Camera
from game.day_cycle import DayCycle
from game.weather import WeatherSystem
from entities.player import Player
from entities.npc_types import create_all_npcs
from ai.behavior import BehaviorSystem
from ai.clustering import ClusteringEngine
from ai.sentiment import DialogueGenerator
from ai.conversation_learning import ConversationLearningModel
from ai.routine import RoutineEngine
from ai.social import SocialSystem
from ai.farming import FarmSystem
from ai.economy import EconomySystem
from ai.emotion_database import EmotionDatabase
from ai.vector_database import VectorDatabase
from ai.interaction_learning import InteractionLearningEngine
from ai.soft_controller import FuzzySocialController
from ai.policy_optimizer import EvolutionaryPolicyTuner
from ai.research_metrics import ResearchMetrics
from ui.chat_box import ChatBox
from ui.hud import HUD
from ui.spectator_ui import SpectatorUI
from config import (
    SOFT_COMPUTING_ENABLED,
    FUZZY_UPDATE_INTERVAL,
    FUZZY_MIN_CONFIDENCE,
    EVOLUTION_UPDATE_INTERVAL,
    METRICS_UPDATE_INTERVAL,
    ADAPTIVE_PROFILE_AUTO_ENABLED,
    ADAPTIVE_PROFILE_SWITCH_COOLDOWN,
    ADAPTIVE_PROFILE_AGGRESSIVE_STABILITY_MAX,
    ADAPTIVE_PROFILE_AGGRESSIVE_TRUST_MAX,
    ADAPTIVE_PROFILE_AGGRESSIVE_CONFLICT_MIN,
    ADAPTIVE_PROFILE_NORMAL_STABILITY_MIN,
    ADAPTIVE_PROFILE_NORMAL_TRUST_MIN,
    ADAPTIVE_PROFILE_NORMAL_CONFLICT_MAX,
)


class GameEngine:
    """Main game engine that runs the simulation."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = STATE_PLAYING

        # ─── Core Systems ─────────────────────────────────────
        self.world = World()
        self.camera = Camera()
        self.day_cycle = DayCycle()
        self.weather = WeatherSystem()

        # ─── Player ───────────────────────────────────────────
        # Spawn player at town square
        ts = ZONES["town_square"]
        player_x = (ts[0] + ts[2] // 2) * TILE_SIZE
        player_y = (ts[1] + ts[3]) * TILE_SIZE + TILE_SIZE * 2
        self.player = Player(player_x, player_y, self.world.tile_map)
        self.camera.set_target(self.player)
        self.camera.x = self.player.x
        self.camera.y = self.player.y

        # ─── NPCs ─────────────────────────────────────────────
        self.npcs = create_all_npcs(self.world.tile_map, self.world)
        for npc in self.npcs:
            npc.world_npcs = self.npcs

        # ─── AI Systems ───────────────────────────────────────
        self.behavior_system = BehaviorSystem()
        self.clustering_engine = ClusteringEngine()
        self.conversation_learner = ConversationLearningModel()
        self.dialogue_generator = DialogueGenerator(conversation_learner=self.conversation_learner)
        self.routine_engine = RoutineEngine()
        self.social_system = SocialSystem()
        self.social_system.set_conversation_logger(self.conversation_learner)
        self.farm_system = FarmSystem(self.world)
        self.economy_system = EconomySystem()

        # ─── Local Memory + Emotion Database ───────────────────
        self.ollama_client = None
        self.vector_database = VectorDatabase()
        self.ollama_dialogue = None
        self.emotion_database = EmotionDatabase(
            vector_database=self.vector_database,
            ollama_client=self.ollama_client,
        )
        self.interaction_learning = InteractionLearningEngine(
            ollama_client=self.ollama_client,
            vector_db=self.vector_database,
        )

        # ─── Soft Computing Research Stack ─────────────────────
        self.soft_computing_enabled = SOFT_COMPUTING_ENABLED
        self.soft_controller = FuzzySocialController()
        self.policy_tuner = EvolutionaryPolicyTuner()
        self.research_metrics = ResearchMetrics()
        self.fuzzy_timer = 0.0
        self.evolution_timer = 0.0
        self.metrics_timer = 0.0
        print("  ✅ Local fuzzy dialogue mode active")
        print("     No external LLM integration in runtime")
        print(f"  📊 Emotion Database: {len(self.emotion_database.records)} past interactions loaded")
        print()

        # Assign default routines
        self.routine_engine.assign_default_routines(self.npcs)
        self.social_system.bootstrap_from_npcs(self.npcs)

        # ─── UI ───────────────────────────────────────────────
        self.chat_box = ChatBox()
        self.chat_box.init_fonts()
        self.chat_box.set_ollama(
            self.ollama_dialogue,
            self.emotion_database,
            self.conversation_learner,
        )
        self.chat_box.set_npcs(self.npcs)
        self.hud = HUD()
        self.hud.init_fonts()
        self.spectator_ui = SpectatorUI()
        self.spectator_ui.init_fonts()

        # ─── Runtime Adaptation Profile ──────────────────────
        self.adaptive_profile_auto = bool(ADAPTIVE_PROFILE_AUTO_ENABLED)
        self.adaptive_profile_name = "normal"
        self.adaptive_profile_cooldown = 0.0
        self._apply_adaptive_profile("normal")

        # ─── Interaction prompt ───────────────────────────────
        self.nearby_npc = None
        self.nearby_building = None
        self.prompt_font = pygame.font.SysFont("Arial", 13, bold=True)

        # ─── Interior Mode ────────────────────────────────────
        self.interior_active = False
        self.interior_building_name = None
        self.interior_player_x = SCREEN_WIDTH // 2
        self.interior_player_y = SCREEN_HEIGHT - 130
        self.interior_player_direction = "down"
        self.interior_player_moving = False
        self.interior_anim_timer = 0.0
        self.enter_cooldown_timer = 0.0

        # ─── Pause menu ──────────────────────────────────────
        self.pause_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.pause_small = pygame.font.SysFont("Arial", 14)

        # ─── Routine observation timer ────────────────────────
        self.routine_observe_timer = 0.0

        # ─── Law / Jail / Trial System ────────────────────────
        self.player_wanted = False
        self.player_arrested = False
        self.guard_capture_active = False
        self.guard_capture_timer = 0.0
        self.guard_reroute_timer = 0.0
        self.current_charge = None
        self.trial_active = False
        self.trial_timer = 0.0
        self.trial_verdict = None
        self.trial_release_timer = 0.0
        self.legal_status_message = ""
        self.push_cooldown_timer = 0.0
        self.player_crime_points = 0.0
        self.trial_judge = "Queen Seraphina"
        self.trial_accuser = ""
        self.trial_defendant = "You"
        self.trial_phase_text = ""
        self.trial_accuser_statement = ""
        self.trial_defendant_statement = ""
        self.trial_stage = "none"
        self.trial_player_argument = None
        self.trial_player_argument_text = ""

        # ─── Raid / Sword Defense System ──────────────────────
        self.raid_active = False
        self.raid_timer = random.uniform(90.0, 150.0)
        self.raid_entities = []
        self.raid_total_spawned = 0
        self.raid_kills_by_player = 0
        self.raid_villager_harm = 0
        self.raid_loot_taken = 0.0
        self.raid_status_message = ""
        self.guard_formation_timer = 0.0
        self.sword_cooldown = 0.0
        self.sword_fx_timer = 0.0
        self.sword_fx_points = None
        self.treasure_chests = []
        self._init_treasure_chests()
        self.night_home_timer = 0.0
        self.night_guard_timer = 0.0
        self.night_theft_timer = 0.0
        self.night_romance_timer = 0.0

        # ─── Proactive NPC-to-Player Chat System ─────────────
        self.proactive_chat_timer = 0.0
        self.proactive_chat_interval = 5.0
        self.proactive_chat_cooldown = 0.0
        self.proactive_chat_cooldown_duration = 22.0
        self.proactive_chat_pending_npc = None
        self.proactive_chat_per_npc_cooldown = {}
        self.proactive_chat_reroute_timer = 0.0

    def run(self):
        """Main game loop."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)  # Cap delta time

            self._handle_events()
            self._update(dt)
            self._render()

        self.social_system.save_memory()
        pygame.quit()
        sys.exit()

    def _handle_events(self):
        """Process all pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # Chat box consumes events first when active
            if self.state == STATE_CHATTING:
                if self.chat_box.handle_event(event):
                    if not self.chat_box.active:
                        self.state = STATE_PLAYING
                    continue

            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self.state == STATE_SPECTATOR:
                        self.spectator_ui.handle_click(
                            event.pos[0], event.pos[1],
                            self.npcs, self.camera
                        )
                    elif self.state == STATE_PLAYING and not self.interior_active and not self.player_arrested:
                        self._player_sword_attack()
                elif event.button == 4:  # Scroll up
                    if self.state == STATE_SPECTATOR:
                        self.camera.handle_zoom(1)
                elif event.button == 5:  # Scroll down
                    if self.state == STATE_SPECTATOR:
                        self.camera.handle_zoom(-1)

    def _handle_keydown(self, event):
        """Handle key press events."""
        if event.key == pygame.K_ESCAPE:
            if self.state == STATE_PAUSED:
                self.state = STATE_PLAYING
            elif self.state == STATE_CHATTING:
                self.chat_box.close()
                self.state = STATE_PLAYING
            elif self.state == STATE_SPECTATOR:
                self.camera.toggle_spectator()
                self.spectator_ui.following_npc = None
                self.state = STATE_PLAYING
            else:
                self.state = STATE_PAUSED

        elif event.key == pygame.K_TAB:
            if self.state in (STATE_PLAYING, STATE_SPECTATOR):
                self.camera.toggle_spectator()
                if self.camera.spectator_mode:
                    self.state = STATE_SPECTATOR
                else:
                    self.spectator_ui.following_npc = None
                    self.state = STATE_PLAYING

        elif event.key == pygame.K_e:
            if self.state == STATE_PLAYING:
                if self.interior_active:
                    if self.player_arrested:
                        return
                    self._exit_interior()
                elif self.nearby_npc:
                    self.chat_box.open(self.nearby_npc, self.dialogue_generator)
                    self.state = STATE_CHATTING

        elif event.key == pygame.K_f:
            if self.state == STATE_PLAYING and not self.interior_active and not self.player_arrested:
                self._attempt_push_npc()

        elif event.key == pygame.K_t:
            if self.player_arrested and self.trial_verdict == "detained":
                self._plead_for_mercy()

        elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
            if self.player_arrested and self.trial_active and self.trial_stage == "defendant":
                self._submit_trial_argument(event.key)

        elif event.key == pygame.K_SPACE:
            if self.state == STATE_PLAYING and not self.interior_active and not self.player_arrested:
                self._player_sword_attack()

        elif event.key == pygame.K_r:
            if self.state == STATE_SPECTATOR and not self.player_arrested and not self.trial_active:
                if not self.raid_active:
                    self._spawn_raid()
                else:
                    self.raid_status_message = "Raid already active. Watch and coordinate defense."

        elif event.key == pygame.K_c:
            self.hud.toggle_chatter_pause(self.social_system)

        elif event.key == pygame.K_PAGEUP:
            self.hud.scroll_chatter(2)

        elif event.key == pygame.K_PAGEDOWN:
            self.hud.scroll_chatter(-2)

        elif event.key == pygame.K_LEFTBRACKET:
            self.hud.cycle_chatter_cluster_mode(-1, self.player, self.npcs, self.social_system)

        elif event.key == pygame.K_RIGHTBRACKET:
            self.hud.cycle_chatter_cluster_mode(1, self.player, self.npcs, self.social_system)

        # Spectator time speed controls
        elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
            if self.state == STATE_SPECTATOR:
                speeds = {pygame.K_1: 1.0, pygame.K_2: 2.0, pygame.K_3: 4.0,
                          pygame.K_4: 6.0, pygame.K_5: 10.0}
                self.day_cycle.set_speed(speeds.get(event.key, 1.0))

        elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
            if self.state == STATE_SPECTATOR:
                self.camera.handle_zoom(1)
        elif event.key == pygame.K_MINUS:
            if self.state == STATE_SPECTATOR:
                self.camera.handle_zoom(-1)

    def _update(self, dt):
        """Update all game systems."""
        if self.state == STATE_PAUSED:
            return

        self._process_crime_reports()
        self._process_social_incidents()

        if self.enter_cooldown_timer > 0.0:
            self.enter_cooldown_timer = max(0.0, self.enter_cooldown_timer - dt)
        if self.push_cooldown_timer > 0.0:
            self.push_cooldown_timer = max(0.0, self.push_cooldown_timer - dt)
        if self.sword_cooldown > 0.0:
            self.sword_cooldown = max(0.0, self.sword_cooldown - dt)
        if self.sword_fx_timer > 0.0:
            self.sword_fx_timer = max(0.0, self.sword_fx_timer - dt)

        if self.guard_capture_active:
            self._update_guard_capture(dt)

        if self.trial_active:
            self._update_trial(dt)

        self._update_raid_system(dt)

        # Day/night cycle
        self.day_cycle.update(dt)
        self.weather.update(dt)

        # Player movement (only in playing mode)
        if self.state == STATE_PLAYING:
            keys = pygame.key.get_pressed()
            if self.interior_active:
                self._update_interior_player(dt, keys)
                self.nearby_npc = None
                self.nearby_building = None
            else:
                if not self.guard_capture_active and not self.player_arrested:
                    self.player.update(dt, keys)
                self.nearby_npc = self.player.get_nearby_npc(self.npcs)
                self.nearby_building = self.world.get_nearby_enterable_building(
                    self.player.x,
                    self.player.y,
                )
                self._maybe_auto_enter_building()

        # Camera
        keys = pygame.key.get_pressed()
        self.camera.update(dt, keys)

        # NPC updates
        for npc in self.npcs:
            npc.update(dt, self.day_cycle)

        # AI Systems
        self.clustering_engine.update(dt, self.npcs, self.behavior_system)
        self.social_system.update(dt, self.npcs)
        self.economy_system.update(
            dt,
            self.npcs,
            self.soft_controller if self.soft_computing_enabled else None,
            self.social_system,
        )
        self._update_proactive_npc_conversations(dt)
        self._update_night_society(dt)
        self.farm_system.update(dt, self.npcs, self.economy_system)
        self.interaction_learning.update(dt, self.npcs)

        # Emotion database — apply learned behavior to NPCs
        self.emotion_database.apply_learned_behavior(self.npcs)

        # Routine observation and adaptation
        self.routine_observe_timer += dt
        if self.routine_observe_timer >= 10.0:
            self.routine_engine.observe_npcs(self.npcs, self.day_cycle.current_hour)
            self.routine_observe_timer = 0.0
        self.routine_engine.adapt_routines(self.npcs, dt)

        # Chat box update
        if self.state == STATE_CHATTING:
            self.chat_box.update(dt)

        # HUD update
        if not self.interior_active:
            self.hud.update(self.player, self.npcs, self.camera)

        if self.soft_computing_enabled:
            self._update_soft_computing(dt)

    def _render(self):
        """Render everything to screen."""
        self.screen.fill((0, 0, 0))

        if self.interior_active:
            self._render_interior(self.screen)
            if self.state == STATE_PAUSED:
                self._render_pause_menu()
            pygame.display.flip()
            return

        # Apply zoom in spectator mode
        if self.camera.spectator_mode and self.camera.zoom != 1.0:
            # Render to a larger surface, then scale
            render_surf = pygame.Surface(
                (self.camera.view_width, self.camera.view_height)
            )
            render_surf.fill((0, 0, 0))
            self._render_world(render_surf)
            scaled = pygame.transform.scale(render_surf, (SCREEN_WIDTH, SCREEN_HEIGHT))
            self.screen.blit(scaled, (0, 0))
        else:
            self._render_world(self.screen)

        # Day/night overlay
        self.day_cycle.render_overlay(self.screen)
        self.weather.render_overlay(self.screen, self.day_cycle)

        # UI layers
        if self.state == STATE_SPECTATOR:
            self.spectator_ui.render(
                self.screen, self.camera, self.day_cycle,
                self.npcs, self.clustering_engine
            )

        # HUD (always visible)
        self.hud.render(
            self.screen, self.day_cycle, self.player, self.npcs,
            self.camera,
            self.clustering_engine,
            self.social_system,
            self.economy_system,
            self.policy_tuner,
            self.research_metrics,
        )

        # Interaction prompt
        if self.state == STATE_PLAYING:
            if self.nearby_npc and not self.player_arrested:
                self._render_interaction_prompt()
            elif self.nearby_building:
                self._render_building_prompt()

        # Chat box
        if self.state == STATE_CHATTING:
            self.chat_box.render(self.screen)

        # Pause menu
        if self.state == STATE_PAUSED:
            self._render_pause_menu()

        self._render_law_overlay()

        pygame.display.flip()

    def _render_world(self, surface):
        """Render world, player, and NPCs."""
        # World (terrain, buildings, decorations)
        self.world.render(surface, self.camera, self.day_cycle, self.farm_system, self.npcs)

        # Treasure chests by district wealth tier.
        self._render_treasure_chests(surface)

        # Raiders
        self._render_raiders(surface)

        # NPCs
        for npc in self.npcs:
            npc.render(surface, self.camera)

        # Player (render on top)
        if not self.camera.spectator_mode:
            self.player.render(surface, self.camera)
            self._render_sword_fx(surface)

    def _render_interaction_prompt(self):
        """Render 'Press E to talk' prompt near NPC."""
        if not self.nearby_npc:
            return

        cam_rect = self.camera.get_visible_rect()
        sx = int(self.nearby_npc.x - cam_rect.x)
        sy = int(self.nearby_npc.y - cam_rect.y) - 45

        prompt_text = f"Press E to talk | F to push {self.nearby_npc.name}"
        prompt_surf = self.prompt_font.render(prompt_text, True, (255, 255, 255))

        bg_w = prompt_surf.get_width() + 16
        bg_h = prompt_surf.get_height() + 8
        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (30, 30, 50, 200), (0, 0, bg_w, bg_h), border_radius=4)
        pygame.draw.rect(bg, (80, 160, 255), (0, 0, bg_w, bg_h), 1, border_radius=4)
        bg.blit(prompt_surf, (8, 4))

        self.screen.blit(bg, (sx - bg_w // 2, sy))

    def _render_pause_menu(self):
        """Render pause overlay."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 150), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(overlay, (0, 0))

        title = self.pause_font.render("⏸ PAUSED", True, (255, 255, 255))
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 40))

        resume = self.pause_small.render("Press ESC to resume", True, (180, 180, 200))
        self.screen.blit(resume, (SCREEN_WIDTH // 2 - resume.get_width() // 2,
                                   SCREEN_HEIGHT // 2 + 10))

        # Emotion DB stats
        db_stats = self.emotion_database.get_stats_for_ui()
        economy_stats = self.economy_system.get_stats_for_ui()
        farm_stats = self.farm_system.get_stats_for_ui()
        llm_status = "🧠 Local Fuzzy Mode"

        info_lines = [
            "Village of Minds — NPC Simulation",
            f"NPCs: {len(self.npcs)} | Day: {self.day_cycle.day_count} | Time: {self.day_cycle.time_string}",
            f"Behavior Clusters: {len(set(self.clustering_engine.behavior_labels) - {-1})}",
            f"Social Gatherings: {self.clustering_engine.spatial_clustering.n_clusters}",
            f"Dialogue: {llm_status}",
            f"Interactions Recorded: {db_stats['total_interactions']} | Player Tendency: {db_stats['player_tendency']}",
            f"Emotion Clusters Discovered: {db_stats['n_clusters']}",
            f"Vector DB: {db_stats['vector_records']} records | KMeans: {db_stats['vector_kmeans']} | DBSCAN: {db_stats['vector_dbscan']}",
            f"Economy: Treasury {economy_stats['treasury']:.1f} | Grain {economy_stats['grain_output']:.1f} | Turnover {economy_stats['market_turnover']:.1f}",
            f"Barter Deals: {economy_stats['barter_count']} | Tax Rate: {economy_stats['last_tax_rate'] * 100:.0f}% | Farms Owned: {economy_stats['owned_farms']}/{economy_stats['farm_count']}",
            f"Farming: Plots {farm_stats['plots']} | Ripe {farm_stats['stage_counts']['ripe']} | Growing {farm_stats['stage_counts']['growing']} | Storage: {farm_stats['storage_zone']}",
        ]

        y = SCREEN_HEIGHT // 2 + 50
        for line in info_lines:
            surf = self.pause_small.render(line, True, (140, 140, 160))
            self.screen.blit(surf, (SCREEN_WIDTH // 2 - surf.get_width() // 2, y))
            y += 22

    def _render_building_prompt(self):
        if not self.nearby_building:
            return
        if self.player_arrested:
            return
        bname = self.nearby_building[0].replace("_", " ").title()
        prompt_text = f"Walk to door to enter {bname}"
        prompt_surf = self.prompt_font.render(prompt_text, True, (255, 255, 255))
        bg_w = prompt_surf.get_width() + 16
        bg_h = prompt_surf.get_height() + 8
        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (22, 34, 28, 220), (0, 0, bg_w, bg_h), border_radius=4)
        pygame.draw.rect(bg, (130, 188, 120), (0, 0, bg_w, bg_h), 1, border_radius=4)
        bg.blit(prompt_surf, (8, 4))
        self.screen.blit(bg, (SCREEN_WIDTH // 2 - bg_w // 2, SCREEN_HEIGHT - 78))

    def _enter_interior(self, building_name):
        self.interior_active = True
        self.interior_building_name = building_name
        self.interior_player_x = SCREEN_WIDTH // 2
        self.interior_player_y = SCREEN_HEIGHT - 132
        self.interior_player_direction = "up"
        self.interior_player_moving = False

    def _exit_interior(self):
        exit_building_name = self.interior_building_name
        self.interior_active = False
        self.interior_building_name = None
        self.enter_cooldown_timer = AUTO_ENTER_COOLDOWN

        # Reposition player just outside the door so auto-enter doesn't loop.
        if exit_building_name and exit_building_name in self.world.buildings:
            ex, ey = self.world.buildings[exit_building_name].get_entrance()
            self.player.x = float(ex)
            self.player.y = float(ey + TILE_SIZE * 0.9)
            self.player.direction = "down"

    def _update_interior_player(self, dt, keys):
        if self.player_arrested:
            self.interior_player_moving = False
            return

        speed = self.player.speed * 0.92
        dx = 0.0
        dy = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy = -1.0
            self.interior_player_direction = "up"
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy = 1.0
            self.interior_player_direction = "down"
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx = -1.0
            self.interior_player_direction = "left"
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx = 1.0
            self.interior_player_direction = "right"

        self.interior_player_moving = (dx != 0.0 or dy != 0.0)
        if self.interior_player_moving:
            length = math.sqrt(dx * dx + dy * dy)
            dx /= length
            dy /= length
            self.interior_player_x += dx * speed * dt
            self.interior_player_y += dy * speed * dt
            self.interior_anim_timer += dt

        room = pygame.Rect(170, 86, SCREEN_WIDTH - 340, SCREEN_HEIGHT - 172)
        self.interior_player_x = max(room.left + 24, min(room.right - 24, self.interior_player_x))
        self.interior_player_y = max(room.top + 40, min(room.bottom - 14, self.interior_player_y))

    def _render_interior(self, surface):
        if self.interior_building_name == "castle_jail":
            self._render_castle_jail_interior(surface)
            return

        room = pygame.Rect(170, 86, SCREEN_WIDTH - 340, SCREEN_HEIGHT - 172)
        wall_color = (90, 72, 58)
        floor_a = (156, 124, 86)
        floor_b = (146, 114, 78)
        trim_color = (58, 42, 30)

        surface.fill((0, 0, 0))
        pygame.draw.rect(surface, wall_color, room)
        pygame.draw.rect(surface, trim_color, room, 5)

        # Ceiling and wall trims add depth.
        pygame.draw.rect(surface, (112, 90, 72), (room.left + 8, room.top + 8, room.width - 16, 24), border_radius=5)
        pygame.draw.rect(surface, (78, 58, 44), (room.left + 8, room.top + 32, room.width - 16, 4))
        for bx in range(room.left + 26, room.right - 20, 88):
            pygame.draw.rect(surface, (74, 54, 40), (bx, room.top + 10, 12, 24), border_radius=2)

        # Floor tiles
        tile = 32
        inner = room.inflate(-24, -28)
        for y in range(inner.top, inner.bottom, tile):
            for x in range(inner.left, inner.right, tile):
                c = floor_a if ((x // tile + y // tile) % 2 == 0) else floor_b
                pygame.draw.rect(surface, c, (x, y, tile, tile))

        # Area rug in center.
        rug = pygame.Rect(room.centerx - 170, room.centery + 20, 340, 120)
        pygame.draw.rect(surface, (126, 38, 36), rug, border_radius=10)
        pygame.draw.rect(surface, (214, 188, 108), rug, 4, border_radius=10)
        for yy in range(rug.top + 10, rug.bottom - 8, 18):
            pygame.draw.line(surface, (170, 62, 58), (rug.left + 12, yy), (rug.right - 12, yy), 2)

        # Top wall windows show a small outside scene only through panes.
        for i in range(3):
            wx = room.left + 100 + i * 220
            wy = room.top + 16
            wrect = pygame.Rect(wx, wy, 92, 62)
            pygame.draw.rect(surface, (56, 44, 34), wrect.inflate(8, 8), border_radius=4)
            pygame.draw.rect(surface, (176, 214, 242), wrect, border_radius=3)
            pygame.draw.rect(surface, (108, 82, 62), wrect, 3, border_radius=3)
            pygame.draw.line(surface, (108, 82, 62), (wx + 46, wy), (wx + 46, wy + 62), 2)
            pygame.draw.line(surface, (108, 82, 62), (wx, wy + 31), (wx + 92, wy + 31), 2)
            # tiny outside hint
            pygame.draw.rect(surface, (142, 205, 120), (wx + 3, wy + 36, 86, 23))
            pygame.draw.circle(surface, (54, 136, 70), (wx + 22, wy + 44), 8)

        # Warm wall sconces.
        for lx in (room.left + 54, room.right - 54):
            pygame.draw.rect(surface, (84, 60, 44), (lx - 8, room.top + 86, 16, 22), border_radius=3)
            pygame.draw.circle(surface, (246, 214, 132), (lx, room.top + 86), 6)
            glow = pygame.Surface((70, 58), pygame.SRCALPHA)
            for r in range(26, 8, -4):
                alpha = int(56 * (r / 26))
                pygame.draw.circle(glow, (255, 210, 120, alpha), (35, 22), r)
            surface.blit(glow, (lx - 35, room.top + 62), special_flags=pygame.BLEND_RGBA_ADD)

        # Door to exit
        door = pygame.Rect(room.centerx - 30, room.bottom - 20, 60, 20)
        pygame.draw.rect(surface, (66, 46, 32), door)
        pygame.draw.rect(surface, (42, 30, 21), door, 2)

        # Furniture layout.
        table = pygame.Rect(room.left + 84, room.centery + 36, 150, 58)
        pygame.draw.rect(surface, (118, 86, 56), table, border_radius=4)
        pygame.draw.rect(surface, (82, 60, 40), table, 2, border_radius=4)
        # Table legs + table props.
        for lx, ly in (
            (table.left + 8, table.bottom - 6),
            (table.right - 14, table.bottom - 6),
            (table.left + 8, table.top + 4),
            (table.right - 14, table.top + 4),
        ):
            pygame.draw.rect(surface, (92, 66, 44), (lx, ly, 6, 18))
        pygame.draw.circle(surface, (214, 214, 206), (table.centerx - 28, table.centery), 8)
        pygame.draw.rect(surface, (138, 102, 66), (table.centerx + 8, table.centery - 8, 24, 12), border_radius=2)

        # Stools near table.
        for sx, sy in ((table.left - 24, table.centery + 16), (table.right + 6, table.centery + 16)):
            pygame.draw.rect(surface, (106, 76, 50), (sx, sy, 18, 12), border_radius=3)
            pygame.draw.rect(surface, (78, 56, 38), (sx + 2, sy + 10, 3, 10))
            pygame.draw.rect(surface, (78, 56, 38), (sx + 13, sy + 10, 3, 10))

        bed = pygame.Rect(room.right - 248, room.centery + 20, 172, 82)
        pygame.draw.rect(surface, (170, 56, 62), bed, border_radius=5)
        pygame.draw.rect(surface, (224, 214, 206), (bed.x + 12, bed.y + 12, 60, 28), border_radius=4)
        pygame.draw.rect(surface, (98, 30, 36), bed, 2, border_radius=5)
        pygame.draw.rect(surface, (88, 62, 42), (bed.left - 8, bed.top + 4, 8, bed.height - 8), border_radius=2)

        # Bookshelf along top-right wall.
        shelf = pygame.Rect(room.right - 190, room.top + 102, 132, 94)
        pygame.draw.rect(surface, (96, 68, 46), shelf, border_radius=4)
        pygame.draw.rect(surface, (64, 46, 32), shelf, 2, border_radius=4)
        for sy in (shelf.top + 24, shelf.top + 50, shelf.top + 76):
            pygame.draw.line(surface, (58, 42, 30), (shelf.left + 6, sy), (shelf.right - 6, sy), 2)
        for i in range(9):
            bx = shelf.left + 10 + i * 12
            by = shelf.top + 8 + (i % 3) * 26
            pygame.draw.rect(surface, ((120 + (i * 14)) % 220, 70 + (i * 9) % 120, 60 + (i * 17) % 120), (bx, by, 8, 14), border_radius=1)

        # Kitchen counter + clay pots on left wall.
        counter = pygame.Rect(room.left + 52, room.top + 122, 160, 56)
        pygame.draw.rect(surface, (122, 90, 58), counter, border_radius=4)
        pygame.draw.rect(surface, (78, 58, 40), counter, 2, border_radius=4)
        pygame.draw.rect(surface, (140, 108, 74), (counter.left + 8, counter.top + 8, counter.width - 16, 14), border_radius=3)
        for px in (counter.left + 26, counter.left + 62, counter.left + 104):
            pygame.draw.circle(surface, (152, 96, 62), (px, counter.top + 30), 8)
            pygame.draw.rect(surface, (136, 86, 56), (px - 6, counter.top + 30, 12, 14), border_radius=3)

        # Treasure chest near bottom-right.
        chest = pygame.Rect(room.right - 150, room.bottom - 96, 88, 52)
        pygame.draw.ellipse(surface, (0, 0, 0, 70), (chest.left + 6, chest.bottom - 6, chest.width - 12, 10))
        pygame.draw.rect(surface, (116, 78, 38), chest, border_radius=7)
        lid = pygame.Rect(chest.left, chest.top - 8, chest.width, 26)
        pygame.draw.rect(surface, (132, 88, 44), lid, border_radius=9)
        pygame.draw.rect(surface, (86, 54, 28), chest, 2, border_radius=7)
        pygame.draw.rect(surface, (86, 54, 28), lid, 2, border_radius=9)
        for band_x in (chest.left + 18, chest.centerx - 4, chest.right - 26):
            pygame.draw.rect(surface, (168, 140, 72), (band_x, chest.top - 8, 8, chest.height + 8), border_radius=2)
        lock = pygame.Rect(chest.centerx - 8, chest.centery + 2, 16, 14)
        pygame.draw.rect(surface, (218, 188, 82), lock, border_radius=3)
        pygame.draw.rect(surface, (124, 96, 36), lock, 2, border_radius=3)
        pygame.draw.circle(surface, (102, 76, 24), (lock.centerx, lock.centery + 2), 2)

        # Small potted plants.
        for px, py in ((room.left + 286, room.top + 112), (room.right - 292, room.top + 112)):
            pygame.draw.rect(surface, (136, 88, 58), (px, py + 18, 22, 18), border_radius=3)
            pygame.draw.circle(surface, (54, 134, 70), (px + 11, py + 12), 11)
            pygame.draw.circle(surface, (62, 150, 84), (px + 5, py + 14), 8)
            pygame.draw.circle(surface, (62, 150, 84), (px + 17, py + 14), 8)

        # Render larger player sprite inside.
        sprite = SpriteAssets.get().get_character(
            "player",
            self.interior_player_direction,
            self.interior_player_moving,
            self.interior_anim_timer,
            tile_size=44,
        )
        if sprite is not None:
            px = int(self.interior_player_x)
            py = int(self.interior_player_y)
            pygame.draw.ellipse(surface, (0, 0, 0, 75), (px - 10, py + 4, 20, 6))
            surface.blit(sprite, (px - sprite.get_width() // 2, py - sprite.get_height() + 8))

        name = self.interior_building_name.replace("_", " ").title() if self.interior_building_name else "House"
        title = self.prompt_font.render(f"{name} Interior", True, (238, 232, 212))
        hint = self.prompt_font.render("Press E to exit", True, (210, 198, 168))
        surface.blit(title, (room.left + 12, room.top - 26))
        surface.blit(hint, (room.right - hint.get_width() - 12, room.top - 26))

    def _maybe_auto_enter_building(self):
        """Automatically enter a building when crossing its doorway."""
        if self.interior_active or self.state != STATE_PLAYING:
            return
        if self.enter_cooldown_timer > 0.0:
            return
        if not self.nearby_building:
            return

        bname, building = self.nearby_building
        ex, ey = building.get_entrance()
        dx = self.player.x - ex
        dy = self.player.y - ey
        dist_sq = dx * dx + dy * dy
        if dist_sq <= (AUTO_ENTER_DOOR_DISTANCE * AUTO_ENTER_DOOR_DISTANCE) and self.player.direction == "up":
            self._enter_interior(bname)
            self.enter_cooldown_timer = AUTO_ENTER_COOLDOWN

    def _update_soft_computing(self, dt):
        self.fuzzy_timer += dt
        self.evolution_timer += dt
        self.metrics_timer += dt
        self.adaptive_profile_cooldown = max(0.0, self.adaptive_profile_cooldown - dt)

        if self.fuzzy_timer >= FUZZY_UPDATE_INTERVAL:
            self.fuzzy_timer = 0.0
            for npc in self.npcs:
                if npc.state in ("talking",):
                    continue
                prev_action = getattr(npc, "soft_last_action", None)
                suggestion = self.soft_controller.recommend(npc, self.npcs, prev_action=prev_action)
                if suggestion.get("confidence", 0.0) >= FUZZY_MIN_CONFIDENCE:
                    npc.soft_action_hint = {
                        "action": suggestion["action"],
                        "zone": suggestion.get("zone"),
                    }
                    npc.soft_last_action = suggestion["action"]

        if self.metrics_timer >= METRICS_UPDATE_INTERVAL:
            self.metrics_timer = 0.0
            self.research_metrics.update(
                self.npcs,
                chat_box=self.chat_box,
                weights=self.soft_controller.get_weights(),
            )
            self._maybe_auto_switch_adaptive_profile()

        if self.evolution_timer >= EVOLUTION_UPDATE_INTERVAL:
            self.evolution_timer = 0.0
            self.policy_tuner.step(self.research_metrics.last_summary, self.soft_controller)

    def _apply_adaptive_profile(self, profile_name):
        self.adaptive_profile_name = str(profile_name)
        self.conversation_learner.set_adaptation_profile(profile_name)
        self.social_system.set_adaptation_profile(profile_name)
        self.hud.set_adaptation_profile(profile_name)

    def _maybe_auto_switch_adaptive_profile(self):
        if not self.adaptive_profile_auto:
            return
        if self.adaptive_profile_cooldown > 0.0:
            return

        metrics = getattr(self.research_metrics, "last_summary", {}) or {}
        stability = float(metrics.get("social_stability", 0.0))
        trust = float(metrics.get("avg_trust", 0.0))
        conflict = float(metrics.get("conflict_rate", 0.0))

        to_aggressive = (
            stability <= float(ADAPTIVE_PROFILE_AGGRESSIVE_STABILITY_MAX)
            or trust <= float(ADAPTIVE_PROFILE_AGGRESSIVE_TRUST_MAX)
            or conflict >= float(ADAPTIVE_PROFILE_AGGRESSIVE_CONFLICT_MIN)
        )
        to_normal = (
            stability >= float(ADAPTIVE_PROFILE_NORMAL_STABILITY_MIN)
            and trust >= float(ADAPTIVE_PROFILE_NORMAL_TRUST_MIN)
            and conflict <= float(ADAPTIVE_PROFILE_NORMAL_CONFLICT_MAX)
        )

        if self.adaptive_profile_name != "aggressive" and to_aggressive:
            self._apply_adaptive_profile("aggressive")
            self.adaptive_profile_cooldown = float(ADAPTIVE_PROFILE_SWITCH_COOLDOWN)
            return

        if self.adaptive_profile_name != "normal" and to_normal:
            self._apply_adaptive_profile("normal")
            self.adaptive_profile_cooldown = float(ADAPTIVE_PROFILE_SWITCH_COOLDOWN)

    def _process_crime_reports(self):
        report = self.chat_box.consume_crime_report()
        if not report:
            return
        if self.player_arrested:
            return
        legal_risk = self._fuzzy_legal_risk(report)
        if legal_risk >= 0.42:
            self._start_guard_capture(report)
        else:
            self.legal_status_message = "Warning recorded by village watch"
            self.player_crime_points = min(3.0, self.player_crime_points + 0.12)

    def _process_social_incidents(self):
        while True:
            incident = self.chat_box.consume_social_incident()
            if not incident:
                break

            source_name = incident.get("npc_name")
            source_npc = next((n for n in self.npcs if n.name == source_name), None)
            if source_npc is None:
                continue

            self.social_system.report_player_social_incident(
                self.npcs,
                source_npc,
                incident.get("type", "silent_prompt"),
                incident.get("style", "neutral"),
                float(incident.get("severity", 0.1)),
                incident.get("text", "The conversation ended awkwardly."),
            )

    def _fuzzy_membership_low(self, x):
        x = max(0.0, min(1.0, float(x)))
        if x <= 0.3:
            return 1.0
        if x >= 0.6:
            return 0.0
        return (0.6 - x) / 0.3

    def _fuzzy_membership_high(self, x):
        x = max(0.0, min(1.0, float(x)))
        if x <= 0.4:
            return 0.0
        if x >= 0.8:
            return 1.0
        return (x - 0.4) / 0.4

    def _fuzzy_legal_risk(self, report):
        """Fuzzy legal risk used for crime escalation, guard response, and trial decisions."""
        severity = float(report.get("severity", 0.5))
        intent = report.get("intent_type", "suspicious")
        avg_trust = sum(n.behavior_vector.get("trust", 0.5) for n in self.npcs) / max(1, len(self.npcs))
        distrust = 1.0 - avg_trust
        repeat_offender = min(1.0, self.player_crime_points / 3.0)

        violent_like = 1.0 if intent in ("threat_violence", "threat_robbery", "push_assault") else 0.45
        sev_high = self._fuzzy_membership_high(severity)
        distrust_high = self._fuzzy_membership_high(distrust)
        repeat_high = self._fuzzy_membership_high(repeat_offender)
        sev_low = self._fuzzy_membership_low(severity)

        # Rule blend
        risk = (
            0.42 * max(sev_high, violent_like * 0.8)
            + 0.28 * distrust_high
            + 0.22 * repeat_high
            + 0.08 * (1.0 - sev_low)
        )
        return max(0.0, min(1.0, risk))

    def _attempt_push_npc(self):
        if self.push_cooldown_timer > 0.0:
            return
        npc = self.player.get_nearby_npc(self.npcs)
        if not npc:
            return

        dx = npc.x - self.player.x
        dy = npc.y - self.player.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= 0.001:
            dir_map = {
                "up": (0.0, -1.0),
                "down": (0.0, 1.0),
                "left": (-1.0, 0.0),
                "right": (1.0, 0.0),
            }
            dx, dy = dir_map.get(self.player.direction, (0.0, 1.0))
            dist = 1.0

        nx, ny = dx / dist, dy / dist
        push_distance = 26.0
        target_x = npc.x + nx * push_distance
        target_y = npc.y + ny * push_distance
        if npc.tile_map.is_walkable_pixel(target_x, target_y, radius=12):
            npc.x = target_x
            npc.y = target_y

        npc.path = []
        npc.path_index = 0
        npc.state = "idle"
        npc.state_timer = 0.0

        self.push_cooldown_timer = 0.6
        self._handle_push_reaction(npc)

    def _handle_push_reaction(self, npc):
        cls = npc.npc_class
        self.player_crime_points = min(3.0, self.player_crime_points + 0.35)

        if cls in ("Peasant", "Labourer"):
            npc.show_emotion(":(", 2.2)
            npc.show_speech("Hey! Don't push me!", 2.5)
            npc.behavior_vector["trust"] = max(0.0, npc.behavior_vector.get("trust", 0.5) - 0.08)
            self.legal_status_message = f"{npc.name} is annoyed by your shove"
            return

        if cls in ("Elite", "Noble", "Royal"):
            npc.show_emotion(":|", 2.5)
            npc.show_speech("Guards! Assault on a citizen!", 2.8)
            report = {
                "npc_name": npc.name,
                "npc_class": cls,
                "intent_type": "push_assault",
                "severity": 0.78 if cls != "Royal" else 0.9,
                "player_message": "[physical shove]",
            }
            if self._fuzzy_legal_risk(report) >= 0.38:
                self._start_guard_capture(report)
            else:
                self.legal_status_message = "Royal warning issued for assault"
            return

        npc.show_emotion(":/", 2.0)
        npc.show_speech("Watch where you're going.", 2.2)
        npc.behavior_vector["trust"] = max(0.0, npc.behavior_vector.get("trust", 0.5) - 0.05)

    def _guard_npcs(self):
        return [n for n in self.npcs if n.npc_class == "Elite"]

    def _npc_in_zone(self, npc, zone_name):
        if zone_name not in ZONES:
            return False
        tx = int(npc.x // TILE_SIZE)
        ty = int(npc.y // TILE_SIZE)
        zx, zy, zw, zh = ZONES[zone_name]
        return zx <= tx < zx + zw and zy <= ty < zy + zh

    def _enforce_night_sleep_and_patrol(self):
        # Guards patrol the city at night.
        for guard in self._guard_npcs():
            patrol_zones = ["castle", "town_square", "trader_house_1", "trader_house_3", "noble_house_w2", "peasant_house_2"]
            zone = random.choice(patrol_zones)
            guard._go_to_zone(zone)
            guard.state = "walking"
            if random.random() < 0.22:
                guard.show_speech("Night watch. Keep moving.", 2.0)

        # Everyone else goes home to sleep.
        for npc in self.npcs:
            if npc.npc_class == "Elite" or npc.state == "talking":
                continue
            home_zone = npc._get_home_zone() if hasattr(npc, "_get_home_zone") else "town_square"
            if self._npc_in_zone(npc, home_zone):
                npc.state = "sleeping"
                npc.state_timer = 0.0
            elif npc.state not in ("walking", "sleeping"):
                npc._go_to_zone(home_zone)
                npc.state = "walking"

    def _night_theft_event(self):
        climate = self.economy_system.get_climate_status()
        if climate.get("stress", 0.0) < 0.62:
            return

        offenders = [
            n for n in self.npcs
            if n.npc_class in ("Peasant", "Labourer")
            and self.economy_system.accounts.get(id(n), {}).get("coin", 0.0) < 9.0
            and n.state != "talking"
        ]
        if not offenders:
            return

        victims = [
            n for n in self.npcs
            if n.npc_class in ("Peasant", "Noble")
            and self.economy_system.accounts.get(id(n), {}).get("coin", 0.0) > 8.0
        ]
        if not victims:
            return

        offender = random.choice(offenders)
        victim = random.choice(victims)
        if offender is victim:
            return

        off_acc = self.economy_system.accounts.get(id(offender))
        vic_acc = self.economy_system.accounts.get(id(victim))
        if not off_acc or not vic_acc:
            return

        amount = min(vic_acc.get("coin", 0.0) * 0.22, random.uniform(1.0, 4.0))
        if amount <= 0.2:
            return

        vic_acc["coin"] = max(0.0, vic_acc.get("coin", 0.0) - amount)
        off_acc["coin"] = off_acc.get("coin", 0.0) + amount
        offender.show_speech("I had no choice tonight...", 2.2)
        victim.show_speech("Someone stole from me!", 2.2)
        offender.show_emotion("😓", 2.0)
        victim.show_emotion("😠", 2.0)

        self.economy_system._event(f"Night theft: {offender.name} stole {amount:.1f} coin from {victim.name}")
        self.social_system.remember_event(
            offender,
            f"Stole coin from {victim.name} during economic hardship",
            trust_shift=-0.04,
            mood_shift=-0.015,
            negative=True,
        )
        self.social_system.remember_event(
            victim,
            f"Was robbed at night by {offender.name}",
            trust_shift=-0.05,
            mood_shift=-0.03,
            negative=True,
        )

    def _night_lovers_event(self):
        # Lovers may meet at night; guard response depends on class combination.
        poor_classes = {"Peasant", "Labourer"}
        upper_classes = {"Noble", "Elite", "Royal"}
        name_to_npc = {n.name: n for n in self.npcs}
        candidates = []
        for pair_key, affinity in self.social_system.relationships.items():
            if affinity < 0.72:
                continue
            if "|" not in pair_key:
                continue
            a_name, b_name = pair_key.split("|", 1)
            a = name_to_npc.get(a_name)
            b = name_to_npc.get(b_name)
            if not a or not b:
                continue
            if a.state == "talking" or b.state == "talking":
                continue
            candidates.append((a, b, affinity))

        if not candidates:
            return

        a, b, _ = random.choice(candidates)
        a.show_speech("Meet me quietly tonight, my heart.", 2.8)
        b.show_speech("I came. Just for a moment.", 2.8)
        a.show_emotion("❤️", 2.2)
        b.show_emotion("❤️", 2.2)
        a._go_to_zone("town_square")
        b._go_to_zone("town_square")
        a.state = "walking"
        b.state = "walking"

        guards = self._guard_npcs()
        seen = False
        guard_actor = None
        both_poor_same_class = (a.npc_class in poor_classes and b.npc_class in poor_classes and a.npc_class == b.npc_class)
        both_upper_class = (a.npc_class in upper_classes and b.npc_class in upper_classes)

        detect_chance = 0.62
        if both_poor_same_class:
            detect_chance = 0.78
        elif both_upper_class:
            detect_chance = 0.3

        if guards:
            for guard in guards:
                d1 = (guard.x - a.x) ** 2 + (guard.y - a.y) ** 2
                d2 = (guard.x - b.x) ** 2 + (guard.y - b.y) ** 2
                if d1 <= (170 * 170) or d2 <= (170 * 170):
                    if random.random() < detect_chance:
                        seen = True
                        guard_actor = guard
                        break

        if seen:
            if both_poor_same_class:
                if guard_actor:
                    guard_actor.show_speech("You two! Go home and sleep, now!", 2.5)
                for lover in (a, b):
                    home_zone = lover._get_home_zone() if hasattr(lover, "_get_home_zone") else "town_square"
                    lover._go_to_zone(home_zone)
                    lover.state = "walking"
                    lover.show_emotion("😳", 2.2)
                    self.social_system.remember_event(
                        lover,
                        f"Was scolded by night guard and sent home with {b.name if lover is a else a.name}",
                        trust_shift=-0.01,
                        mood_shift=-0.02,
                        negative=True,
                    )
                self.economy_system._event(f"Night curfew warning: guards sent {a.name} and {b.name} home")
                return

            if both_upper_class:
                # Upper-class couples are treated politely, and may even be ignored.
                if random.random() < 0.5:
                    if guard_actor:
                        guard_actor.show_speech("Good evening, my lords. Please return before curfew.", 2.4)
                    for lover in (a, b):
                        lover.show_emotion("🙂", 2.0)
                        self.social_system.remember_event(
                            lover,
                            f"Received polite curfew request from guard while meeting {b.name if lover is a else a.name}",
                            trust_shift=0.01,
                            mood_shift=0.01,
                        )
                    self.economy_system._event(f"Night courtesy: guards politely warned {a.name} and {b.name}")
                else:
                    for lover in (a, b):
                        self.social_system.remember_event(
                            lover,
                            f"Guard ignored their upper-class night meeting with {b.name if lover is a else a.name}",
                            trust_shift=0.01,
                            mood_shift=0.02,
                        )
                    self.economy_system._event(f"Night privilege: guards ignored {a.name} and {b.name}")
                return

            # Mixed/cross-class couples remain punishable.
            if guard_actor:
                guard_actor.show_speech("By order of the watch, you are under arrest!", 2.5)
            for lover in (a, b):
                lover._go_to_zone("castle")
                lover.state = "walking"
                lover.show_emotion("😨", 2.4)
                self.social_system.remember_event(
                    lover,
                    f"Was arrested by guards for forbidden cross-class romance with {b.name if lover is a else a.name}",
                    trust_shift=-0.05,
                    mood_shift=-0.07,
                    negative=True,
                )
            self.economy_system._event(f"Night arrest: guards detained {a.name} and {b.name} for illicit romance")
        else:
            for lover in (a, b):
                self.social_system.remember_event(
                    lover,
                    f"Shared a secret night meeting with {b.name if lover is a else a.name}",
                    trust_shift=0.01,
                    mood_shift=0.04,
                )

    def _update_night_society(self, dt):
        if self.player_arrested or self.trial_active:
            return
        if not self.day_cycle.is_sleep_hours():
            self.night_home_timer = 0.0
            self.night_guard_timer = 0.0
            self.night_theft_timer = 0.0
            self.night_romance_timer = 0.0
            return

        self.night_home_timer += dt
        self.night_guard_timer += dt
        self.night_theft_timer += dt
        self.night_romance_timer += dt

        if self.night_home_timer >= 3.0 or self.night_guard_timer >= 4.0:
            self.night_home_timer = 0.0
            self.night_guard_timer = 0.0
            self._enforce_night_sleep_and_patrol()

        if self.night_theft_timer >= 8.0:
            self.night_theft_timer = 0.0
            if random.random() < 0.6:
                self._night_theft_event()

        if self.night_romance_timer >= 10.0:
            self.night_romance_timer = 0.0
            if random.random() < 0.45:
                self._night_lovers_event()

    def _proactive_memory_signal(self, npc):
        profile = self.social_system.npc_profiles.get(npc.name, {})
        life_events = profile.get("life_events", [])
        if not life_events:
            return 0.0
        recent = " ".join(life_events[-4:]).lower()
        signal = 0.0
        if any(k in recent for k in ("raid", "stole", "robbed", "arrest", "guard")):
            signal += 0.2
        if any(k in recent for k in ("economy", "tax", "hardship", "shortage", "hunger")):
            signal += 0.16
        if any(k in recent for k in ("meeting", "love", "romance", "friend")):
            signal += 0.1
        return min(0.35, signal)

    def _proactive_chat_score(self, npc):
        if npc.state in ("talking", "sleeping"):
            return 0.0
        if npc.npc_class == "Elite" and self.guard_capture_active:
            return 0.0

        dx = npc.x - self.player.x
        dy = npc.y - self.player.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 320:
            return 0.0

        dist_factor = max(0.0, 1.0 - dist / 320.0)
        sociability = float(npc.personality.get("sociability", 0.4))
        trust = float(npc.behavior_vector.get("trust", 0.5))
        mood = float(npc.behavior_vector.get("mood", 0.5))
        memory_signal = self._proactive_memory_signal(npc)
        interaction_factor = min(0.2, float(getattr(npc, "interaction_count", 0)) / 25.0)

        urgency = 0.0
        if trust < 0.28:
            urgency += 0.12  # complain/confront
        if mood < 0.32:
            urgency += 0.1
        if mood > 0.72 and trust > 0.6:
            urgency += 0.08  # positive approach

        score = 0.08 + dist_factor * 0.28 + sociability * 0.22 + memory_signal + interaction_factor + urgency
        return max(0.0, min(1.0, score))

    def _build_proactive_opening(self, npc):
        profile = self.social_system.npc_profiles.get(npc.name, {})
        opener = profile.get("speech_opener", "listen")
        events = profile.get("life_events", [])
        hardships = profile.get("hardships", ["hard times"])
        desires = profile.get("desires", ["better days"])
        private_note = profile.get("private_memory", "")

        event = events[-1] if events else None
        hardship = random.choice(hardships)
        desire = random.choice(desires)

        if event:
            return f"{opener}, I wanted to tell you this: {event}."
        if private_note and random.random() < 0.45:
            return f"{opener}, I never say this out loud, but I {private_note}."
        if npc.behavior_vector.get("mood", 0.5) < 0.35:
            return f"{opener}, things feel rough lately. I'm carrying {hardship}."
        if npc.behavior_vector.get("trust", 0.5) > 0.62:
            return f"{opener}, I trust you enough to say this: I want {desire}."
        return f"{opener}, can we talk for a moment?"

    def _update_proactive_npc_conversations(self, dt):
        if self.state != STATE_PLAYING:
            return
        if self.interior_active or self.player_arrested or self.trial_active or self.guard_capture_active:
            return
        if self.state == STATE_CHATTING:
            return

        self.proactive_chat_cooldown = max(0.0, self.proactive_chat_cooldown - dt)
        self.proactive_chat_timer += dt
        self.proactive_chat_reroute_timer += dt

        expired = []
        for name, t in self.proactive_chat_per_npc_cooldown.items():
            nt = max(0.0, t - dt)
            self.proactive_chat_per_npc_cooldown[name] = nt
            if nt <= 0.0:
                expired.append(name)
        for name in expired:
            self.proactive_chat_per_npc_cooldown.pop(name, None)

        pending = self.proactive_chat_pending_npc
        if pending is not None:
            pdx = pending.x - self.player.x
            pdy = pending.y - self.player.y
            pdist = math.sqrt(pdx * pdx + pdy * pdy)

            if self.proactive_chat_reroute_timer >= 1.1:
                self.proactive_chat_reroute_timer = 0.0
                pending._navigate_to(self.player.x, self.player.y)
                pending.state = "walking"

            if pdist <= 56:
                self.proactive_chat_pending_npc = None
                self.proactive_chat_cooldown = self.proactive_chat_cooldown_duration
                self.proactive_chat_per_npc_cooldown[pending.name] = 90.0

                self.chat_box.open(pending, self.dialogue_generator)
                self.state = STATE_CHATTING
                opening = self._build_proactive_opening(pending)
                self.chat_box.history.append((pending.name, opening, COLORS["ui_chat_npc"]))
                pending.dialogue_history.append({
                    "player": "[npc initiated conversation]",
                    "npc": opening,
                    "sentiment": 0.0,
                })
            return

        if self.proactive_chat_cooldown > 0.0:
            return
        if self.proactive_chat_timer < self.proactive_chat_interval:
            return
        self.proactive_chat_timer = 0.0

        candidates = []
        weights = []
        for npc in self.npcs:
            if self.proactive_chat_per_npc_cooldown.get(npc.name, 0.0) > 0.0:
                continue
            score = self._proactive_chat_score(npc)
            if score < 0.5:
                continue
            candidates.append(npc)
            weights.append(score)

        if not candidates:
            return

        chosen = random.choices(candidates, weights=weights, k=1)[0]
        self.proactive_chat_pending_npc = chosen
        self.proactive_chat_reroute_timer = 1.1
        chosen.show_speech("Wait, I need to speak with you.", 2.2)

    def _start_guard_capture(self, report):
        self.player_wanted = True
        self.guard_capture_active = True
        self.guard_capture_timer = 0.0
        self.guard_reroute_timer = 0.0
        self.current_charge = report
        self.legal_status_message = "Guards alerted: suspicious statement recorded"

        if self.state == STATE_CHATTING:
            self.chat_box.close()
            self.state = STATE_PLAYING

        for guard in self._guard_npcs():
            guard.show_speech("Stop! You are under investigation!", 2.5)
            guard._navigate_to(self.player.x, self.player.y)
            guard.state = "walking"

    def _update_guard_capture(self, dt):
        if self.player_arrested:
            self.guard_capture_active = False
            return

        self.guard_capture_timer += dt
        self.guard_reroute_timer += dt

        if self.guard_reroute_timer >= 0.8:
            self.guard_reroute_timer = 0.0
            for guard in self._guard_npcs():
                guard._navigate_to(self.player.x, self.player.y)
                guard.state = "walking"

        for guard in self._guard_npcs():
            dx = guard.x - self.player.x
            dy = guard.y - self.player.y
            if (dx * dx + dy * dy) <= (42 * 42):
                self._capture_player()
                return

        if self.guard_capture_timer >= 10.0:
            self._capture_player()

    def _capture_player(self):
        self.guard_capture_active = False
        self.player_wanted = False
        self.player_arrested = True
        self.trial_active = True
        self.trial_timer = 0.0
        self.trial_verdict = None
        self.trial_release_timer = 0.0
        self.legal_status_message = "Captured by guards. Transported to castle jail."
        self.trial_accuser = (self.current_charge or {}).get("npc_name", "Captain Gareth")
        self._build_trial_statements()
        self.trial_stage = "jail"
        self.trial_player_argument = None
        self.trial_player_argument_text = ""

        self.interior_active = True
        self.interior_building_name = "castle_jail"
        self.interior_player_x = SCREEN_WIDTH // 2
        self.interior_player_y = SCREEN_HEIGHT // 2 + 70
        self.interior_player_direction = "up"

    def _build_trial_statements(self):
        intent = (self.current_charge or {}).get("intent_type", "suspicious")
        sev = float((self.current_charge or {}).get("severity", 0.5))

        charge_line = {
            "threat_violence": "The defendant issued violent threats.",
            "threat_robbery": "The defendant declared robbery intent.",
            "push_assault": "The defendant physically shoved a citizen.",
            "suspicious": "The defendant made suspicious criminal statements.",
        }
        self.trial_accuser_statement = charge_line.get(intent, "The defendant endangered public order.")

        if sev >= 0.85:
            self.trial_defendant_statement = "I acted in anger. I ask for mercy from the Crown."
        elif sev >= 0.65:
            self.trial_defendant_statement = "I made a mistake and regret my actions."
        else:
            self.trial_defendant_statement = "I meant no harm and ask the court for leniency."

    def _compute_trial_verdict(self):
        severity = float((self.current_charge or {}).get("severity", 0.5))
        intent_type = (self.current_charge or {}).get("intent_type", "suspicious")
        avg_trust = sum(n.behavior_vector.get("trust", 0.5) for n in self.npcs) / max(1, len(self.npcs))
        effective_severity = min(1.0, severity + 0.12 * self.player_crime_points)

        if self.trial_player_argument == "remorse":
            effective_severity = max(0.0, effective_severity - 0.12)
        elif self.trial_player_argument == "deflect":
            effective_severity = min(1.0, effective_severity + 0.05)
        elif self.trial_player_argument == "evidence":
            effective_severity = max(0.0, effective_severity - 0.08)

        risk = self._fuzzy_legal_risk({
            "severity": effective_severity,
            "intent_type": intent_type,
        })

        if intent_type in ("threat_violence", "threat_robbery") and risk >= 0.9:
            return "detained"
        if intent_type == "push_assault" and risk < 0.55:
            return "warning"
        if risk < 0.4 and avg_trust >= 0.45:
            return "acquitted"
        if risk >= 0.72 and avg_trust < 0.45:
            return "guilty"
        if risk >= 0.9:
            return "detained"
        if risk < 0.62 and avg_trust >= 0.35:
            return "warning"
        return "guilty" if avg_trust < 0.28 else "warning"

    def _update_trial(self, dt):
        self.trial_timer += dt

        if self.trial_timer < 2.0:
            self.trial_stage = "jail"
            self.trial_phase_text = f"{self.trial_judge}: Court is in session."
        elif self.trial_timer < 4.0:
            self.trial_stage = "court_open"
            self.interior_building_name = "castle_courtroom"
            self.interior_player_x = SCREEN_WIDTH // 2
            self.interior_player_y = SCREEN_HEIGHT - 180
            self.trial_phase_text = f"Accuser ({self.trial_accuser}): {self.trial_accuser_statement}"
        elif self.trial_timer < 7.0:
            self.trial_stage = "defendant"
            self.trial_phase_text = f"Defendant ({self.trial_defendant}): {self.trial_defendant_statement}"
            if self.trial_player_argument is None:
                self.trial_phase_text += " | Press 1: Remorse  2: Deflect  3: Evidence"
        elif self.trial_timer < 9.5:
            self.trial_stage = "deliberation"
            self.trial_phase_text = f"{self.trial_judge}: The court is deliberating based on severity and testimony."

        if self.trial_verdict is None and self.trial_timer >= 9.5:
            self.trial_stage = "verdict"
            self.trial_verdict = self._compute_trial_verdict()
            if self.trial_verdict == "detained":
                self.legal_status_message = "Trial verdict: DETAINED by royal order. No immediate release."
                self.trial_phase_text = f"{self.trial_judge}: Due to severe offense, detention continues."
                for npc in self.npcs:
                    npc.behavior_vector["trust"] = max(0.0, npc.behavior_vector.get("trust", 0.5) - 0.1)
            elif self.trial_verdict == "guilty":
                self.legal_status_message = "Trial verdict: GUILTY. Temporary detention enforced."
                self.trial_phase_text = f"{self.trial_judge}: Guilty. Detention is ordered."
                for npc in self.npcs:
                    npc.behavior_vector["trust"] = max(0.0, npc.behavior_vector.get("trust", 0.5) - 0.06)
            elif self.trial_verdict == "acquitted":
                self.legal_status_message = "Trial verdict: ACQUITTED. You are free to go."
                self.trial_phase_text = f"{self.trial_judge}: Evidence insufficient. Acquitted."
            else:
                self.legal_status_message = "Trial verdict: WARNING. You will be released soon."
                self.trial_phase_text = f"{self.trial_judge}: Final warning. Maintain public order."
                for npc in self.npcs:
                    npc.behavior_vector["trust"] = max(0.0, npc.behavior_vector.get("trust", 0.5) - 0.02)

        if self.trial_verdict is None:
            return

        self.trial_release_timer += dt
        if self.trial_verdict == "acquitted" and self.trial_release_timer >= 2.0:
            self._release_from_jail()
        elif self.trial_verdict == "guilty" and self.trial_release_timer >= 10.0:
            self._release_from_jail()
        elif self.trial_verdict == "warning" and self.trial_release_timer >= 4.0:
            self._release_from_jail()

    def _plead_for_mercy(self):
        self.trial_verdict = "guilty"
        self.trial_release_timer = 0.0
        self.trial_phase_text = f"{self.trial_judge}: Mercy granted. Sentence reduced to temporary detention."
        self.legal_status_message = "Royal mercy granted after plea."

    def _submit_trial_argument(self, key):
        if key == pygame.K_1:
            self.trial_player_argument = "remorse"
            self.trial_player_argument_text = "I deeply regret my actions and ask forgiveness."
        elif key == pygame.K_2:
            self.trial_player_argument = "deflect"
            self.trial_player_argument_text = "This is exaggerated; I am being unfairly judged."
        else:
            self.trial_player_argument = "evidence"
            self.trial_player_argument_text = "Please review all evidence carefully before judgment."
        self.trial_phase_text = f"Defendant ({self.trial_defendant}): {self.trial_player_argument_text}"

    def _release_from_jail(self):
        self.player_arrested = False
        self.trial_active = False
        self.interior_active = False
        self.interior_building_name = None
        self.enter_cooldown_timer = AUTO_ENTER_COOLDOWN

        cx, cy, cw, ch = ZONES.get("castle", (30, 0, 18, 10))
        self.player.x = float((cx + cw // 2) * TILE_SIZE)
        self.player.y = float((cy + ch + 2) * TILE_SIZE)
        self.player.direction = "down"
        self.legal_status_message = "Released after trial. Stay lawful."
        self.player_crime_points = max(0.0, self.player_crime_points - 0.9)

    def _render_castle_jail_interior(self, surface):
        room = pygame.Rect(220, 96, SCREEN_WIDTH - 440, SCREEN_HEIGHT - 192)
        surface.fill((8, 10, 14))
        pygame.draw.rect(surface, (66, 72, 84), room)

        # Stone tiles
        tile = 32
        inner = room.inflate(-18, -22)
        for y in range(inner.top, inner.bottom, tile):
            for x in range(inner.left, inner.right, tile):
                shade = (96, 102, 114) if ((x // tile + y // tile) % 2 == 0) else (86, 92, 104)
                pygame.draw.rect(surface, shade, (x, y, tile, tile))

        # Iron bars
        bar_top = room.top + 34
        bar_bottom = room.bottom - 36
        for x in range(room.left + 36, room.right - 35, 26):
            pygame.draw.line(surface, (170, 176, 188), (x, bar_top), (x, bar_bottom), 4)
        pygame.draw.line(surface, (150, 156, 168), (room.left + 24, bar_top), (room.right - 24, bar_top), 5)
        pygame.draw.line(surface, (130, 136, 148), (room.left + 24, bar_bottom), (room.right - 24, bar_bottom), 5)

        # Player sprite in cell
        sprite = SpriteAssets.get().get_character(
            "player",
            self.interior_player_direction,
            False,
            self.interior_anim_timer,
            tile_size=44,
        )
        px = int(self.interior_player_x)
        py = int(self.interior_player_y)
        pygame.draw.ellipse(surface, (0, 0, 0, 85), (px - 10, py + 4, 20, 6))
        if sprite is not None:
            surface.blit(sprite, (px - sprite.get_width() // 2, py - sprite.get_height() + 8))

        title = self.prompt_font.render("Castle Jail Court", True, (234, 236, 245))
        msg = self.prompt_font.render(self.legal_status_message, True, (220, 202, 146))
        hint_text = "Trial in progress..."
        if self.trial_verdict == "detained":
            hint_text = "Press T to plead for royal mercy"
        hint = self.prompt_font.render(hint_text, True, (186, 192, 208))
        judge = self.prompt_font.render(f"Judge: {self.trial_judge}", True, (210, 218, 232))
        accuser = self.prompt_font.render(f"Accuser: {self.trial_accuser or 'Town Guard'}", True, (210, 218, 232))
        defendant = self.prompt_font.render(f"Defendant: {self.trial_defendant}", True, (210, 218, 232))
        phase = self.prompt_font.render(self.trial_phase_text, True, (228, 214, 178))
        surface.blit(title, (room.left + 8, room.top - 26))
        surface.blit(msg, (room.left + 8, room.bottom + 10))
        surface.blit(hint, (room.right - hint.get_width() - 8, room.top - 26))
        surface.blit(judge, (room.left + 8, room.bottom + 32))
        surface.blit(accuser, (room.left + 8, room.bottom + 52))
        surface.blit(defendant, (room.left + 8, room.bottom + 72))
        surface.blit(phase, (room.left + 8, room.bottom + 94))

    def _render_castle_courtroom(self, surface):
        hall = pygame.Rect(120, 72, SCREEN_WIDTH - 240, SCREEN_HEIGHT - 170)
        surface.fill((14, 10, 8))
        pygame.draw.rect(surface, (86, 64, 48), hall)

        # Stone floor pattern
        tile = 34
        inner = hall.inflate(-20, -22)
        for y in range(inner.top, inner.bottom, tile):
            for x in range(inner.left, inner.right, tile):
                c = (128, 104, 84) if ((x // tile + y // tile) % 2 == 0) else (118, 94, 76)
                pygame.draw.rect(surface, c, (x, y, tile, tile))

        # Judge bench and throne (Queen)
        bench = pygame.Rect(hall.centerx - 180, hall.top + 34, 360, 80)
        pygame.draw.rect(surface, (104, 74, 52), bench, border_radius=6)
        pygame.draw.rect(surface, (72, 50, 36), bench, 2, border_radius=6)
        throne = pygame.Rect(hall.centerx - 40, hall.top + 12, 80, 70)
        pygame.draw.rect(surface, (170, 44, 54), throne, border_radius=8)
        pygame.draw.rect(surface, (212, 182, 76), throne, 2, border_radius=8)

        queen_label = self.prompt_font.render("Queen Seraphina (Judge)", True, (245, 220, 168))
        surface.blit(queen_label, (hall.centerx - queen_label.get_width() // 2, hall.top + 118))

        # Attendee rows: render all NPCs as audience markers.
        per_row = 8
        start_y = hall.top + 180
        for idx, npc in enumerate(self.npcs):
            row = idx // per_row
            col = idx % per_row
            cx = hall.left + 90 + col * 110
            cy = start_y + row * 70
            if cy > hall.bottom - 70:
                break
            sprite = SpriteAssets.get().get_character(
                npc.npc_class,
                "down",
                False,
                0.0,
                tint=npc.color,
                tile_size=30,
                variant_seed=npc.name,
            )
            if sprite is not None:
                pygame.draw.ellipse(surface, (0, 0, 0, 70), (cx - 9, cy + 6, 18, 5))
                surface.blit(sprite, (cx - sprite.get_width() // 2, cy - sprite.get_height() + 8))

        # Defendant and accuser stands
        acc_pos = (hall.left + 170, hall.bottom - 84)
        def_pos = (hall.right - 170, hall.bottom - 84)
        pygame.draw.rect(surface, (78, 58, 42), (acc_pos[0] - 28, acc_pos[1] + 6, 56, 26), border_radius=4)
        pygame.draw.rect(surface, (78, 58, 42), (def_pos[0] - 28, def_pos[1] + 6, 56, 26), border_radius=4)

        acc = self.prompt_font.render(f"Accuser: {self.trial_accuser or 'Guard'}", True, (236, 224, 204))
        dfd = self.prompt_font.render("Defendant: You", True, (236, 224, 204))
        surface.blit(acc, (acc_pos[0] - acc.get_width() // 2, acc_pos[1] + 36))
        surface.blit(dfd, (def_pos[0] - dfd.get_width() // 2, def_pos[1] + 36))

        # Player marker at defendant stand
        player_sprite = SpriteAssets.get().get_character(
            "player", "up", False, self.interior_anim_timer, tile_size=36
        )
        if player_sprite is not None:
            surface.blit(player_sprite, (def_pos[0] - player_sprite.get_width() // 2, def_pos[1] - 28))

        title = self.prompt_font.render("Royal Court of Castle", True, (246, 236, 206))
        phase = self.prompt_font.render(self.trial_phase_text, True, (242, 212, 154))
        surface.blit(title, (hall.left + 10, hall.top - 28))
        surface.blit(phase, (hall.left + 10, hall.bottom + 8))
        if self.trial_player_argument_text:
            arg = self.prompt_font.render(f"Your point: {self.trial_player_argument_text}", True, (194, 212, 246))
            surface.blit(arg, (hall.left + 10, hall.bottom + 30))

    def _render_law_overlay(self):
        if self.guard_capture_active:
            text = "WANTED: Guards are pursuing you"
            color = (242, 86, 86)
        elif self.player_arrested:
            text = "ARRESTED: Awaiting trial verdict"
            color = (246, 196, 96)
        elif self.legal_status_message:
            text = self.legal_status_message
            color = (176, 214, 255)
        else:
            return

        surf = self.prompt_font.render(text, True, (255, 255, 255))
        w = surf.get_width() + 20
        h = surf.get_height() + 10
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(box, (20, 24, 34, 210), (0, 0, w, h), border_radius=5)
        pygame.draw.rect(box, color, (0, 0, w, h), 1, border_radius=5)
        box.blit(surf, (10, 5))
        self.screen.blit(box, (SCREEN_WIDTH // 2 - w // 2, 10))

        if self.raid_active or self.raid_status_message:
            rt = self._raid_live_status_text()
            rs = self.prompt_font.render(rt, True, (255, 255, 255))
            rw = rs.get_width() + 20
            rh = rs.get_height() + 10
            rbox = pygame.Surface((rw, rh), pygame.SRCALPHA)
            pygame.draw.rect(rbox, (30, 18, 16, 220), (0, 0, rw, rh), border_radius=5)
            pygame.draw.rect(rbox, (220, 102, 82), (0, 0, rw, rh), 1, border_radius=5)
            rbox.blit(rs, (10, 5))
            self.screen.blit(rbox, (SCREEN_WIDTH // 2 - rw // 2, 34))

    def _raid_live_status_text(self):
        if not self.raid_active:
            return self.raid_status_message or ""

        threat = None
        for chest in sorted(self.treasure_chests, key=lambda c: c.get("value", 0.0), reverse=True):
            if chest.get("value", 0.0) > 0.0:
                threat = chest
                break

        loot_text = f"Loot stolen: {self.raid_loot_taken:.1f}"
        if threat is None:
            return f"Raid ongoing | {loot_text} | No chest stock left"
        return (
            f"Raid ongoing | {loot_text} | Threat: {threat['tier']} chest @ {threat['zone']}"
        )

    def _chest_tier_for_zone(self, zone_name):
        if zone_name == "castle":
            return "royal", 220.0
        if zone_name.startswith(("noble_house_",)):
            return "noble", 120.0
        if zone_name.startswith(("trader_house_",)):
            return "merchant", 72.0
        if zone_name.startswith(("peasant_house_",)):
            return "peasant", 28.0
        return "common", 40.0

    def _owner_class_for_zone(self, zone_name):
        if zone_name == "castle":
            return "Royal"
        if zone_name.startswith("noble_house_"):
            return random.choice(["Noble", "Elite"])
        if zone_name.startswith("trader_house_"):
            return random.choice(["Merchant", "Traveller", "Blacksmith"])
        if zone_name.startswith("peasant_house_"):
            return random.choice(["Peasant", "Labourer"])
        return "Peasant"

    def _init_treasure_chests(self):
        """Create treasure chest nodes by class/zone hierarchy for raid looting."""
        self.treasure_chests = []
        for zone_name in sorted(ZONES.keys()):
            if zone_name == "castle" or zone_name.startswith(("noble_house_", "trader_house_", "peasant_house_")):
                tier, base_value = self._chest_tier_for_zone(zone_name)
                cx, cy = self.world.get_zone_center(zone_name)
                value = max(6.0, base_value + random.uniform(-0.1, 0.2) * base_value)
                self.treasure_chests.append({
                    "zone": zone_name,
                    "tier": tier,
                    "value": float(value),
                    "max_value": float(value),
                    "x": float(cx),
                    "y": float(cy),
                    "owner_class": self._owner_class_for_zone(zone_name),
                })

    def _render_treasure_chests(self, surface):
        if not self.treasure_chests:
            return
        cam_rect = self.camera.get_visible_rect()
        for chest in self.treasure_chests:
            if chest["value"] <= 0.0:
                continue
            sx = int(chest["x"] - cam_rect.x)
            sy = int(chest["y"] - cam_rect.y)
            if sx < -30 or sy < -30 or sx > SCREEN_WIDTH + 30 or sy > SCREEN_HEIGHT + 30:
                continue

            bw, bh = 18, 12
            lid = (154, 104, 56)
            box = (120, 78, 38)
            trim = (212, 176, 76)
            if chest["tier"] == "royal":
                trim = (242, 206, 88)
            elif chest["tier"] == "peasant":
                trim = (180, 136, 62)

            pygame.draw.rect(surface, box, (sx - bw // 2, sy - bh // 2, bw, bh), border_radius=2)
            pygame.draw.rect(surface, lid, (sx - bw // 2, sy - bh // 2 - 4, bw, 5), border_radius=2)
            pygame.draw.rect(surface, trim, (sx - bw // 2, sy - bh // 2, bw, bh), 1, border_radius=2)
            pygame.draw.rect(surface, trim, (sx - 1, sy - 2, 2, 4))

    def _select_raid_chest_target(self, raider):
        """Raiders prefer highest-value chests, then nearest among top-value tier."""
        available = [c for c in self.treasure_chests if c["value"] > 0.0]
        if not available:
            return None
        best_value = max(c["value"] for c in available)
        candidates = [c for c in available if c["value"] >= best_value * 0.75]
        best = None
        best_d2 = float("inf")
        for c in candidates:
            dx = c["x"] - raider["x"]
            dy = c["y"] - raider["y"]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = c
        return best

    def _castle_entry_points(self):
        """Return castle entrance and interior target points for raid routing."""
        castle = self.world.buildings.get("castle")
        if castle is None:
            cx, cy = self.world.get_zone_center("castle")
            return (float(cx), float(cy), float(cx), float(cy))

        ex, ey = castle.get_entrance()
        ix, iy = castle.get_interior_point()
        return (float(ex), float(ey), float(ix), float(iy))

    def _apply_raid_npc_response(self):
        """Only guards defend treasure; all other villagers run to their homes in chaos."""
        guard_targets = self._build_guard_defense_points()
        guard_index = 0

        shelter_classes = {"Noble", "Merchant", "Peasant", "Blacksmith", "Traveller", "Labourer"}

        for npc in self.npcs:
            if npc.npc_class == "Elite":
                npc.show_speech("Hold the line! Defend the treasure!", 2.3)
                npc.show_emotion("⚔️", 2.4)
                npc.raid_shelter_mode = False
                if guard_targets:
                    tx, ty = guard_targets[min(guard_index, len(guard_targets) - 1)]
                    guard_index += 1
                    npc.raid_guard_target = (tx, ty)
                    npc._navigate_to(tx, ty)
                else:
                    npc._go_to_zone("castle")
                npc.state = "walking"
                self.social_system.remember_event(
                    npc,
                    "Rallied to defend the village during a raid",
                    trust_shift=0.02,
                    mood_shift=0.01,
                )
                continue

            if npc.npc_class in shelter_classes:
                home_zone = npc._get_home_zone() if hasattr(npc, "_get_home_zone") else "town_square"
                npc.raid_shelter_mode = True
                npc.show_speech("Raid! Run inside, lock the doors!", 2.0)
                npc.show_emotion("😨", 2.2)
                npc._go_to_zone(home_zone)
                npc.state = "walking"
                self.social_system.remember_event(
                    npc,
                    "Ran inside home during raid panic",
                    trust_shift=-0.01,
                    mood_shift=-0.02,
                    negative=True,
                )
            else:
                npc.raid_shelter_mode = False

    def _build_guard_defense_points(self):
        """Create two guard squads: one at castle gate, one around inner treasure."""
        guards = [n for n in self.npcs if n.npc_class == "Elite"]
        if not guards:
            return []

        entry_x, entry_y, inner_x, inner_y = self._castle_entry_points()
        guard_count = len(guards)
        gate_count = max(1, int(round(guard_count * 0.6)))
        inner_count = max(0, guard_count - gate_count)
        points = []

        # Gate squad holds an arc in front of the entrance.
        for i in range(gate_count):
            frac = 0.0 if gate_count == 1 else (i / float(gate_count - 1))
            angle = (-0.8 + (1.6 * frac))
            gx = entry_x + math.sin(angle) * 28.0
            gy = entry_y + 14.0 + abs(math.cos(angle)) * 10.0
            points.append((gx, gy))

        # Inner squad circles treasury interior point.
        for i in range(inner_count):
            angle = (2.0 * math.pi * i) / max(1, inner_count)
            gx = inner_x + math.cos(angle) * 18.0
            gy = inner_y + math.sin(angle) * 14.0
            points.append((gx, gy))

        return points

    def _maintain_guard_formation(self):
        """Keep guard squads anchored to castle defense points during active raids."""
        guards = sorted([n for n in self.npcs if n.npc_class == "Elite"], key=lambda n: n.name)
        if not guards:
            return

        targets = self._build_guard_defense_points()
        if not targets:
            return

        for idx, guard in enumerate(guards):
            tx, ty = targets[min(idx, len(targets) - 1)]
            guard.raid_guard_target = (tx, ty)
            dx = tx - guard.x
            dy = ty - guard.y
            d2 = dx * dx + dy * dy
            if d2 > (20.0 * 20.0):
                guard._navigate_to(tx, ty)
                guard.state = "walking"
            elif guard.state == "walking" and (not guard.path or guard.path_index >= len(guard.path)):
                guard.state = "working"
                guard.state_timer = 0.0

    def _spawn_raid(self):
        self.raid_active = True
        self.raid_entities = []
        self.raid_total_spawned = random.randint(5, 8)
        self.raid_kills_by_player = 0
        self.raid_villager_harm = 0
        self.raid_loot_taken = 0.0
        self.raid_status_message = "Raiders are after treasure chests! Guards defend, villagers flee home."
        self._apply_raid_npc_response()
        self.social_system.remember_group_event(
            self.npcs,
            "A sudden raid threatened village homes and treasure",
            mood_shift=-0.01,
            negative=True,
        )

        for _ in range(self.raid_total_spawned):
            side = random.choice(["left", "right", "top", "bottom"])
            if side == "left":
                x = random.randint(1, 3) * TILE_SIZE
                y = random.randint(4, 56) * TILE_SIZE
            elif side == "right":
                x = random.randint(76, 78) * TILE_SIZE
                y = random.randint(4, 56) * TILE_SIZE
            elif side == "top":
                x = random.randint(5, 74) * TILE_SIZE
                y = random.randint(1, 3) * TILE_SIZE
            else:
                x = random.randint(5, 74) * TILE_SIZE
                y = random.randint(56, 58) * TILE_SIZE

            retreat_x = -40.0 if side == "left" else (WORLD_WIDTH + 40.0 if side == "right" else float(x))
            retreat_y = -40.0 if side == "top" else (WORLD_HEIGHT + 40.0 if side == "bottom" else float(y))
            role = random.choices(
                ["looter", "hunter", "skirmisher"],
                weights=[0.5, 0.28, 0.22],
                k=1,
            )[0]
            tint = random.choice([
                (176, 58, 56),
                (148, 44, 44),
                (122, 60, 48),
                (168, 78, 52),
            ])

            self.raid_entities.append({
                "x": float(x),
                "y": float(y),
                "speed": random.uniform(56.0, 78.0),
                "hp": 1,
                "role": role,
                "state": "advance",
                "state_timer": random.uniform(1.4, 2.6),
                "attack_cooldown": random.uniform(0.3, 1.1),
                "loot_cooldown": random.uniform(0.4, 1.0),
                "carried_loot": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "anim_phase": random.random() * 10.0,
                "direction": random.choice(["up", "down", "left", "right"]),
                "retreat_x": retreat_x,
                "retreat_y": retreat_y,
                "castle_inside": False,
                "tint": tint,
                "name": f"Raider-{len(self.raid_entities) + 1}",
                "target_npc": None,
                "target_chest_zone": None,
            })

    def _nearest_villager_for_raid(self, rx, ry):
        candidates = [n for n in self.npcs if n.npc_class not in ("Elite", "Royal")]
        if not candidates:
            return None
        best = None
        best_d2 = 10e12
        for npc in candidates:
            dx = npc.x - rx
            dy = npc.y - ry
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = npc
        return best

    def _update_raid_system(self, dt):
        if self.player_arrested or self.trial_active:
            return

        if not self.raid_active:
            self.raid_timer -= dt
            if self.raid_timer <= 0.0:
                self._spawn_raid()
            return

        self.guard_formation_timer += dt
        if self.guard_formation_timer >= 1.3:
            self.guard_formation_timer = 0.0
            self._maintain_guard_formation()

        survivors = []
        castle_entry_x, castle_entry_y, castle_inner_x, castle_inner_y = self._castle_entry_points()
        for raider in self.raid_entities:
            raider["state_timer"] -= dt
            raider["attack_cooldown"] = max(0.0, raider.get("attack_cooldown", 0.0) - dt)
            raider["loot_cooldown"] = max(0.0, raider.get("loot_cooldown", 0.0) - dt)
            raider["anim_phase"] = raider.get("anim_phase", 0.0) + dt

            target_chest = self._select_raid_chest_target(raider)
            target_npc = self._nearest_villager_for_raid(raider["x"], raider["y"])

            if raider.get("carried_loot", 0.0) > 0.0:
                raider["state"] = "retreat"

            if raider["state"] == "retreat":
                tx, ty = raider.get("retreat_x", raider["x"]), raider.get("retreat_y", raider["y"])
                raider["target_chest_zone"] = None
            elif target_chest is not None:
                if target_chest["zone"] == "castle" and not raider.get("castle_inside", False):
                    tx, ty = castle_entry_x, castle_entry_y
                    raider["target_chest_zone"] = "castle_entry"
                else:
                    tx, ty = target_chest["x"], target_chest["y"]
                    raider["target_chest_zone"] = target_chest["zone"]
            else:
                tx, ty = (self.player.x, self.player.y)
                if target_npc is not None:
                    tx, ty = target_npc.x, target_npc.y
                raider["target_chest_zone"] = None

            if target_chest is not None and target_chest["zone"] == "castle":
                exdx = castle_entry_x - raider["x"]
                exdy = castle_entry_y - raider["y"]
                if (exdx * exdx + exdy * exdy) <= (24 * 24):
                    raider["castle_inside"] = True
                if raider.get("castle_inside", False) and raider["state"] != "retreat":
                    tx, ty = castle_inner_x, castle_inner_y

            if raider["state"] != "retreat" and target_npc is not None and raider.get("role") in ("hunter", "skirmisher"):
                ndx = target_npc.x - raider["x"]
                ndy = target_npc.y - raider["y"]
                npc_d2 = ndx * ndx + ndy * ndy
                if npc_d2 <= (170 * 170):
                    if raider.get("role") == "hunter":
                        raider["state"] = "harass"
                        tx, ty = target_npc.x, target_npc.y
                    else:
                        raider["state"] = "flank"
                        flank = -1.0 if (hash(raider.get("name", "r")) % 2 == 0) else 1.0
                        tx = target_npc.x + flank * 34.0
                        ty = target_npc.y + (22.0 if flank < 0 else -22.0)

            if raider["state"] in ("advance", "harass", "flank") and raider["state_timer"] <= 0.0:
                raider["state"] = "advance"
                raider["state_timer"] = random.uniform(1.2, 2.4)

            dx = tx - raider["x"]
            dy = ty - raider["y"]
            dist = math.sqrt(dx * dx + dy * dy) if (dx or dy) else 1.0

            repulse_x = 0.0
            repulse_y = 0.0
            for other in self.raid_entities:
                if other is raider:
                    continue
                odx = raider["x"] - other["x"]
                ody = raider["y"] - other["y"]
                od2 = odx * odx + ody * ody
                if 1.0 < od2 < (34.0 * 34.0):
                    inv = 1.0 / math.sqrt(od2)
                    repulse_x += odx * inv
                    repulse_y += ody * inv

            steer_x = (dx / dist) + repulse_x * 0.34
            steer_y = (dy / dist) + repulse_y * 0.34
            steer_len = math.sqrt(steer_x * steer_x + steer_y * steer_y) if (steer_x or steer_y) else 1.0
            steer_x /= steer_len
            steer_y /= steer_len

            speed_mul = 1.0
            if raider["state"] == "harass":
                speed_mul = 1.08
            elif raider["state"] == "flank":
                speed_mul = 1.14
            elif raider["state"] == "retreat":
                speed_mul = 1.18

            raider["vx"] = steer_x * raider["speed"] * speed_mul
            raider["vy"] = steer_y * raider["speed"] * speed_mul
            raider["x"] += raider["vx"] * dt
            raider["y"] += raider["vy"] * dt

            if abs(raider["vx"]) > abs(raider["vy"]):
                raider["direction"] = "right" if raider["vx"] > 0 else "left"
            else:
                raider["direction"] = "down" if raider["vy"] > 0 else "up"

            harmed = False
            looted = False

            if raider["state"] == "retreat":
                edge_margin = 46.0
                if (
                    raider["x"] < -edge_margin or raider["x"] > WORLD_WIDTH + edge_margin or
                    raider["y"] < -edge_margin or raider["y"] > WORLD_HEIGHT + edge_margin
                ):
                    continue

            if target_chest is not None and raider["state"] != "retreat":
                cdx = target_chest["x"] - raider["x"]
                cdy = target_chest["y"] - raider["y"]
                if (cdx * cdx + cdy * cdy) <= (30 * 30) and raider.get("loot_cooldown", 0.0) <= 0.0:
                    loot_amt = min(target_chest["value"], random.uniform(6.0, 12.0))
                    target_chest["value"] = max(0.0, target_chest["value"] - loot_amt)
                    self.raid_loot_taken += loot_amt
                    raider["carried_loot"] = raider.get("carried_loot", 0.0) + loot_amt
                    raider["state"] = "retreat"
                    raider["loot_cooldown"] = random.uniform(0.8, 1.4)
                    self.raid_status_message = f"Raiders looted {target_chest['tier']} chest at {target_chest['zone']}"

                    # Raid loot impacts village economy immediately.
                    self.economy_system.ledger["treasury"] = max(
                        0.0,
                        self.economy_system.ledger.get("treasury", 0.0) - loot_amt * 0.35,
                    )
                    self.economy_system.ledger["market_turnover"] = max(
                        0.0,
                        self.economy_system.ledger.get("market_turnover", 0.0) - loot_amt * 0.08,
                    )
                    self.economy_system._event(
                        f"Raid loss: {loot_amt:.1f} coin stolen from {target_chest['zone']}"
                    )

                    affected = [
                        n for n in self.npcs
                        if n.npc_class == target_chest.get("owner_class")
                    ]
                    if affected:
                        for npc in affected:
                            acc = self.economy_system.accounts.get(id(npc))
                            if acc is not None:
                                acc["coin"] = max(0.0, acc.get("coin", 0.0) - loot_amt * 0.25)
                        self.social_system.remember_group_event(
                            affected,
                            f"Lost treasure in {target_chest['zone']} during raid",
                            trust_shift=-0.02,
                            mood_shift=-0.04,
                            negative=True,
                        )
                    self.social_system.remember_group_event(
                        self.npcs,
                        f"Raiders looted treasure from {target_chest['zone']}",
                        trust_shift=-0.012,
                        mood_shift=-0.02,
                        negative=True,
                    )
                    looted = True
                    raider["castle_inside"] = False

            if target_npc is not None:
                ddx = target_npc.x - raider["x"]
                ddy = target_npc.y - raider["y"]
                if (ddx * ddx + ddy * ddy) <= (26 * 26) and raider.get("attack_cooldown", 0.0) <= 0.0:
                    target_npc.behavior_vector["trust"] = max(0.0, target_npc.behavior_vector.get("trust", 0.5) - 0.08)
                    target_npc.behavior_vector["mood"] = max(0.0, target_npc.behavior_vector.get("mood", 0.5) - 0.08)
                    target_npc.show_emotion(":(", 2.0)
                    target_npc.show_speech("Raiders! Help!", 2.0)
                    self.raid_villager_harm += 1
                    raider["attack_cooldown"] = random.uniform(1.2, 2.0)
                    if raider.get("role") == "hunter":
                        raider["state"] = "advance"
                        raider["state_timer"] = random.uniform(1.0, 2.0)
                    self.social_system.remember_event(
                        target_npc,
                        "Was attacked during a raid",
                        trust_shift=-0.05,
                        mood_shift=-0.08,
                        negative=True,
                    )
                    harmed = True

            if not harmed and not looted:
                survivors.append(raider)

        self.raid_entities = survivors

        if not self.raid_entities:
            self._finish_raid()

    def _finish_raid(self):
        self.raid_active = False
        self.guard_formation_timer = 0.0
        loot_penalty = min(1.0, self.raid_loot_taken / 180.0)
        reward_score = (
            (self.raid_kills_by_player / max(1, self.raid_total_spawned))
            - (0.12 * self.raid_villager_harm)
            - (0.8 * loot_penalty)
        )
        if reward_score >= 0.55:
            for npc in self.npcs:
                npc.behavior_vector["trust"] = min(1.0, npc.behavior_vector.get("trust", 0.5) + 0.08)
                npc.behavior_vector["mood"] = min(1.0, npc.behavior_vector.get("mood", 0.5) + 0.04)
            self.raid_status_message = "Queen Seraphina rewards your defense. Village trust increased."
            self.social_system.remember_group_event(
                self.npcs,
                "Raid was repelled and treasures were protected",
                trust_shift=0.05,
                mood_shift=0.03,
            )
        elif reward_score >= 0.2:
            for npc in self.npcs:
                npc.behavior_vector["trust"] = min(1.0, npc.behavior_vector.get("trust", 0.5) + 0.03)
            self.raid_status_message = "Raid repelled, but some loss occurred. The Queen acknowledges your effort."
            self.social_system.remember_group_event(
                self.npcs,
                "Raid ended with partial losses but village survived",
                trust_shift=0.01,
                mood_shift=-0.01,
                negative=True,
            )
        else:
            for npc in self.npcs:
                npc.behavior_vector["trust"] = max(0.0, npc.behavior_vector.get("trust", 0.5) - 0.03)
            self.raid_status_message = "Raid ended with heavy losses and stolen treasure. Villagers are shaken."
            self.social_system.remember_group_event(
                self.npcs,
                "Raiders stole village treasure and left deep fear",
                trust_shift=-0.04,
                mood_shift=-0.05,
                negative=True,
            )

        # Refill a share of wealth over time so future raids remain meaningful.
        for chest in self.treasure_chests:
            regen = chest["max_value"] * random.uniform(0.18, 0.32)
            chest["value"] = min(chest["max_value"], chest["value"] + regen)

        for npc in self.npcs:
            npc.raid_shelter_mode = False

        self.raid_timer = random.uniform(130.0, 220.0)

    def _player_sword_attack(self):
        if self.sword_cooldown > 0.0:
            return
        self.sword_cooldown = 0.35
        self.sword_fx_timer = 0.12

        dir_vec = {
            "up": (0.0, -1.0),
            "down": (0.0, 1.0),
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
        }.get(self.player.direction, (0.0, 1.0))
        sx = self.player.x + dir_vec[0] * 30.0
        sy = self.player.y + dir_vec[1] * 30.0
        self.sword_fx_points = (sx, sy)

        # Raiders hit check
        survivors = []
        for raider in self.raid_entities:
            dx = raider["x"] - sx
            dy = raider["y"] - sy
            if (dx * dx + dy * dy) <= (34 * 34):
                self.raid_kills_by_player += 1
                continue
            survivors.append(raider)
        self.raid_entities = survivors

        # Villager hit check: heavy trust penalty and legal escalation.
        hit_villager = None
        for npc in self.npcs:
            dx = npc.x - sx
            dy = npc.y - sy
            if (dx * dx + dy * dy) <= (30 * 30):
                hit_villager = npc
                break

        if hit_villager is not None:
            hit_villager.show_emotion(":O", 2.2)
            hit_villager.show_speech("You attacked me! Guards!", 2.5)
            for npc in self.npcs:
                npc.behavior_vector["trust"] = max(0.0, npc.behavior_vector.get("trust", 0.5) - 0.07)
            self.player_crime_points = min(3.0, self.player_crime_points + 0.7)
            self.legal_status_message = "You attacked a villager. Trust has dropped sharply."
            self._start_guard_capture({
                "npc_name": hit_villager.name,
                "npc_class": hit_villager.npc_class,
                "intent_type": "threat_violence",
                "severity": 0.92,
                "player_message": "[sword attack on villager]",
            })

    def _render_raiders(self, surface):
        if not self.raid_entities:
            return
        cam_rect = self.camera.get_visible_rect()
        font = pygame.font.SysFont("Arial", 10, bold=True)
        for raider in self.raid_entities:
            sx = int(raider["x"] - cam_rect.x)
            sy = int(raider["y"] - cam_rect.y)
            if sx < -40 or sy < -40 or sx > SCREEN_WIDTH + 40 or sy > SCREEN_HEIGHT + 40:
                continue

            moving = (abs(raider.get("vx", 0.0)) + abs(raider.get("vy", 0.0))) > 4.0
            sprite = SpriteAssets.get().get_character(
                "Elite",
                raider.get("direction", "down"),
                moving,
                raider.get("anim_phase", 0.0),
                tint=raider.get("tint", (156, 52, 50)),
                tile_size=36,
                variant_seed=raider.get("name", "Raider"),
            )

            if sprite is not None:
                bob = int(math.sin(raider.get("anim_phase", 0.0) * 8.0) * 2) if moving else 0
                pygame.draw.ellipse(surface, (0, 0, 0, 60), (sx - 8, sy + 5, 16, 6))
                surface.blit(sprite, (sx - sprite.get_width() // 2, sy - sprite.get_height() + 9 + bob))
            else:
                pygame.draw.ellipse(surface, (0, 0, 0, 80), (sx - 8, sy + 5, 16, 6))
                pygame.draw.circle(surface, (146, 36, 34), (sx, sy - 4), 10)
                pygame.draw.circle(surface, (220, 78, 72), (sx - 2, sy - 6), 4)

            role = raider.get("role", "raider")
            badge_color = {
                "looter": (230, 190, 74),
                "hunter": (230, 120, 90),
                "skirmisher": (120, 190, 230),
            }.get(role, (190, 190, 190))
            pygame.draw.circle(surface, badge_color, (sx + 10, sy - 11), 3)

            if raider.get("carried_loot", 0.0) > 0.0:
                pygame.draw.circle(surface, (224, 196, 84), (sx - 9, sy - 12), 3)

            name_surf = font.render(raider.get("name", "Raider"), True, (255, 235, 225))
            surface.blit(name_surf, (sx - name_surf.get_width() // 2, sy - 26))

    def _render_sword_fx(self, surface):
        if self.sword_fx_timer <= 0.0 or not self.sword_fx_points:
            return
        cam_rect = self.camera.get_visible_rect()
        sx = int(self.sword_fx_points[0] - cam_rect.x)
        sy = int(self.sword_fx_points[1] - cam_rect.y)
        pygame.draw.arc(surface, (236, 236, 255), (sx - 24, sy - 24, 48, 48), 0.4, 2.5, 3)
        pygame.draw.arc(surface, (180, 220, 255), (sx - 20, sy - 20, 40, 40), 0.45, 2.4, 2)

    def _render_interior(self, surface):
        if self.interior_building_name == "castle_jail":
            self._render_castle_jail_interior(surface)
            return
        if self.interior_building_name == "castle_courtroom":
            self._render_castle_courtroom(surface)
            return

        room = pygame.Rect(170, 86, SCREEN_WIDTH - 340, SCREEN_HEIGHT - 172)
        wall_color = (90, 72, 58)
        floor_a = (156, 124, 86)
        floor_b = (146, 114, 78)
        trim_color = (58, 42, 30)

        surface.fill((0, 0, 0))
        pygame.draw.rect(surface, wall_color, room)
        pygame.draw.rect(surface, trim_color, room, 5)

        pygame.draw.rect(surface, (112, 90, 72), (room.left + 8, room.top + 8, room.width - 16, 24), border_radius=5)
        pygame.draw.rect(surface, (78, 58, 44), (room.left + 8, room.top + 32, room.width - 16, 4))
        for bx in range(room.left + 26, room.right - 20, 88):
            pygame.draw.rect(surface, (74, 54, 40), (bx, room.top + 10, 12, 24), border_radius=2)

        tile = 32
        inner = room.inflate(-24, -28)
        for y in range(inner.top, inner.bottom, tile):
            for x in range(inner.left, inner.right, tile):
                c = floor_a if ((x // tile + y // tile) % 2 == 0) else floor_b
                pygame.draw.rect(surface, c, (x, y, tile, tile))

        rug = pygame.Rect(room.centerx - 170, room.centery + 20, 340, 120)
        pygame.draw.rect(surface, (126, 38, 36), rug, border_radius=10)
        pygame.draw.rect(surface, (214, 188, 108), rug, 4, border_radius=10)
        for yy in range(rug.top + 10, rug.bottom - 8, 18):
            pygame.draw.line(surface, (170, 62, 58), (rug.left + 12, yy), (rug.right - 12, yy), 2)

        for i in range(3):
            wx = room.left + 100 + i * 220
            wy = room.top + 16
            wrect = pygame.Rect(wx, wy, 92, 62)
            pygame.draw.rect(surface, (56, 44, 34), wrect.inflate(8, 8), border_radius=4)
            pygame.draw.rect(surface, (176, 214, 242), wrect, border_radius=3)
            pygame.draw.rect(surface, (108, 82, 62), wrect, 3, border_radius=3)
            pygame.draw.line(surface, (108, 82, 62), (wx + 46, wy), (wx + 46, wy + 62), 2)
            pygame.draw.line(surface, (108, 82, 62), (wx, wy + 31), (wx + 92, wy + 31), 2)
            pygame.draw.rect(surface, (142, 205, 120), (wx + 3, wy + 36, 86, 23))
            pygame.draw.circle(surface, (54, 136, 70), (wx + 22, wy + 44), 8)

        for lx in (room.left + 54, room.right - 54):
            pygame.draw.rect(surface, (84, 60, 44), (lx - 8, room.top + 86, 16, 22), border_radius=3)
            pygame.draw.circle(surface, (246, 214, 132), (lx, room.top + 86), 6)
            glow = pygame.Surface((70, 58), pygame.SRCALPHA)
            for r in range(26, 8, -4):
                alpha = int(56 * (r / 26))
                pygame.draw.circle(glow, (255, 210, 120, alpha), (35, 22), r)
            surface.blit(glow, (lx - 35, room.top + 62), special_flags=pygame.BLEND_RGBA_ADD)

        door = pygame.Rect(room.centerx - 30, room.bottom - 20, 60, 20)
        pygame.draw.rect(surface, (66, 46, 32), door)
        pygame.draw.rect(surface, (42, 30, 21), door, 2)

        table = pygame.Rect(room.left + 84, room.centery + 36, 150, 58)
        pygame.draw.rect(surface, (118, 86, 56), table, border_radius=4)
        pygame.draw.rect(surface, (82, 60, 40), table, 2, border_radius=4)
        for lx, ly in (
            (table.left + 8, table.bottom - 6),
            (table.right - 14, table.bottom - 6),
            (table.left + 8, table.top + 4),
            (table.right - 14, table.top + 4),
        ):
            pygame.draw.rect(surface, (92, 66, 44), (lx, ly, 6, 18))
        pygame.draw.circle(surface, (214, 214, 206), (table.centerx - 28, table.centery), 8)
        pygame.draw.rect(surface, (138, 102, 66), (table.centerx + 8, table.centery - 8, 24, 12), border_radius=2)

        for sx, sy in ((table.left - 24, table.centery + 16), (table.right + 6, table.centery + 16)):
            pygame.draw.rect(surface, (106, 76, 50), (sx, sy, 18, 12), border_radius=3)
            pygame.draw.rect(surface, (78, 56, 38), (sx + 2, sy + 10, 3, 10))
            pygame.draw.rect(surface, (78, 56, 38), (sx + 13, sy + 10, 3, 10))

        bed = pygame.Rect(room.right - 248, room.centery + 20, 172, 82)
        pygame.draw.rect(surface, (170, 56, 62), bed, border_radius=5)
        pygame.draw.rect(surface, (224, 214, 206), (bed.x + 12, bed.y + 12, 60, 28), border_radius=4)
        pygame.draw.rect(surface, (98, 30, 36), bed, 2, border_radius=5)
        pygame.draw.rect(surface, (88, 62, 42), (bed.left - 8, bed.top + 4, 8, bed.height - 8), border_radius=2)

        shelf = pygame.Rect(room.right - 190, room.top + 102, 132, 94)
        pygame.draw.rect(surface, (96, 68, 46), shelf, border_radius=4)
        pygame.draw.rect(surface, (64, 46, 32), shelf, 2, border_radius=4)
        for sy in (shelf.top + 24, shelf.top + 50, shelf.top + 76):
            pygame.draw.line(surface, (58, 42, 30), (shelf.left + 6, sy), (shelf.right - 6, sy), 2)
        for i in range(9):
            bx = shelf.left + 10 + i * 12
            by = shelf.top + 8 + (i % 3) * 26
            pygame.draw.rect(surface, ((120 + (i * 14)) % 220, 70 + (i * 9) % 120, 60 + (i * 17) % 120), (bx, by, 8, 14), border_radius=1)

        counter = pygame.Rect(room.left + 52, room.top + 122, 160, 56)
        pygame.draw.rect(surface, (122, 90, 58), counter, border_radius=4)
        pygame.draw.rect(surface, (78, 58, 40), counter, 2, border_radius=4)
        pygame.draw.rect(surface, (140, 108, 74), (counter.left + 8, counter.top + 8, counter.width - 16, 14), border_radius=3)
        for px in (counter.left + 26, counter.left + 62, counter.left + 104):
            pygame.draw.circle(surface, (152, 96, 62), (px, counter.top + 30), 8)
            pygame.draw.rect(surface, (136, 86, 56), (px - 6, counter.top + 30, 12, 14), border_radius=3)

        chest = pygame.Rect(room.right - 150, room.bottom - 96, 88, 52)
        pygame.draw.ellipse(surface, (0, 0, 0, 70), (chest.left + 6, chest.bottom - 6, chest.width - 12, 10))
        pygame.draw.rect(surface, (116, 78, 38), chest, border_radius=7)
        lid = pygame.Rect(chest.left, chest.top - 8, chest.width, 26)
        pygame.draw.rect(surface, (132, 88, 44), lid, border_radius=9)
        pygame.draw.rect(surface, (86, 54, 28), chest, 2, border_radius=7)
        pygame.draw.rect(surface, (86, 54, 28), lid, 2, border_radius=9)
        for band_x in (chest.left + 18, chest.centerx - 4, chest.right - 26):
            pygame.draw.rect(surface, (168, 140, 72), (band_x, chest.top - 8, 8, chest.height + 8), border_radius=2)
        lock = pygame.Rect(chest.centerx - 8, chest.centery + 2, 16, 14)
        pygame.draw.rect(surface, (218, 188, 82), lock, border_radius=3)
        pygame.draw.rect(surface, (124, 96, 36), lock, 2, border_radius=3)
        pygame.draw.circle(surface, (102, 76, 24), (lock.centerx, lock.centery + 2), 2)

        for px, py in ((room.left + 286, room.top + 112), (room.right - 292, room.top + 112)):
            pygame.draw.rect(surface, (136, 88, 58), (px, py + 18, 22, 18), border_radius=3)
            pygame.draw.circle(surface, (54, 134, 70), (px + 11, py + 12), 11)
            pygame.draw.circle(surface, (62, 150, 84), (px + 5, py + 14), 8)
            pygame.draw.circle(surface, (62, 150, 84), (px + 17, py + 14), 8)

        sprite = SpriteAssets.get().get_character(
            "player",
            self.interior_player_direction,
            self.interior_player_moving,
            self.interior_anim_timer,
            tile_size=44,
        )
        if sprite is not None:
            px = int(self.interior_player_x)
            py = int(self.interior_player_y)
            pygame.draw.ellipse(surface, (0, 0, 0, 75), (px - 10, py + 4, 20, 6))
            surface.blit(sprite, (px - sprite.get_width() // 2, py - sprite.get_height() + 8))

        name = self.interior_building_name.replace("_", " ").title() if self.interior_building_name else "House"
        title = self.prompt_font.render(f"{name} Interior", True, (238, 232, 212))
        hint = self.prompt_font.render("Press E to exit", True, (210, 198, 168))
        surface.blit(title, (room.left + 12, room.top - 26))
        surface.blit(hint, (room.right - hint.get_width() - 12, room.top - 26))
