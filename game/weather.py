"""Lightweight weather system with clear/cloudy/mist/rain states."""
import random
import pygame
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    WEATHER_CHANGE_INTERVAL_MIN,
    WEATHER_CHANGE_INTERVAL_MAX,
    WEATHER_ENABLED,
    NIGHT_FOG_ALPHA,
)


class WeatherSystem:
    def __init__(self):
        self.enabled = WEATHER_ENABLED
        self.state = "clear"
        self.timer = 0.0
        self.next_change = random.uniform(WEATHER_CHANGE_INTERVAL_MIN, WEATHER_CHANGE_INTERVAL_MAX)
        self.overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.rain = []
        self._seed_rain()

    def _seed_rain(self):
        self.rain = []
        for _ in range(120):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            sp = random.uniform(220.0, 360.0)
            self.rain.append([x, y, sp])

    def update(self, dt):
        if not self.enabled:
            return
        self.timer += dt
        if self.timer >= self.next_change:
            self.timer = 0.0
            self.next_change = random.uniform(WEATHER_CHANGE_INTERVAL_MIN, WEATHER_CHANGE_INTERVAL_MAX)
            self.state = random.choices(
                ["clear", "cloudy", "mist", "rain"],
                weights=[0.34, 0.30, 0.22, 0.14],
                k=1,
            )[0]
            if self.state == "rain":
                self._seed_rain()

        if self.state == "rain":
            for d in self.rain:
                d[1] += d[2] * dt
                d[0] -= 46.0 * dt
                if d[1] > SCREEN_HEIGHT + 8 or d[0] < -10:
                    d[0] = random.randint(0, SCREEN_WIDTH)
                    d[1] = random.randint(-120, -8)

    def render_overlay(self, surface, day_cycle):
        if not self.enabled:
            return

        self.overlay.fill((0, 0, 0, 0))
        phase = day_cycle.day_phase

        # Soft global weather tint
        if self.state == "cloudy":
            self.overlay.fill((28, 34, 42, 36 if phase == "day" else 22))
        elif self.state == "mist":
            self.overlay.fill((180, 190, 200, 30 if phase == "day" else 18))
        elif self.state == "rain":
            self.overlay.fill((30, 40, 52, 52 if phase == "day" else 34))

        # Night fog regardless of weather (very light).
        if phase == "night":
            fog = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            fog.fill((160, 170, 180, NIGHT_FOG_ALPHA))
            self.overlay.blit(fog, (0, 0))

        if self.state == "rain":
            for x, y, _ in self.rain:
                pygame.draw.line(self.overlay, (185, 210, 240, 130), (int(x), int(y)), (int(x + 3), int(y + 8)), 1)

        surface.blit(self.overlay, (0, 0))
