"""
Day/Night cycle and time system.
"""
import pygame
import math
from config import DAY_LENGTH_SECONDS, HOURS_PER_DAY, SECONDS_PER_HOUR, SCREEN_WIDTH, SCREEN_HEIGHT


class DayCycle:
    """Manages in-game time and day/night lighting overlay."""

    def __init__(self):
        self.time_elapsed = 6.0 * SECONDS_PER_HOUR  # Start at 6 AM
        self.day_count = 1
        self.speed_multiplier = 1.0
        self._overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

    @property
    def current_hour(self):
        """Current hour (0-23) as a float."""
        total_seconds = self.time_elapsed % DAY_LENGTH_SECONDS
        return (total_seconds / SECONDS_PER_HOUR) % HOURS_PER_DAY

    @property
    def hour_int(self):
        return int(self.current_hour)

    @property
    def minute_int(self):
        frac = self.current_hour - int(self.current_hour)
        return int(frac * 60)

    @property
    def time_string(self):
        h = self.hour_int
        m = self.minute_int
        period = "AM" if h < 12 else "PM"
        display_h = h % 12
        if display_h == 0:
            display_h = 12
        return f"{display_h}:{m:02d} {period}"

    @property
    def day_phase(self):
        """Return current phase: dawn, day, dusk, night."""
        h = self.current_hour
        if 5 <= h < 7:
            return "dawn"
        elif 7 <= h < 17:
            return "day"
        elif 17 <= h < 19:
            return "dusk"
        else:
            return "night"

    def update(self, dt):
        """Update time by dt seconds (real time)."""
        self.time_elapsed += dt * self.speed_multiplier
        if self.time_elapsed >= DAY_LENGTH_SECONDS * self.day_count:
            # Check if we crossed midnight
            pass
        # Track day count
        self.day_count = int(self.time_elapsed / DAY_LENGTH_SECONDS) + 1

    def get_light_level(self):
        """Get ambient light level 0.0 (dark) to 1.0 (bright)."""
        h = self.current_hour
        if 7 <= h < 17:
            return 1.0
        elif 5 <= h < 7:
            return 0.3 + 0.7 * ((h - 5) / 2.0)
        elif 17 <= h < 19:
            return 1.0 - 0.7 * ((h - 17) / 2.0)
        else:
            return 0.3

    def get_overlay_color(self):
        """Get the tint color for the lighting overlay."""
        h = self.current_hour
        phase = self.day_phase

        if phase == "day":
            return (0, 0, 0, 0)  # No overlay
        elif phase == "dawn":
            t = (h - 5) / 2.0
            r = int(60 * (1 - t))
            g = int(40 * (1 - t))
            b = int(80 * (1 - t))
            a = int(100 * (1 - t))
            return (r, g, b, a)
        elif phase == "dusk":
            t = (h - 17) / 2.0
            r = int(40 + 30 * t)
            g = int(20 * t)
            b = int(40 + 50 * t)
            a = int(60 * t + 20)
            return (r, g, b, a)
        else:  # night
            return (20, 10, 50, 120)

    def render_overlay(self, surface):
        """Render the day/night lighting overlay."""
        color = self.get_overlay_color()
        if color[3] > 0:
            self._overlay.fill(color)
            surface.blit(self._overlay, (0, 0))

    def set_speed(self, multiplier):
        """Set time speed multiplier (for spectator mode)."""
        self.speed_multiplier = max(0.5, min(10.0, multiplier))

    def is_work_hours(self):
        return 7 <= self.current_hour < 17

    def is_sleep_hours(self):
        return self.current_hour >= 22 or self.current_hour < 5

    def is_evening(self):
        return 17 <= self.current_hour < 22
