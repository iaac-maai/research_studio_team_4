"""
ui.py - minimal Pygame widgets (buttons and sliders) for the control panel.
"""
import pygame

import config as cfg

C = cfg.COLORS


class Button:
    def __init__(self, rect, label, on_click, is_on=None, enabled=None, hint=""):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.is_on = is_on          # callable -> bool for toggle buttons
        self.enabled = enabled      # callable -> bool
        self.hint = hint            # keyboard shortcut shown on the button

    def _enabled(self):
        return self.enabled() if self.enabled else True

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            if self._enabled():
                self.on_click()
            return True
        return False

    def draw(self, surf, fonts, mouse):
        label = self.label() if callable(self.label) else self.label
        on = self.is_on() if self.is_on else False
        enabled = self._enabled()
        color = C["button_on"] if on else (C["button_hover"] if self.rect.collidepoint(mouse) else C["button"])
        pygame.draw.rect(surf, color, self.rect, border_radius=7)
        text_color = C["panel_text"] if enabled else C["panel_dim"]
        t = fonts["button"].render(label, True, text_color)
        surf.blit(t, t.get_rect(midleft=(self.rect.x + 10, self.rect.centery)))
        if self.hint:
            h = fonts["small"].render(self.hint, True, C["panel_dim"] if not on else C["panel_text"])
            surf.blit(h, h.get_rect(midright=(self.rect.right - 8, self.rect.centery)))


class Slider:
    def __init__(self, rect, label, lo, hi, get, set_, fmt="{:.2f}", integer=False, enabled=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.lo, self.hi = lo, hi
        self.get, self.set = get, set_
        self.fmt = fmt
        self.integer = integer
        self.enabled = enabled
        self.dragging = False

    @property
    def track(self):
        return pygame.Rect(self.rect.x, self.rect.bottom - 10, self.rect.w, 6)

    def _value_at(self, x):
        t = min(1.0, max(0.0, (x - self.track.x) / self.track.w))
        v = self.lo + t * (self.hi - self.lo)
        return int(round(v)) if self.integer else v

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit = self.rect.inflate(0, 6).collidepoint(event.pos)
            if hit and (self.enabled() if self.enabled else True):
                self.dragging = True
                self.set(self._value_at(event.pos[0]))
            return hit
        if event.type == pygame.MOUSEMOTION and self.dragging:
            self.set(self._value_at(event.pos[0]))
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            return True
        return False

    def draw(self, surf, fonts, mouse):
        enabled = self.enabled() if self.enabled else True
        v = self.get()
        surf.blit(fonts["small"].render(self.label, True, C["panel_text"] if enabled else C["panel_dim"]),
                  (self.rect.x, self.rect.y))
        val = fonts["small"].render(self.fmt.format(v), True, C["accent"] if enabled else C["panel_dim"])
        surf.blit(val, val.get_rect(topright=(self.rect.right, self.rect.y)))
        tr = self.track
        pygame.draw.rect(surf, C["button"], tr, border_radius=3)
        t = (v - self.lo) / (self.hi - self.lo)
        fill = tr.copy()
        fill.w = max(1, int(tr.w * t))
        pygame.draw.rect(surf, C["button_on"] if enabled else C["panel_dim"], fill, border_radius=3)
        pygame.draw.circle(surf, C["panel_text"], (tr.x + int(tr.w * t), tr.centery), 7)
