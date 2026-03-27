"""
Racing Game AI — main entry point.

Run:
    python main.py

Controls:
    R     — restart current generation (keep brains)
    T     — toggle sensor rays on best car
    SPACE — pause / resume
    ESC   — quit
"""

import sys
import pygame
from config import SCREEN_W, SCREEN_H, FPS, MAX_GEN_STEPS
from track import build_track, TRACK_NAMES
from agent import PopulationManager
from renderer import Renderer
from ui import LeftPanel, RightPanel


def main():
    pygame.init()
    pygame.display.set_caption("Racing AI — Neuroevolution")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()

    # ── Initial setup ─────────────────────────────────────────────────────────
    left_panel  = LeftPanel()
    right_panel = RightPanel()
    renderer    = Renderer(screen)

    params      = left_panel.params
    track_name  = left_panel.track_name
    track       = build_track(track_name)
    pop         = PopulationManager(track, params)

    paused       = False
    step_counter = 0       # frames in current generation

    while True:
        dt = clock.tick(FPS) / 1000.0   # seconds since last frame

        # ── Events ────────────────────────────────────────────────────────────
        params_changed = track_changed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    pop.update_params(params)   # restart gen, keep brains
                elif event.key == pygame.K_t:
                    renderer.show_rays = not renderer.show_rays

            pc, tc = left_panel.handle_event(event)
            if pc: params_changed = True
            if tc: track_changed  = True

        # Apply changes
        if track_changed:
            track_name = left_panel.track_name
            track      = build_track(track_name)
            pop.change_track(track)
            renderer.prerender_track(track)
            step_counter = 0

        if params_changed:
            params = left_panel.params
            pop.update_params(params)
            step_counter = 0

        # ── Simulation step ───────────────────────────────────────────────────
        if not paused:
            alive = pop.step(dt)
            step_counter += 1

            # Breed when all dead or time limit reached
            if alive == 0 or step_counter >= MAX_GEN_STEPS:
                pop.breed()
                step_counter = 0

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill((14, 15, 22))

        renderer.draw(track, pop, clock.get_fps())

        left_panel.draw(screen)
        right_panel.draw(screen, pop, step_counter / FPS)

        # Pause overlay
        if paused:
            _draw_pause(screen)

        pygame.display.flip()


def _draw_pause(screen: pygame.Surface):
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    screen.blit(overlay, (0, 0))
    font = pygame.font.SysFont("monospace", 36, bold=True)
    txt  = font.render("PAUSED", True, (255, 255, 255))
    screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2,
                      SCREEN_H // 2 - txt.get_height() // 2))


if __name__ == "__main__":
    main()
