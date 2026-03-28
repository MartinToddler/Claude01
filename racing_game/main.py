"""
Racing Game AI — main entry point (v2).

Controls:
    SPACE   — pause / resume
    R       — restart current generation (keep brains, apply current params)
    T       — toggle sensor rays on best / selected car
    K       — kill all cars now → breed next generation immediately
    ESC     — quit
    CLICK   — select a car (shows individual telemetry on right panel)
              click empty area to deselect
"""

import sys
import pygame
from config import SCREEN_W, SCREEN_H, FPS, SIM_SPEED_MAX
from track import build_track
from agent import PopulationManager
from renderer import Renderer
from ui import LeftPanel, RightPanel


def main():
    pygame.init()
    pygame.display.set_caption("Racing AI — Neuroevolution")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()

    left_panel  = LeftPanel()
    right_panel = RightPanel()
    renderer    = Renderer(screen)

    params      = left_panel.params
    track_name  = left_panel.track_name
    track       = build_track(track_name)
    pop         = PopulationManager(track, params)

    paused       = False
    step_counter = 0   # physics frames in current generation

    while True:
        # Enforce FPS — dt based on real elapsed time
        clock.tick(FPS)

        # ── Events ────────────────────────────────────────────────────────────
        params_changed = track_changed = speed_changed = kill_all = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    pop.update_params(params)
                    step_counter = 0
                elif event.key == pygame.K_t:
                    renderer.show_rays = not renderer.show_rays
                elif event.key == pygame.K_k:
                    kill_all = True

            # Left panel click detection (track area is to the right of panel)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Only test inside the track area
                from config import TRACK_AREA_X, TRACK_AREA_W
                if TRACK_AREA_X <= mx < TRACK_AREA_X + TRACK_AREA_W:
                    hit = renderer.hit_test(mx, my, pop)
                    if hit != renderer.selected_idx:
                        renderer.selected_idx = hit
                    else:
                        renderer.selected_idx = -1   # toggle off

            pc, tc, sc, ka = left_panel.handle_event(event)
            if pc: params_changed = True
            if tc: track_changed  = True
            if sc: speed_changed  = True
            if ka: kill_all       = True

        # ── Apply changes ─────────────────────────────────────────────────────
        if track_changed:
            track_name = left_panel.track_name
            track      = build_track(track_name)
            pop.change_track(track)
            renderer.prerender_track(track)
            renderer.selected_idx = -1
            step_counter = 0

        if params_changed:
            params = left_panel.params
            pop.update_params(params)
            step_counter = 0

        if kill_all:
            pop.kill_all()

        # ── Simulation (multi-step per render frame) ──────────────────────────
        if not paused:
            sim_speed = left_panel.sim_speed   # integer 1–100
            dt_phys   = 1.0 / FPS              # physics timestep

            for _ in range(sim_speed):
                alive = pop.step(dt_phys)
                step_counter += 1

                max_steps = pop.max_laps * 3600   # 60 s/lap × max_laps safety cap
                if alive == 0 or step_counter >= max_steps:
                    pop.breed()
                    step_counter = 0
                    renderer.selected_idx = -1   # clear selection on new gen
                    break   # render once before next gen starts

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill((14, 15, 22))
        renderer.draw(track, pop, clock.get_fps())
        left_panel.draw(screen)
        right_panel.draw(screen, pop,
                         step_counter / FPS,
                         selected_idx=renderer.selected_idx)

        if paused:
            _draw_pause(screen)

        pygame.display.flip()


def _draw_pause(screen: pygame.Surface):
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 110))
    screen.blit(overlay, (0, 0))
    font = pygame.font.SysFont("monospace", 42, bold=True)
    txt  = font.render("PAUSED", True, (255, 255, 255))
    screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2,
                      SCREEN_H // 2 - txt.get_height() // 2))


if __name__ == "__main__":
    main()
