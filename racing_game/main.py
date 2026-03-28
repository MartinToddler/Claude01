"""
Racing Game AI — main entry point v3.

Keyboard shortcuts:
    SPACE       pause / resume
    R           restart current generation (keep brains, re-apply params)
    T           toggle sensor rays on best / selected car
    K           kill all AI cars → immediate next generation
    ↑↓←→        player car throttle / steer (when enabled)
    A / Z       player shift up / down (manual gearbox)
    Q           toggle player auto / manual gearbox
    ESC         quit
    CLICK       select AI car or player car for detailed telemetry
                (click selected again or empty area → deselect)
"""

import sys
import pygame
from config import SCREEN_W, SCREEN_H, FPS, TRACK_AREA_X, TRACK_AREA_W
from track import build_track
from agent import PopulationManager
from player import PlayerCar
from renderer import Renderer
from ui import LeftPanel, RightPanel, TooltipState


def _make_pop(track, params):
    return PopulationManager(track, params)


def _make_player(track, params):
    from track import get_start_pose
    pc = PlayerCar(params)
    x, y, h = get_start_pose(track)
    pc.reset(x, y, h, params)
    return pc


def main():
    pygame.init()
    pygame.display.set_caption("Racing AI — Neuroevolution v3")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()

    # ── Initial state ─────────────────────────────────────────────────────────
    left    = LeftPanel()
    right   = RightPanel()
    tooltip = TooltipState()
    renderer = Renderer(screen)

    params     = left.params
    track_name = left.track_name
    track      = build_track(track_name)
    pop        = _make_pop(track, params)
    player     = _make_player(track, params)

    paused           = False
    step_counter     = 0
    selected_idx     = -1       # AI agent index; -1 = none
    selected_player  = False    # player car selected

    player_enabled   = False
    show_track_lines = True

    while True:
        clock.tick(FPS)

        # ── Events ────────────────────────────────────────────────────────────
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
                    pop.kill_all()
                # Player car keys
                player.handle_keydown(event.key)

            # Click selection (track area only)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if TRACK_AREA_X <= mx < TRACK_AREA_X + TRACK_AREA_W:
                    ai_idx, hit_player = renderer.hit_test(mx, my, pop, player)
                    if hit_player:
                        selected_player = not selected_player
                        selected_idx    = -1
                    elif ai_idx >= 0:
                        if ai_idx == selected_idx:
                            selected_idx = -1
                        else:
                            selected_idx    = ai_idx
                            selected_player = False
                    else:
                        selected_idx    = -1
                        selected_player = False
                    tooltip.dismiss()

            # Left panel events
            ev = left.handle_event(event, tooltip)

            if ev["params_changed"]:
                params = left.params
                pop.update_params(params)
                if player.enabled:
                    player.update_params(params)
                step_counter = 0

            if ev["track_changed"]:
                track_name = left.track_name
                track = build_track(track_name)
                pop.change_track(track)
                renderer.prerender_track(track)
                if player.enabled:
                    from track import get_start_pose
                    x, y, h = get_start_pose(track)
                    player.reset(x, y, h, params)
                selected_idx = -1
                step_counter = 0

            if ev["kill_all"]:
                pop.kill_all()

            if ev["restart"]:
                pop = _make_pop(track, params)
                player = _make_player(track, params)
                player.enabled = player_enabled
                selected_idx   = -1
                selected_player = False
                step_counter   = 0

            if ev["toggle_player"]:
                player_enabled = not player_enabled
                player.enabled = player_enabled
                if player_enabled:
                    from track import get_start_pose
                    x, y, h = get_start_pose(track)
                    player.reset(x, y, h, params)

            if ev["toggle_lines"]:
                show_track_lines = not show_track_lines
                renderer.show_track_lines = show_track_lines

        # ── Simulation ────────────────────────────────────────────────────────
        if not paused:
            sim_speed  = left.sim_speed
            dt_phys    = 1.0 / FPS
            max_steps  = pop.max_laps * 3600   # 60 s/lap safety cap

            keys = pygame.key.get_pressed()   # snapshot for player

            for _ in range(sim_speed):
                alive = pop.step(dt_phys)
                step_counter += 1

                if player.enabled:
                    player.step(keys, track, step_counter * dt_phys, dt_phys)

                if alive == 0 or step_counter >= max_steps:
                    pop.breed()
                    step_counter = 0
                    selected_idx = -1
                    break

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill((14, 15, 22))
        renderer.selected_idx    = selected_idx
        renderer.show_track_lines = show_track_lines
        renderer.draw(track, pop, clock.get_fps(), player)
        left.draw(screen, player_enabled, show_track_lines)
        right.draw(screen, pop, step_counter / FPS,
                   selected_idx=selected_idx,
                   selected_player=selected_player,
                   player=player)
        tooltip.draw_overlay(screen)

        if paused:
            _draw_pause(screen)

        pygame.display.flip()


def _draw_pause(screen):
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 110))
    screen.blit(overlay, (0, 0))
    font = pygame.font.SysFont("monospace", 42, bold=True)
    txt  = font.render("PAUSED", True, (255, 255, 255))
    screen.blit(txt, (SCREEN_W // 2 - txt.get_width()  // 2,
                      SCREEN_H // 2 - txt.get_height() // 2))


if __name__ == "__main__":
    main()
