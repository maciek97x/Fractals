#!/usr/bin/env python3
import pygame
import sys
from pygame.locals import *
import time
from math import sin, e

window_width = 600
window_height = window_width

pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)
pygame.display.set_caption('Scale Test')

rect_surface = pygame.Surface((400, 400)).convert()
draw_surface = pygame.Surface((400, 400)).convert()
for x in range(400):
    for y in range(400):
        draw_surface.set_at((x, y), tuple(map(lambda x: max(0, min(int(255 + 127*x), 255)),\
                                              (sin(x/50), sin(y/30)*sin(x/20), sin(x/30 + y/90)))))
rect_surface.blit(draw_surface, (0, 0))

def scale_surf(surf, s, x, y):
    copy = surf.copy()
    copy = pygame.transform.scale(copy, (tuple(map(lambda q: int(q*s),\
                                                   copy.get_size()))))
    new_surf = pygame.Surface(surf.get_size())
    print(copy.get_size(), end='')
    new_surf.blit(copy, (x*(1 - s), y*(1 - s)))
    return new_surf
    
scale = 0
clock = pygame.time.Clock()
time0 = time.perf_counter()
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            if event.button == BUTTON_LEFT:
                scale = 2
            if event.button == BUTTON_RIGHT:
                scale = 1/2
        if event.type == MOUSEBUTTONUP:
            if event.button in (BUTTON_LEFT, BUTTON_RIGHT):
                scale = 0
            
    delta_time = time.perf_counter() - time0
    time0 = time.perf_counter()
    mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
    if scale != 0:
        rect_surface = scale_surf(rect_surface,\
                                  pow(scale, delta_time),\
                                  mouse_pos_x - 100,\
                                  mouse_pos_y - 100)

    window.fill((0,0,0))
    window.blit(rect_surface, (100, 100))
    print('\r {}             '.format(scale), end='')
    clock.tick(10)
    pygame.display.update()