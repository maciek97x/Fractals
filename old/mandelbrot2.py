#!/usr/bin/env python3
import pygame
import sys
from pygame.locals import *
import time
from math import sin, e
import numpy as np
from threading import Thread
import colorsys
import cmath

window_width = 600
window_height = window_width

delta_color = 0.001
mid = complex(0)
scale = 100
scaling = 0
stage = 0
exit = False

values = np.zeros((window_width, window_height), dtype=complex)
steps = (-1)*np.ones((window_width, window_height), dtype=int)

pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)
pygame.display.set_caption('Fractals')

rect_surface = pygame.Surface((window_width, window_height)).convert()

def pixel_to_complex(ptc_p):
    global mid
    global scale
    return mid + complex(ptc_p[0] - window_width//2,\
                         ptc_p[1] - window_height//2)/scale

def complex_to_pixel(ctp_z):
    global mid
    global scale
    return (int(window_width/2 + (ctp_z.real - mid.real)*scale),\
            int(window_height/2 + (ctp_z.imag - mid.imag)*scale))    

def scale_surf(ss_surf, ss_scaling, ss_x, ss_y):
    global mid
    global scale
    global window_width
    global window_height
    
    copy = ss_surf.copy()
    copy = pygame.transform.scale(copy, (tuple(map(lambda q: int(q*ss_scaling),\
                                                   copy.get_size()))))
    new_surf = pygame.Surface(ss_surf.get_size())
    print(copy.get_size(), end='')
    new_surf.blit(copy, (ss_x*(1 - ss_scaling), ss_y*(1 - ss_scaling)))
    
    new_scale = scale * ss_scaling
    ss_mid_pixel = complex_to_pixel(mid)
    new_mid_pixel = (ss_mid_pixel[0] + (ss_mid_pixel[0] - ss_x)*(1 - ss_scaling),\
                     ss_mid_pixel[1] + (ss_mid_pixel[1] - ss_y)*(1 - ss_scaling))
    new_mid = pixel_to_complex(new_mid_pixel)
    
    return new_surf, new_scale, new_mid


def series(n, x, y):
    global values
    global steps
    p = pixel_to_complex((x, y))
    if n == 0:
        values[x, y] = p
        steps[x, y] = 0
        return p
    if steps[x, y] == n - 1:
        prev = values[x, y]
    else:
        prev = series(n - 1, x, y)
    result = prev**2 + complex(-0.73, 0.19)
    values[x, y] = result
    steps[x, y] += 1
    return result

def update():
    global rect_surface
    global window_width
    global window_height
    global scaling
    global steps
    global stage
    global exit
    global delta_color
    while not exit:
        if stage == 0:
            for x in range(window_width):
                if stage != 0:
                    break
                for y in range(window_height):
                    if stage != 0:
                        break
                    if rect_surface.get_at((x, y)) == (0, 0, 0):
                        step = max(steps[x, y], 0)
                        val = series(step + 1, x, y)
                        if abs(val) > 1e5:
                            col = colorsys.hsv_to_rgb(step*delta_color, 1., 1.)
                            rect_surface.set_at((x, y), tuple(map(lambda q: int(q*255), col)))
        elif stage == 2:
            steps = (-1)*np.ones((window_width, window_height), dtype=int)
            copy = rect_surface.copy()
            for x in range(window_width):
                if stage != 2:
                    break
                for y in range(window_height):
                    if stage != 2:
                        break
                    for dx in range(-2, 3):
                        if stage != 2:
                            break
                        for dy in range(-2, 3):
                            if dx != 0 or dy != 0:
                                if stage != 2:
                                    break
                                if 0 <= x + dx < window_width and 0 <= y + dy < window_height:
                                    if rect_surface.get_at((x + dx, y + dy)) != rect_surface.get_at((x, y)):
                                        copy.set_at((x, y), (0, 0, 0))
            rect_surface = copy
            
            stage = 0

'''
for x in range(-10, 10, 2):
    for y in range(-10, 10, 2):
        c = complex(x, y)
        for s in range(1, 20):
            for xm in range(-3, 3):
                for ym in range(-3, 3):
                    m = complex(xm, ym)
                    print(c,\
                          complex_to_pixel(c, s/10, m),\
                          pixel_to_complex(complex_to_pixel(c, s/10, m), s/10, m))
                    
exit(1)
'''
Thread(target=update).start()
clock = pygame.time.Clock()
time0 = time.perf_counter()
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            exit = True
            pygame.quit()
            sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                scaling = 2
                stage = 1
            if event.button == 3:
                scaling = 1/2
                stage = 1
        if event.type == MOUSEBUTTONUP:
            if event.button in (1, 3):
                scaling = 0
                stage = 2
            
    delta_time = time.perf_counter() - time0
    time0 = time.perf_counter()
    mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
    if scaling != 0:
        rect_surface, scale, mid = scale_surf(rect_surface,\
                                              pow(scaling, delta_time),\
                                              mouse_pos_x,\
                                              mouse_pos_y)

    window.fill((0,0,0))
    window.blit(rect_surface, (0, 0))
    print('\r {} {} {}           '.format(mid, complex_to_pixel(mid), scale), end='')
    clock.tick(10)
    pygame.display.update()