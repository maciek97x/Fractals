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
pygame.display.set_caption('Fractals')

rect_surface = pygame.Surface((window_width, window_height)).convert()
draw_surface = pygame.Surface((window_width, window_height)).convert()

rect_surface.blit(draw_surface, (0, 0))

def pixel_to_complex(ptc_p, ptc_scale, ptc_mid):
    return ptc_mid + complex(ptc_p[0] - window_width//2,\
                             ptc_p[1] - window_height//2)/ptc_scale

def complex_to_pixel(ctp_z, ctp_scale, ctp_mid):
    return (int(window_width/2 + (ctp_z.real - ctp_mid.real)*ctp_scale),\
            int(window_height/2 + (ctp_z.imag - ctp_mid.imag)*ctp_scale))    

def scale_surf(ss_surf, ss_scaling, ss_x, ss_y, ss_scale, ss_mid):
    copy = ss_surf.copy()
    copy = pygame.transform.scale(copy, (tuple(map(lambda q: int(q*ss_scaling),\
                                                   copy.get_size()))))
    new_surf = pygame.Surface(ss_surf.get_size())
    print(copy.get_size(), end='')
    new_surf.blit(copy, (ss_x*(1 - ss_scaling), ss_y*(1 - ss_scaling)))
    
    new_scale = ss_scale * ss_scaling
    ss_mid_pixel = complex_to_pixel(ss_mid, ss_scale, ss_mid)
    new_mid_pixel = (ss_mid_pixel[0] + (ss_mid_pixel[0] - ss_x)*(1 - ss_scaling),\
                     ss_mid_pixel[1] + (ss_mid_pixel[1] - ss_y)*(1 - ss_scaling))
    new_mid = pixel_to_complex(new_mid_pixel, ss_scale, ss_mid)
    
    return new_surf, new_scale, new_mid

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

mid = complex(0)
one = complex(1)
zero = complex(0) 
im = complex(0, 1)
scale = 100
scaling = 0
clock = pygame.time.Clock()
time0 = time.perf_counter()
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                scaling = 2
            if event.button == 3:
                scaling = 1/2
        if event.type == MOUSEBUTTONUP:
            if event.button in (1, 3):
                scaling = 0
            
    delta_time = time.perf_counter() - time0
    time0 = time.perf_counter()
    mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
    if scaling != 0:
        rect_surface, scale, mid = scale_surf(rect_surface,\
                                              pow(scaling, delta_time),\
                                              mouse_pos_x,\
                                              mouse_pos_y,\
                                              scale,\
                                              mid)

    window.fill((0,0,0))
    window.blit(rect_surface, (0, 0))
    pygame.draw.circle(window, (255, 0, 0), complex_to_pixel(mid, scale, mid), 3)
    pygame.draw.circle(window, (255, 0, 0), complex_to_pixel(one, scale, mid), 3)
    pygame.draw.circle(window, (255, 0, 0), complex_to_pixel(zero, scale, mid), 3)
    pygame.draw.circle(window, (255, 0, 0), complex_to_pixel(im, scale, mid), 3)
    print('\r {} {} {}           '.format(mid, complex_to_pixel(mid, scale, mid), scale), end='')
    clock.tick(10)
    pygame.display.update()