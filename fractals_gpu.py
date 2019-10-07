#!/usr/bin/env python3
import os

os.environ['NUMBAPRO_NVVM'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\nvvm\bin\nvvm64_33_0.dll'

os.environ['NUMBAPRO_LIBDEVICE'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\nvvm\libdevice'

from numba import cuda
import pygame
import sys
from pygame.locals import *
import time
from math import sin, e
import numpy as np
from threading import Thread
import colorsys
import cmath
import numba as nb
from itertools import product

window_width = 800
window_height = window_width

delta_color = 0.005
mid = complex(0)
scale = 100
scaling = 0
stage = 0
exit = False
max_iter = 1000

values = np.zeros((window_width, window_height), dtype=complex)
steps = (-1)*np.ones((window_width, window_height), dtype=int)


pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)
pygame.display.set_caption('Fractals')

rect_surface = pygame.Surface((window_width, window_height)).convert()
rect_pairs = np.array(product(range(window_width), range(window_height)))

temp_surface = pygame.Surface((window_width, window_height)).convert()

def pixel_to_complex(ptc_p):
    global mid
    global scale
    global window_width
    global window_height
    return mid + complex(ptc_p[0] - window_width//2,\
                         ptc_p[1] - window_height//2)/scale

def complex_to_pixel(ctp_z):
    global mid
    global scale
    global window_width
    global window_height
    return (int(window_width/2 + (ctp_z.real - mid.real)*scale),\
            int(window_height/2 + (ctp_z.imag - mid.imag)*scale))    

def scale_surf(ss_surf, ss_scaling, ss_x, ss_y):
    global mid
    global scale
    global window_width
    global window_height
    global temp_surface
    
    copy = ss_surf.copy()
    copy = pygame.transform.scale(copy, (tuple(map(lambda q: int(q*ss_scaling),\
                                                   copy.get_size()))))
    new_surf = pygame.Surface(ss_surf.get_size())
    new_surf.blit(copy, (ss_x*(1 - ss_scaling), ss_y*(1 - ss_scaling)))
     
    new_scale = scale * ss_scaling
    ss_mid_pixel = complex_to_pixel(mid)
    new_mid_pixel = (ss_mid_pixel[0] + (ss_mid_pixel[0] - ss_x)*(1 - ss_scaling)/ss_scaling,\
                     ss_mid_pixel[1] + (ss_mid_pixel[1] - ss_y)*(1 - ss_scaling)/ss_scaling)
    new_mid = pixel_to_complex(new_mid_pixel)
    return new_scale, new_mid, new_surf

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

for x in range(window_width):
    for y in range(window_height):
        values[x, y] = pixel_to_complex((x, y))
        
@nb.vectorize('int32(complex128)')
def calculate_step(cs_p):
    global delta_color
    global steps
    global max_iter
    val = 0
    n = 0
    while abs(val) < 1e4 and n < max_iter:
        val = val**2 + cs_p
        n += 1
    return n
'''
@nb.jit('UniTuple(float64[:,:], 3)(int32[:,:])')
def steps_to_surface(sts_step):
    global delta_color
    if sts_step < 10000:
        col = tuple(colorsys.hsv_to_rgb(sts_step*delta_color, 1., 1.))
    else:
        col = (0, 0, 0)
    return col
'''

@nb.vectorize('float64(int32)')
def steps_to_surface(sts_step):
    global delta_color
    global max_iter
    if sts_step >= max_iter:
        return 0.
    return sts_step

@nb.vectorize('int32(int32)')
def steps_to_gray(stg_step):
    global max_iter
    if stg_step >= max_iter:
        return 0
    stg_step ^= (stg_step >> 1)
    return stg_step

'''
@nb.jit
def steps_to_surface(sts_steps):
    global window_width
    global window_height
    global max_iter
    global delta_color
    sts_array = np.zeros((window_width, window_height))
    for x in range(window_width):
        for y in range(window_height):
            if sts_steps[x, y] < max_iter:
                sts_array[x, y] = sts_steps[x, y]
    return sts_array
'''
    
def update():
    global rect_surface
    global window_width
    global window_height
    global scaling
    global steps
    global stage
    global exit
    global delta_color
    global temp_surface
    global scale
    global mid
    global max_iter
    u_scale = 1
    temp_scale = 1
    temp_mid = complex(0)
    while not exit:
        if stage == 0:
            steps = calculate_step(values)
            #surf = steps_to_surface(steps)
            #rect_surface = pygame.surfarray.make_surface(255*surf)
            '''
            for x in range(window_width):
                for y in range(window_height):
                    if steps[x, y] < max_iter:
                        col = colorsys.hsv_to_rgb(steps[x, y]*delta_color, 1., 1.)
                    else:
                        col = (0, 0, 0)
                    rect_surface.set_at((x, y), tuple(map(lambda q: int(q*255), col)))
            '''
            surface_array = steps_to_surface(steps_to_gray(steps))
            rect_surface = pygame.surfarray.make_surface(surface_array)
            stage = -1
        elif stage == 1:
            temp_surface = rect_surface
            u_scale = 1
            stage = 2
            time0 = 0
        elif stage == 2:
            if time0 != 0:
                delta_time = time.perf_counter() - time0
                time0 = time.perf_counter()
                u_scale *= pow(scaling, delta_time)
                x, y = pygame.mouse.get_pos()
                temp_scale, temp_mid, rect_surface = scale_surf(temp_surface, u_scale, x, y)
            else:
                time0 = time.perf_counter()
        elif stage == 3:
            scale = temp_scale
            mid = temp_mid
            for x in range(window_width):
                for y in range(window_height):
                    values[x, y] = pixel_to_complex((x, y))
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

mandel_gpu = cuda.jit(device=True)(update)
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
                stage = 3
            
    window.fill((0,0,0))
    window.blit(rect_surface, (0, 0))
    print('\r {} {} {} {}           '.format(stage, mid, complex_to_pixel(mid), scale), end='')
    clock.tick(60)
    pygame.display.update()