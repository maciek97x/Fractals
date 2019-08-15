#!/usr/bin/env python3
import pygame
import sys
from pygame.locals import *
import colorsys
import numpy as np
from numpy import angle, pi
from threading import Thread
from copy import deepcopy
import time
import cmath

window_width = 600
window_height = window_width

delta_color = .005
#scale = 1000000000
#middle = complex(.3502702, 0.09033)
scale = 100
middle = complex(0, 0)
d_step = 10

pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)
draw_surface = pygame.Surface((window_width, window_height))
pygame.display.set_caption('Mandelbrot')
window.fill((0, 0, 0))
font = pygame.font.SysFont('consolas', 16)

values = np.zeros((window_width*window_height), dtype=complex)
steps = np.ones((window_width*window_height))
def series(k, p, prev):
    if k == 0:
        return 0
    result = prev**2 + p
    # result = p**prev.real - complex(0, 1)*(p**prev.imag) # + cmath.sin(p)
    # result = prev**2 - p**prev + cmath.e**p
    # result = prev**3 + complex(-0.1, 0.79)# julia, start with p
    # result = prev**2 + complex(0.1, 0.63)# julia, start with p
    # result = (prev**prev + prev*p + prev**2 + p)/cmath.exp(p)
    # result = prev**prev + prev + p
    # result = prev**prev + p #start with p
    # result = prev**2 + complex(-0.73, 0.19)# julia, start with p
    # result = prev**3/(p-prev)**2 + prev*p + p + p**3
    # result = (prev - 1)**3/(p-1)**2
    # result = prev/p + prev**2 # start with p
    # result = prev**1.5 + p*prev**2 + p
    # result = prev**2 + cmath.sin(cmath.exp(p).real)
    
    return result 

def update():
    global draw_surface
    global scale
    global middle
    global delta_color
    global window_width
    global window_height
    global values
    global steps
    print('updating')
    while True:
        for im in range(window_height):
            for re in range(window_width):
                for _ in range(d_step):
                    if draw_surface.get_at((re, im)) == (0, 0, 0):
                        try:
                            c = middle + complex(re - window_width//2, im - window_height//2)/scale
                            prev = values[re*window_height + im]
                            k = steps[re*window_height + im]
                            c = series(k, c, prev)
                            values[re*window_height + im] = c
                            steps[re*window_height + im] = k + 1
                            #col = colorsys.hls_to_rgb((angle(c)%(2*pi))/(2*pi), 1 - 0.5**abs(c), 1.)
                            if abs(c) >= 1e10000:
                                draw_surface.set_at((re, im), tuple(map(lambda x: 255*x, colorsys.hsv_to_rgb((k*delta_color)%1., 1., 1.))))
                        except:
                            pass
        print(np.max(steps), end='')

Thread(target=update).start()

scale_delta = 1
time0 = time.clock()
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            if event.button == BUTTON_LEFT:
                scale_delta = 1.1
            if event.button == BUTTON_RIGHT:
                scale_delta = 1/1.1
        if event.type == MOUSEBUTTONUP:
            if event.button in (BUTTON_LEFT, BUTTON_RIGHT):
                scale_delta = 1
                
    delta_time = time.clock() - time0
    time0 = time.clock()
    print('\r {} {}'.format(scale, middle), end='')
    if scale_delta != 1:
        steps = np.ones((window_width*window_height))
        window.fill((0, 0, 0))
        scale *= scale_delta
    
    mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
    text = str(middle + complex(mouse_pos_x - window_width//2,\
                                mouse_pos_y - window_height//2)/scale)
    window.blit(draw_surface, (0, 0))
    window.blit(font.render(text, 1, (255, 255, 255)),\
                 (mouse_pos_x - 10, mouse_pos_y - 30))    
    pygame.display.update()