import pygame
from pygame.locals import *
import sys
from time import perf_counter
import numpy as np

import fractal

# window size
window_width = 1024
window_height = window_width

# initializing window
pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)
pygame.display.set_caption('Fractal')

resize = 1

f = fractal.Mandelbrot((window_width//resize, window_height//resize))#, p=complex(0.5, 0.5))
f.start_compute()

def terminate():
    pygame.quit()
    sys.exit()

clock = pygame.time.Clock()

time_0 = perf_counter()
s = 0
# main loop
while True:
    # handling events
    for event in pygame.event.get():
        if event.type == QUIT:
            f.stop_compute()
            terminate()
    
    if pygame.mouse.get_pressed()[0]:
        point = (pygame.mouse.get_pos()[0]//resize,
                 pygame.mouse.get_pos()[1]//resize)
        f.zoom_to_point(point, 2)
    
    elif pygame.mouse.get_pressed()[2]:
        point = (pygame.mouse.get_pos()[0]//resize,
                 pygame.mouse.get_pos()[1]//resize)
        f.zoom_to_point(point, 1./2)
    
    window.fill((0, 0, 0))
    window.blit(
        pygame.transform.scale(
            pygame.surfarray.make_surface(f.image),
            (window_width, window_height)),
        (0, 0))
    pygame.display.update()
    clock.tick(100)
    print(f'\r{f.mid}, {f.scale}', end='')
print()
f.stop_copmute()