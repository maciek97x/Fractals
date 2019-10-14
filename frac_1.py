import pygame
from pygame.locals import *
import sys
from time import perf_counter
import numpy as np

from fractal_v2 import Fractal

# window size
window_width = 512
window_height = 512

# initializing window
pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)
pygame.display.set_caption('Fractal')

resize = 16

f = Fractal((window_width//resize, window_height//resize), lambda v, z: v**2 + z)
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