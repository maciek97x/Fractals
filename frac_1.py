import pygame
from pygame.locals import *
import sys
from time import perf_counter
import numpy as np
from math import sin, cos, pi

import fractal

# window size
window_width = 512
window_height = window_width

# initializing window
pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)
pygame.display.set_caption('Fractal')

resize = 1

def error():
    print('\n    usage: ./frac_1.py <fractal_name> [<p.real> <p.imag> (for Jutia fractal)]\n    available names: mandelbrot, julia, burningship\n')    
    exit()

if len(sys.argv) < 2:
    error()
if sys.argv[1] == 'julia':
    if len(sys.argv) < 4:
        f = fractal.Julia((window_width//resize, window_height//resize), p=complex(0.7885))
    else:
        f = fractal.Julia((window_width//resize, window_height//resize), p=complex(float(sys.argv[2]), float(sys.argv[3])))
elif sys.argv[1] == 'mandelbrot':
    f = fractal.Mandelbrot((window_width//resize, window_height//resize))
elif sys.argv[1] == 'burningship':
    f = fractal.BurningShip((window_width//resize, window_height//resize))
else:
    error()
    
f.start_compute()

def draw_text(text, size, color, surface, position, outline=False):
    font = pygame.font.SysFont('consolas', size*3//4)
    if outline:
        p_x, p_y = position
        col = tuple([255 - c for c in color])
        textobj = font.render(text, 1, col)
        for x in (p_x - 1, p_x + 1):
            for y in (p_y - 1, p_y + 1):
                textrect = textobj.get_rect()
                textrect.bottomright = (x, y)
                surface.blit(textobj, textrect)
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.bottomright = position
    surface.blit(textobj, textrect)

def terminate():
    pygame.quit()
    sys.exit()

show_complex_value = True

clock = pygame.time.Clock()

delta_time = 0
time_0 = perf_counter()
s = 0

mandelbrot_show_julia = False
mandelbrot_julia_f = fractal.Julia((window_width//4, window_height//4))
mandelbrot_julia_surface = pygame.Surface((window_width//4, window_height//4))

julia_animate_arg = False
julia_animate_mod = False

p_arg = 0
p_mod = .7885
# main loop
while True:
    delta_time, time_0 = perf_counter() - time_0, perf_counter()
    # handling events
    for event in pygame.event.get():
        if event.type == QUIT:
            f.stop_compute()
            terminate()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                f.stop_compute()
                terminate()
            if event.key == K_v:
                show_complex_value = not show_complex_value
            if f.__class__.__name__ == 'Julia':                
                if event.key == K_j:
                    julia_animate_arg = not julia_animate_arg
                if event.key == K_k:
                    julia_animate_mod = not julia_animate_mod
            if f.__class__.__name__ == 'Mandelbrot':                
                if event.key == K_j:
                    mandelbrot_show_julia = not mandelbrot_show_julia
                
    if f.__class__.__name__ == 'Julia':
        if julia_animate_arg:
            p_arg += delta_time/8
            p_arg %= 2*pi
            
        if julia_animate_mod:
            p_mod += delta_time/16
            p_mod %= 2
        
        f.p = (1 - abs(p_mod - 1))*(cos(p_arg) + complex(0, 1)*sin(p_arg))
            
    if f.__class__.__name__ == 'Mandelbrot' and mandelbrot_show_julia:
        mandelbrot_julia_f.p = f.complex_at(pygame.mouse.get_pos())
        mandelbrot_julia_f.compute_one_step()
        mandelbrot_julia_surface.blit(mandelbrot_julia_f.pg_surface, (0, 0))
        pygame.draw.rect(mandelbrot_julia_surface, (0, 0, 0), (0, 0, window_width//4, window_height//4), 1)
        
    if pygame.mouse.get_pressed()[0]:
        point = (pygame.mouse.get_pos()[0]//resize,
                 pygame.mouse.get_pos()[1]//resize)
        f.zoom_to_point(point, 2**delta_time)
    
    elif pygame.mouse.get_pressed()[2]:
        point = (pygame.mouse.get_pos()[0]//resize,
                 pygame.mouse.get_pos()[1]//resize)
        f.zoom_to_point(point, (1./2)**delta_time)
    
    window.fill((0, 0, 0))
    window.blit(
        pygame.transform.scale(
            f.pg_surface,
            (window_width, window_height)),
        (0, 0))
    if f.__class__.__name__ == 'Mandelbrot' and mandelbrot_show_julia:
        window.blit(mandelbrot_julia_surface, pygame.mouse.get_pos())
    if show_complex_value:
        draw_text(f'{f.complex_at(pygame.mouse.get_pos())}', 16, (0, 0, 0), window, pygame.mouse.get_pos(), outline=True)
    if f.__class__.__name__ == 'Julia':
        draw_text(f'p={1 - abs(p_mod - 1):.3f}*e^{p_arg:.3f}i', 16, (0, 0, 0), window, (window_width, window_height), outline=True)
    pygame.display.update()
    clock.tick(30)
    print(f'\rfps: {1./delta_time:.0f} {f.mid:.8f}, {f.scale:.8f}', end='')
print()
f.stop_copmute()