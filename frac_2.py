import pygame
from pygame.locals import *
import sys
from time import perf_counter
import numpy as np
import cv2
from threading import Thread

import fractal

if '--help' in sys.argv:
    print('\nusage:'+\
          '\npython frac_2.py fps=30 length=120 zoom_per_sec=2 point=0.36024_-0.64131 output=render\n')
    exit()

kwargs = dict([])
for arg in sys.argv[1:]:
    if arg.count('=') == 1:
        k, v = arg.split('=')
        kwargs[k] = v

fps = 30
length = 20
resize = 1
zoom_per_sec = 4
point = complex(0.36024044343, -0.64131306106)
output = 'render'

# window size
window_width = 1024
window_height = window_width

for k, v in kwargs.items():
    if k == 'fps':
        try:
            fps = int(v)
        except:
            print('\nfps should be integer\n')
            exit()
    elif k == 'length':
        try:
            length = int(v)
        except:
            print('\nlength should be integer\n')
            exit()
    elif k == 'zoom_per_sec':
        try:
            zoom_per_sec = float(v)
        except:
            print('\nzoom_per_sec should be float\n')
            exit()
    elif k == 'point':
        try:
            real, imag = v.split('_')
            point = complex(float(real), float(imag))
        except:
            print('\npoint should be complex number in form real_imag\n')
            exit()
    elif k == 'output':
        output = v

# initializing window
pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)
pygame.display.set_caption('Fractal')

f = fractal.Mandelbrot((window_width//resize, window_height//resize))

def terminate():
    pygame.quit()
    sys.exit()
    exit()

out = cv2.VideoWriter(f'rendered/{output}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (window_width//resize, window_height//resize))

if_compute = True
def compute_frames():
    global if_compute
    global fps
    global length
    global zoom_per_sec
    global point
    frame = 0
    time_0 = perf_counter()
    while if_compute and frame/fps < length:
        f.zoom_to_point(point, pow(zoom_per_sec, 1./fps), smooth=False)
        f.compute_one_step()
        out.write(f.image[:,:,::-1].transpose(1, 0, 2).astype('uint8'))
        frame += 1
        eta = (perf_counter() - time_0)*(fps*length - frame)/frame
        print('\r[' +\
              '='*((30*(frame + 1))//(fps*length)) +\
              ' '*(30 - (30*(frame + 1))//(fps*length)) +\
              '] - ' +\
              f'{frame}/{length*fps} - ' +\
              f'{(perf_counter() - time_0)/frame:.2f} sec/frame - '+\
              f'ETA: {int(eta/3600):02d}:{int((eta%3600)/60):02d}:{int(eta%60):02d} ', end='')
    out.release()
    terminate()

Thread(target=compute_frames).start()

clock = pygame.time.Clock()

time_0 = perf_counter()
s = 0
# main loop
while True:
    # handling events
    for event in pygame.event.get():
        if event.type == QUIT:
            if_compute = False
            terminate()
    
    window.fill((0, 0, 0))
    window.blit(
        pygame.transform.scale(
            pygame.surfarray.make_surface(f.image),
            (window_width, window_height)),
        (0, 0))
    pygame.display.update()
    clock.tick(60)
print()
f.stop_copmute()