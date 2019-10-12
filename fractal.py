import numpy as np
import numba as nb
from threading import Thread
from time import perf_counter


class Fractal:
    __time_scale = perf_counter()
    def __init__(self, size, formula):
        self.__size = size
        self.__formula = formula
        self.__image = np.zeros((*size, 3))
        self.__mid = complex(0)
        self.__scale = .25
        self.__if_compute = False
    
    @property
    def image(self):
        return self.__image
    
    def start_compute(self):
        self.__if_compute = True
        Thread(target=self.__compute).start()
    
    def stop_compute(self):
        self.__if_compute = False
    
    @staticmethod
    @nb.vectorize('int32(complex128)')
    def __compute_step(z):
        val = 0
        n = 0
        while abs(val) < 1e4 and n < 1000:
            val = val**2 + z
            n += 1
        if n >= 1000:
            return -1
        return n
    
    @staticmethod
    @nb.jit
    def __step_to_color(pixel, step):
        if step == -1:
            return np.zeros((3))
        h = (step*0.01)%1
        i = int(h*6.)
        f = (h*6.)-i
        q = int(255*(1.-f))
        t = int(255*f)
        i %= 6
        if i == 0:
            pixel[:] = np.array([255, t, 0])
        if i == 1:
            pixel[:] = np.array([q, 255, 0])
        if i == 2:
            pixel[:] = np.array([0, 255, t])
        if i == 3:
            pixel[:] = np.array([0, q, 255])
        if i == 4:
            pixel[:] = np.array([t, 0, 255])
        if i == 5:
            pixel[:] = np.array([255, 0, q])
    
    @staticmethod
    @nb.jit
    def __compute_image(steps_array, size):
        width, height = size
        image = np.zeros((*size, 3))
        for x in nb.prange(width):
            for y in nb.prange(height):
                Fractal.__step_to_color(image[x,y,:], steps_array[x,y])
        return image
    
    def __complex_array(self):
        width, height = self.__size
        real = np.arange(-.5/self.__scale, .5/self.__scale, 1/(self.__scale*width))
        imag = np.arange(-.5/self.__scale, .5/self.__scale, 1/(self.__scale*height))
        real += self.__mid.real
        imag += self.__mid.imag
        
        grid = np.zeros(self.__size, dtype='complex128')
        grid += real[:, None]
        grid += 1j*imag
        
        return grid
    
    def __compute(self):
        i = 0
        while self.__if_compute:
            print(f'\r {self.__mid} {self.__scale} computing' + '.'*(i%10) + ' ', end = '')
            complex_array = self.__complex_array()
            steps_array = Fractal.__compute_step(complex_array)
            self.__image = Fractal.__compute_image(steps_array, self.__size)
            i += 1
    
    def __complex_to_pixel(self, z):
        w, h = self.__size
        m = self.__mid
        s = self.__scale
        return (int(w/2 + (z.real - m.real)/s),
                int(h/2 + (z.imag - m.imag)/s))
    
    def __pixel_to_complex(self, p):
        w, h = self.__size
        m = self.__mid
        s = self.__scale
        return m + complex(-.5 + p[0]/w, -.5 + p[1]/h)/s
        #return m + complex(p[0] - w//2, p[1] - h//2)*s
    
    def zoom_to_point(self, point, factor):
        delta_time = perf_counter() - Fractal.__time_scale
        Fractal.__time_scale = perf_counter()
        if delta_time < .2:
            factor = pow(factor, delta_time)
            
            new_scale = self.__scale*factor
            new_mid = self.__mid + (self.__mid - self.__pixel_to_complex(point))*(1 - factor)/factor
            print()
            print(f'{self.__mid.real: .3f}_{self.__mid.imag: .3f}i' +\
                  f'  {new_mid.real: .3f}_{new_mid.imag: .3f}i' +\
                  f'  {self.__pixel_to_complex(point).real: .3f}_{self.__pixel_to_complex(point).imag: .3f}i' +\
                  f'  {point}')
            print()
            self.__scale = new_scale
            self.__mid = new_mid