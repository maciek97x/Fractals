import numpy as np
import numba as nb
import mpmath as mp
from threading import Thread
from time import perf_counter


class Fractal:
    __time_scale = perf_counter()
    def __init__(self, size, formula):
        self.__size = size
        self.__formula = formula
        self.__image = np.zeros((*size, 3))
        self.__mid = mp.mpc(0)
        self.__scale = .25
        self.__if_compute = False
    
    @property
    def mid(self):
        return self.__mid

    @property
    def scale(self):
        return self.__scale
    
    @property
    def image(self):
        return self.__image
    
    def start_compute(self):
        self.__if_compute = True
        Thread(target=self.__compute).start()
    
    def stop_compute(self):
        self.__if_compute = False
    
    @staticmethod
    #@nb.jit
    def __compute_step(z, r):
        for x in nb.prange(z.shape[0]):
            for y in nb.prange(z.shape[1]):
                val = 0
                n = 0
                while abs(val) < 1e4 and n < 1000:
                    val = val*val + z[x, y]
                    n += 1
                if n >= 1000:
                    r[x, y] = -1
                else:
                    r[x, y] = n
    '''

    @staticmethod
    @np.vectorize
    def __compute_step(z):
        val = 0
        n = 0
        while abs(val) < 1e4 and n < 1000:
            val = val*val + z
            n += 1
        if n >= 1000:
            return -1
        return n
    '''
    
    @staticmethod
    @nb.jit
    def __compute_image(steps_array, size):
        def step_to_color_1(pixel, step):
            if step == -1:
                pixel[:] =  np.zeros((3))
            else:
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
        
        def step_to_color_2(pixel, step):
            if step == -1:
                pixel[:] = np.zeros((3))
            else:
                pixel[:] = np.array([255*abs((step%256)/128 - 1),
                                     255*abs((step%256)/128 - 1),
                                     255])
        
        width, height = size
        image = np.empty((*size, 3))
        for x in nb.prange(width):
            for y in nb.prange(height):
                step_to_color_1(image[x,y,:], steps_array[x,y])
        return image
    
    def __complex_array(self):
        width, height = self.__size
        real = np.arange(-.5/self.__scale, .5/self.__scale, 1/(self.__scale*width), dtype=mp.mpc)
        imag = np.arange(-.5/self.__scale, .5/self.__scale, 1/(self.__scale*height), dtype=mp.mpc)
        real += self.__mid.real
        imag += self.__mid.imag
        
        grid = np.zeros(self.__size, dtype=mp.mpc)
        grid += real[:, None]
        grid += 1j*imag
        
        return grid
    
    def __compute(self):
        i = 0
        while self.__if_compute:
            time_0 = perf_counter()
            complex_array = self.__complex_array()
            time_1 = perf_counter()
            steps_array = np.zeros(complex_array.shape)
            Fractal.__compute_step(complex_array, steps_array)
            #steps_array = Fractal.__compute_step(complex_array)
            time_2 = perf_counter()
            self.__image = Fractal.__compute_image(steps_array, self.__size)
            time_3 = perf_counter()
            #print(f'\r{time_1 - time_0: 0.5f} {time_2 - time_1: 0.5f} {time_3 - time_2: 0.5f}', end='')
            i += 1
    
    def compute_one_step(self):
        complex_array = self.__complex_array()
        steps_array = np.zeros(complex_array.shape)
        Fractal.__compute_step(complex_array, steps_array)
        #steps_array = Fractal.__compute_step(complex_array)
        self.__image = Fractal.__compute_image(steps_array, self.__size)
    
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
    
    def zoom_to_point(self, point, factor, smooth=True):
        if not isinstance(point, complex):
            point = self.__pixel_to_complex(point)
        point = mp.mpc(point)
        delta_time = perf_counter() - Fractal.__time_scale
        Fractal.__time_scale = perf_counter()
        if delta_time < .2 or not smooth:
            if smooth:
                factor = pow(factor, delta_time)
            new_scale = self.__scale*factor
            new_mid = self.__mid + (self.__mid - point)*(1 - factor)/factor
            self.__scale = new_scale
            self.__mid = new_mid