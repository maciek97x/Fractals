import numpy as np
import numba as nb
from threading import Thread
import cv2
import pygame
import cmath

class Fractal(object):
    def __init__(self, size):
        self.__size = size
        self.__image = np.zeros((*size, 3))
        self.__mid = complex(0)
        self.__scale = .25
        
        self.__lc_mid = complex(0)
        self.__lc_scale = .25
        
        self.__force_compute = True
        self.__if_compute = False
    
    def __getattr__(self, name):
        if name == 'mid':
            return self.__mid
        if name == 'scale':
            return self.__scale
        if name == 'image':
            return self.__image
        if name == 'pg_surface':
            surf = pygame.surfarray.make_surface(self.__image)
            surf_size = surf.get_size()
            new_surf = pygame.Surface(surf_size)
            s = self.__scale/self.__lc_scale
            x, y = self.__complex_to_pixel(self.__lc_mid)
            x -= surf_size[0]*s/2
            y -= surf_size[1]*s/2
            surf = pygame.transform.scale(surf, tuple(map(lambda q: int(q*s), surf_size)))
            new_surf.blit(surf, (x, y))
            return new_surf            
    
    def start_compute(self):
        self.__if_compute = True
        Thread(target=self.__compute).start()
    
    def stop_compute(self):
        self.__if_compute = False
    
    @staticmethod
    @nb.vectorize('int32(complex128, uint8)', nopython=True)
    def _compute_step(z, m):
        if m == 0:
            return 0
        val = 0
        n = 0
        while abs(val) < 2:
            val = val*val + z
            n += 1
            if n > 2048:
                return -1
        return n
    
    @staticmethod
    @nb.guvectorize(['int32[:,:,:], uint8[:,:,:]'], '(n, k, l)->(n, k, l)', nopython=True)
    def __compute_image(step, pixel):
        for i in range(step.shape[0]):
            for j in range(step.shape[1]):
                if step[i,j,0] == -1:
                    pixel[i,j,0] = 0
                    pixel[i,j,1] = 0
                    pixel[i,j,2] = 0
                else:
                    pixel[i,j,0] = 255*abs((step[i,j,0]%64)/32 - 1)
                    pixel[i,j,1] = 255*abs((step[i,j,0]%64)/32 - 1)
                    pixel[i,j,2] = 255
    
    @staticmethod
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
    
    @staticmethod
    def step_to_color_2(pixel, step):
        if step == -1:
            pixel[:] = np.zeros((3))
        else:
            pixel[:] = np.array([255*abs((step%256)/128 - 1),
                                 255*abs((step%256)/128 - 1),
                                 255])
    
    def __complex_array(self, size):
        width, height = size
        real = np.arange(-.5/self.__scale, .5/self.__scale, 1/(self.__scale*width))
        imag = np.arange(-.5/self.__scale, .5/self.__scale, 1/(self.__scale*height))
        real += self.__mid.real
        imag += self.__mid.imag
        
        grid = np.zeros(size, dtype='complex128')
        grid += real[:, None]
        grid += 1j*imag
        
        return grid
    
    def __compute(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        factor = 16
        kernel_size = 3
        while self.__if_compute:
            if self.__mid != self.__lc_mid or self.__scale != self.__lc_scale or self.__force_compute:
                self.force_compute = False
                size = np.array(self.__size)
                size_0 = size//factor
                a = self.__complex_array(size_0)
                size_0 *= 2
                mask = np.ones(a.shape, dtype='uint8')
                s = self.__class__._compute_step(a, mask)
                s_0 = np.zeros((*s.shape, 3), dtype='int32')
                s_0[:,:,0] = s
                img = self.__compute_image(s_0)
                #prev_img = cv2.resize(img, tuple(size_0))
                prev_img = np.copy(img)
                while size_0[0] <= size[0]:
                    mask = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                    mask = cv2.filter2D(mask.astype('uint8'), -1, kernel)/8
                    mask = cv2.filter2D(mask, -1, np.ones((kernel_size, kernel_size)))
                    mask = cv2.resize(mask, tuple(size_0))
                    prev_img = cv2.resize(prev_img, tuple(size_0))
                    mask[mask[:] > 0] = 255
                    a = self.__complex_array(size_0)
                    s = self.__class__._compute_step(a, mask.astype('uint8'))
                    s_0 = np.zeros((*s.shape, 3), dtype='int32')
                    s_0[:,:,0] = s
                    img = self.__compute_image(s_0)
                    prev_img[mask > 0] = 0
                    img[mask < 1] = 0
                    img += prev_img
                    size_0 *= 2
                    prev_img = np.copy(img)
                    #prev_img = cv2.resize(img, tuple(size_0))
                self.__image = img
                #print(f'\r{time_1 - time_0: 0.5f} {time_2 - time_1: 0.5f} {time_3 - time_2: 0.5f}', end='')
                self.__lc_mid = self.__mid
                self.__lc_scale = self.__scale
    
    def compute_one_step(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        factor = 16
        kernel_size = 3
        size = np.array(self.__size)
        size_0 = size//factor
        a = self.__complex_array(size_0)
        size_0 *= 2
        mask = np.ones(a.shape, dtype='uint8')
        s = self.__class__._compute_step(a, mask)
        s_0 = np.zeros((*s.shape, 3), dtype='int32')
        s_0[:,:,0] = s
        img = self.__compute_image(s_0)
        #prev_img = cv2.resize(img, tuple(size_0))
        prev_img = np.copy(img)
        while size_0[0] <= size[0]:
            mask = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
            mask = cv2.filter2D(mask.astype('uint8'), -1, kernel)/8
            mask = cv2.filter2D(mask, -1, np.ones((kernel_size, kernel_size)))
            mask = cv2.resize(mask, tuple(size_0))
            prev_img = cv2.resize(prev_img, tuple(size_0))
            mask[mask[:] > 0] = 255
            a = self.__complex_array(size_0)
            s = self.__class__._compute_step(a, mask.astype('uint8'))
            s_0 = np.zeros((*s.shape, 3), dtype='int32')
            s_0[:,:,0] = s
            img = self.__compute_image(s_0)
            prev_img[mask > 0] = 0
            img[mask < 1] = 0
            img += prev_img
            size_0 *= 2
            prev_img = np.copy(img)
            #prev_img = cv2.resize(img, tuple(size_0))
        self.__image = img
        #print(f'\r{time_1 - time_0: 0.5f} {time_2 - time_1: 0.5f} {time_3 - time_2: 0.5f}', end='')
        self.__lc_mid = self.__mid
        self.__lc_scale = self.__scale
    
    def __complex_to_pixel(self, z):
        w, h = self.__size
        m = self.__mid
        s = self.__scale
        return (int(w/2 + w*(z.real - m.real)*s),
                int(h/2 + h*(z.imag - m.imag)*s))
    
    def __pixel_to_complex(self, p):
        w, h = self.__size
        m = self.__mid
        s = self.__scale
        return m + complex(-.5 + p[0]/w, -.5 + p[1]/h)/s
    
    def complex_at(self, p):
        return self.__pixel_to_complex(p)#, self.__complex_to_pixel(self.__pixel_to_complex(p))
    
    def zoom_to_point(self, point, factor):
        if not isinstance(point, complex):
            point = self.__pixel_to_complex(point)
        new_scale = self.__scale*factor
        new_mid = self.__mid + (self.__mid - point)*(1 - factor)/factor
        self.__scale = new_scale
        self.__mid = new_mid

class Mandelbrot(Fractal):
    def __init__(self, *args):
        super(Mandelbrot, self).__init__(*args)
    
    @staticmethod
    @nb.vectorize('int32(complex128, uint8)', nopython=True)
    def _compute_step(z, m):
        if m == 0:
            return 0
        val = 0
        n = 0
        while abs(val) < 2:
            val = val*val + z
            n += 1
            if n > 2048:
                return -1
        return n

class Julia(Fractal):
    __p = complex(0.279)
    def __init__(self, *args, p=complex(0, 0)):
        super(Julia, self).__init__(*args)
        Julia.__p = p
    
    def __setattr__(self, name, value):
        if name == 'p':
            Julia.__p = value
            self.__force_compute = True
        super(Julia, self).__setattr__(name, value)
    
    @staticmethod
    def _compute_step(z, mask):
        return Julia.__compute_step_julia(z, mask, Julia.__p)

    @staticmethod
    @nb.guvectorize(['complex128[:,:], uint8[:,:], complex128, int32[:,:]'], '(n,k),(n,k),()->(n,k)', nopython=True)
    def __compute_step_julia(z, mask, p, s):
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                val = z[i,j]
                n = 0
                while abs(val) < 2 and n < 1024:
                    val = val*val + p
                    n += 1
                if n >= 1024:
                    s[i,j] = -1
                else:
                    s[i,j] = n
    
class BurningShip(Fractal):
    def __init__(self, *args, p=complex(0, 0)):
        super(BurningShip, self).__init__(*args)
    
    @staticmethod
    @nb.vectorize('int32(complex128, uint8)', nopython=True)
    def _compute_step(z, m):
        if m == 0:
            return 0
        val = 0
        n = 0
        while abs(val) < 2 and n < 2048:
            val = (abs(val.real) + 1j*abs(val.imag))**2 + z
            n += 1
        if n >= 1000:
            return -1
        return n