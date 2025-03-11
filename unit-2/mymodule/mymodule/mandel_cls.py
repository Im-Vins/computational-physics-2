import matplotlib.pyplot as plt
import numpy as np

class MandelbrotSet:
    def __init__(self, nx, xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0, max_iter=10, take_log=False):
        """Initialize Mandelbrot set parameters."""
        self.nx = nx
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.max_iter = max_iter
        self.take_log = take_log
    
    def generate(self):
        """Generate the Mandelbrot set and return the iteration matrix."""
        x = np.linspace(self.xmin, self.xmax, self.nx)
        y = np.linspace(self.ymin, self.ymax, self.nx)
        xv, yv = np.meshgrid(x, y, indexing="ij")
        
        c = xv + 1j * yv
        z = np.zeros((self.nx, self.nx), dtype=np.complex128)
        m = np.zeros((self.nx, self.nx), dtype=np.int32)
        
        for i in range(self.max_iter):
            z = z**2 + c
            m[np.logical_and(np.abs(z) > 2, m == 0)] = i
        
        if self.take_log:
            m = np.log10(m + 1)
        
        return m
    
    def plot(self):
        """Plot the Mandelbrot set and return the figure object."""
        m = self.generate()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(np.transpose(m), origin="lower",
                       extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        fig.colorbar(im, ax=ax)
        
        return fig
