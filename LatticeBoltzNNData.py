import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Button

def distance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def is_inside_ellipse(x, y, center_x, center_y, semi_major, semi_minor, angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    x_rot = cos_angle * (x - center_x) + sin_angle * (y - center_y)
    y_rot = -sin_angle * (x - center_x) + cos_angle * (y - center_y)

    ellipse_eq = (x_rot**2 / semi_major**2) + (y_rot**2 / semi_minor**2)
    return ellipse_eq <= 1

class Obsticle():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.name = None
        self.object = None
    
class Circle(Obsticle):
    def __init__(self, Nx,Ny, circleX, circleY, radius):
        super().__init__(circleX,circleY)

        # circleX, circleY = self.Nx//4, self.Ny//2
        radius = 13
        circle = np.full((Ny,Nx),False)
        for y in range(Ny):
            for x in range(Nx):
                if distance(circleX,circleY,x,y) < radius: circle[y][x] = True
        self.object = circle
        self.name = "Circle"

class Ellipse(Obsticle):
    def __init__(self,Nx,Ny, ellipse_center_x, ellipse_center_y, semi_major_axis, semi_minor_axis, angle):
        super().__init__(ellipse_center_x,ellipse_center_y)

        # # Ellipse
        # ellipse_center_x, ellipse_center_y = self.Nx // 4, self.Ny // 2
        # semi_major_axis = 20
        # semi_minor_axis = 5
        # angle = np.pi / 7  # 45 degrees
        ellipse = np.full((Ny, Nx), False)
        for y in range(Ny):
            for x in range(Nx):
                if is_inside_ellipse(x, y, ellipse_center_x, ellipse_center_y, semi_major_axis, semi_minor_axis, angle):
                    ellipse[y][x] = True
        self.object = ellipse
        self.name = "Ellipse"

class Scene():
    def __init__(self):
        # Simulation parameters
        self.Nx = 400    # resolution x-dir
        self.Ny = 100    # resolution y-dir
        self.rho0 = 100  # average density
        self.tau = 0.6   # collision timescale
        self.Nt = 4000   # number of timesteps
        self.plotRealTime = True  # switch on for plotting as the simulation goes along

        # Lattice speeds / self.weights
        self.NL = 9
        self.idxs = np.arange(self.NL)
        self.cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
        self.cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
        self.weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) 

        # Initial Conditions
        self.F = np.ones((self.Ny, self.Nx, self.NL))  # * self.rho0 / NL
        self.F += 0.01 * np.random.randn(self.Ny, self.Nx, self.NL)  # Perturb initial state
        self.F[:, :, 3] += 2.3 #2 * (1 + 0.2 * np.cos(2 * np.pi * X / self.Nx * 4))

        self.objects = []

        self.rho = np.sum(self.F, 2)
        for i in self.idxs:
            self.F[:, :, i] *= self.rho0 / self.rho

        # Cylinder boundary
        circleX, circleY = self.Nx//4, self.Ny//2
        radius = 13
        self.objects.append(Circle(self.Nx,self.Ny,circleX,circleY,radius))
        
        # Ellipse
        ellipse_center_x, ellipse_center_y = self.Nx // 4, self.Ny // 2
        semi_major_axis = 20
        semi_minor_axis = 5
        angle = np.pi / 7  # 45 degrees
        self.objects.append(Ellipse(self.Nx,self.Ny, ellipse_center_x,ellipse_center_y,semi_major_axis, semi_minor_axis, angle))
        
    
    def runAll(self):
        for thing in self.objects:
            self.Simulate(thing)


    def Simulate(self, obsticle):
        # Prep figure
        objects = obsticle.object
        fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Initialize the plot with empty data
        vorticity_img = ax.imshow(np.zeros((self.Ny, self.Nx)), cmap='bwr', animated=True)
        velocity_img = ax.imshow(np.zeros((self.Ny, self.Nx)), cmap='viridis', animated=True)
        cylinder_img = ax.imshow(~objects, cmap='gray', alpha=0.3, animated=True)
        velocity_img.set_visible(False)
        vorticity_img.set_visible(True)

        def toggle_visibility(event):
            if vorticity_img.get_visible():
                vorticity_img.set_visible(False)
                velocity_img.set_visible(True)
            else:
                vorticity_img.set_visible(True)
                velocity_img.set_visible(False)
            fig.canvas.draw_idle()

        def update(it):
            """ Update function for animation """
            global data_storage
            if it%100 == 0: print(it)
            
            self.F[:, -1, [6,7,8]] = self.F[:, -2, [6,7,8]]
            self.F[:, 0, [2,3,4]] = self.F[:, 1, [2,3,4]]
            # Drift
            for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
                self.F[:, :, i] = np.roll(self.F[:, :, i], cx, axis=1)  # Shift by cx in x-direction
                self.F[:, :, i] = np.roll(self.F[:, :, i], cy, axis=0)  # Shift by cy in y-direction
            
            # Set reflective boundaries
            bndryF = self.F[objects, :]
            bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

            # Calculate fluid variables
            rho = np.sum(self.F, 2)
            ux = np.sum(self.F * self.cxs, 2) / rho
            uy = np.sum(self.F * self.cys, 2) / rho

            # Apply Collision
            Feq = np.zeros(self.F.shape)
            for i, cx, cy, w in zip(self.idxs, self.cxs, self.cys, self.weights):
                Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2)
            
            self.F += -(1.0 / self.tau) * (self.F - Feq)

            # Apply boundary 
            self.F[objects, :] = bndryF
            ux[objects] = 0
            uy[objects] = 0

            velocity = np.sqrt(ux**2+uy**2)
            velocity_img.set_array(velocity)
            velocity_img.set_clim(0, np.max(velocity)) 

            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[objects] = np.nan
            vorticity = np.ma.array(vorticity, mask=objects)
            
            vorticity_img.set_array(vorticity)
            vorticity_img.set_clim(-0.1, 0.1)

            data_storage['velocity'].append(velocity.data)
            data_storage['density'].append(rho)
            data_storage['boundary'].append(objects)
            
            return velocity_img, vorticity_img, cylinder_img

        # Create animation
        anim = FuncAnimation(fig, update, frames=range(self.Nt), interval=1, blit=True, repeat=False)

        ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(ax_button, 'Toggle Plot')
        button.on_clicked(toggle_visibility)

        plt.show()
        
        # anim.save(f'animation{obsticle.name}.gif', writer='PillowWriter', fps=120, dpi=220)

        
# Storage for the data
data_storage = {
    'velocity': [],
    'density': [],
    'boundary': [],  # Boundary condition is static, save it once
}

def save_data(filename="simulation_data.h5"):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('velocity', data=np.array(data_storage['velocity']))
        f.create_dataset('density', data=np.array(data_storage['density']))
        f.create_dataset('boundary', data=data_storage['boundary'])

scene = Scene()
scene.runAll()

#save_data('data.h5')