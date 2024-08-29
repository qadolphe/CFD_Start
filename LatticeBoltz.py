import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Button

"""
 
Philip Mocz (2020) Princeton University, @PMocz

Simulate flow past a cylinder for an isothermal fluid
"""

def distance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def toggle_visibility(event):
    if vorticity_img.get_visible():
        vorticity_img.set_visible(False)
        velocity_img.set_visible(True)
    else:
        vorticity_img.set_visible(True)
        velocity_img.set_visible(False)
    fig.canvas.draw_idle()

# Simulation parameters
Nx = 400    # resolution x-dir
Ny = 100    # resolution y-dir
rho0 = 100  # average density
tau = 0.6   # collision timescale
Nt = 40   # number of timesteps
plotRealTime = True  # switch on for plotting as the simulation goes along

# Lattice speeds / weights
NL = 9
idxs = np.arange(NL)
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) 

# Initial Conditions
F = np.ones((Ny, Nx, NL))  # * rho0 / NL
F += 0.01 * np.random.randn(Ny, Nx, NL)  # Perturb initial state
F[:, :, 3] += 2.3 #2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))

rho = np.sum(F, 2)
for i in idxs:
    F[:, :, i] *= rho0 / rho

# Cylinder boundary
circleX, circleY = Nx//4, Ny//2
radius = 13
objects = np.full((Ny,Nx),False)
for y in range(Ny):
    for x in range(Nx):
        if distance(circleX,circleY,x,y) < radius: objects[y][x] = True


# Prep figure
fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Initialize the plot with empty data
vorticity_img = ax.imshow(np.zeros((Ny, Nx)), cmap='bwr', animated=True)
velocity_img = ax.imshow(np.zeros((Ny, Nx)), cmap='viridis', animated=True)
cylinder_img = ax.imshow(~objects, cmap='gray', alpha=0.3, animated=True)

velocity_img.set_visible(False)
vorticity_img.set_visible(True)

def update(it):
    """ Update function for animation """
    global F
    print(it)
    
    F[:, -1, [6,7,8]] = F[:, -2, [6,7,8]]
    F[:, 0, [2,3,4]] = F[:, 1, [2,3,4]]
    # Drift
    for i, cx, cy in zip(idxs, cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)  # Shift by cx in x-direction
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)  # Shift by cy in y-direction
    
    # Set reflective boundaries
    bndryF = F[objects, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

    # Calculate fluid variables
    rho = np.sum(F, 2)
    ux = np.sum(F * cxs, 2) / rho
    uy = np.sum(F * cys, 2) / rho

    # Apply Collision
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2)
    
    F += -(1.0 / tau) * (F - Feq)

    # Apply boundary 
    F[objects, :] = bndryF
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
    
    return velocity_img, vorticity_img, cylinder_img

# Create animation
anim = FuncAnimation(fig, update, frames=range(Nt), interval=1, blit=True, repeat=False)

# ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(ax_button, 'Toggle Plot')
# button.on_clicked(toggle_visibility)

# plt.show()

FFwriter = FFMpegWriter(fps=30)
anim.save('animation.mp4', writer = FFwriter)
# anim.save('animation.gif', writer='ffmpeg')