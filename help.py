import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

class Fluid:
    def __init__(self, density, maxWidth, maxHeight, h):
        self.maxWidth = maxWidth + 2
        self.maxHeight = maxHeight + 2
        self.numCells = self.maxWidth * self.maxHeight
        self.u = [ [0.0]*self.maxWidth for i in range(self.maxHeight)]
        self.v = [ [0.0]*self.maxWidth for i in range(self.maxHeight)]
        self.newU = [ [0.0]*self.maxWidth for i in range(self.maxHeight)]
        self.newV = [ [0.0]*self.maxWidth for i in range(self.maxHeight)]
        self.p = [ [0.0]*self.maxWidth for i in range(self.maxHeight)]
        self.s = [ [0.0]*self.maxWidth for i in range(self.maxHeight)]
        self.smoke = [ [1.0]*self.maxWidth for i in range(self.maxHeight)]
        self.newSmoke = [ [0.0]*self.maxWidth for i in range(self.maxHeight)]

        self.density = density
        self.h = h

    def integrate(self, dt, gravity):
        for i in range(1, self.maxWidth-1):
            for j in range(1, self.maxHeight-1):
                if self.s[i][j] != 0.0 and self.s[i][j-1] != 0.0:
                    self.v[i][j] += gravity * dt

    def solveIncompressibility(self, numIters, overRelaxation, dt):
        
        pressureChange = self.density * self.h / dt

        for _ in range(numIters):
            for i in range(1, self.maxWidth - 1):
                for j in range(1, self.maxHeight - 1):
                    if self.s[i][j] == 0.0: continue

                    sx0, sx1 = self.s[i-1][j], self.s[i+1][j]
                    sy0, sy1 = self.s[i][j-1], self.s[i][j+1]
                    s = sx0 + sx1 + sy0 + sy1

                    divergence = -self.u[i][j] + self.u[i+1][j] - self.v[i][j] + self.v[i][j+1]

                    p = overRelaxation * (-divergence / s)
                    self.p[i][j] += pressureChange * p

                    self.u[i][j] -= sx0 * p
                    self.u[i+1][j] += sx1 * p
                    self.v[i][j] -= sy0 * p
                    self.v[i][j+1] += sy1 * p



    def extrapolate(self):
        for i in range(self.maxHeight):
            self.u[i][0] = self.u[i][1]
            self.u[i][-1] = self.u[i][-2]

        for i in range(self.maxWidth):
            self.v[0][i] = self.v[1][i]
            self.v[-1][i] = self.v[-2][i]


    def sampleField(self, x, y, field):
        h = self.h
        h1 = 1.0 / h
        h2 = h * 0.5

        x = np.clip(x, h, self.maxWidth * h, out=None)
        y = np.clip(y, h, self.maxHeight * h, out=None)

        dx, dy = 0.0, 0.0

        if field == "u":
            f = self.u
            dy = h2
        elif field == "v": 
            f = self.v
            dx = h2
        elif field == "s":
            f = self.s
            dx, dy = h2, h2
        
        x0 = min(int((x - dx) * h1), self.maxWidth - 1)
        tx = ((x - dx) - x0 * h) * h1
        x1 = min(x0 + 1, self.maxWidth - 1)

        y0 = min(int((y - dy) * h1), self.maxHeight - 1)
        ty = ((y - dy) - y0 * h) * h1
        y1 = min(y0 + 1, self.maxHeight - 1)

        sx = 1.0 - tx
        sy = 1.0 - ty

        val = (sx * sy * f[x0][y0] +
            tx * sy * f[x1][y0] +
            tx * ty * f[x1][y1] +
            sx * ty * f[x0][y1])
    
        return val


    def avgU(self, i, j):
        u = self.u
        uVel = (u[i][j-1] + u[i][j] + u[i+1][j-1] + u[i][j+1]) * 0.25
        return uVel

    def avgV(self, i, j):
        v = self.v
        vVel = (v[i-1][j] + v[i][j] + v[i-1][j+1] + v[i][j+1]) * 0.25
        return vVel

    def advectVel(self, dt):
        self.newU[:] = [[val for val in row] for row in self.u]
        self.newV[:] = [[val for val in row] for row in self.v]

        h = self.h
        h2 = h * 0.5
        
        for i in range(self.maxWidth):
            for j in range(self.maxHeight):
                if self.s[i][j] != 0.0 and self.s[i-1][j] != 0.0 and j < self.maxHeight - 1:
                    x = i * h
                    y = j * h + h2
                    u = self.u[i][j]
                    v = self.avgV(i, j)

                    x = x - dt * u
                    y = y - dt * v
                    u =  self.sampleField(x,y, "u")
                    self.newU[i][j] = u
                
                if self.s[i][j] != 0.0 and self.s[i][j-1] != 0 and i < self.maxHeight - 1:
                    x = i * h + h2
                    y = j * h
                    u = self.avgU(i, j)
                    v = self.v[i][j]

                    x = x - dt * u
                    y = y - dt * v
                    u =  self.sampleField(x,y, "v")
                    self.newV[i][j] = v

        self.u[:] = [[val for val in row] for row in self.newU]
        self.v[:] = [[val for val in row] for row in self.newV]
        

    def advectSmoke(self, dt):
        self.newSmoke[:] = [[val for val in row] for row in self.smoke]
        h = self.h
        h2 = h * 0.5
        
        for i in range(self.maxWidth):
            for j in range(self.maxHeight):
                if self.s[i][j] != 0.0:
                    x = i * h + h2
                    y = j * h + h2
                    u = (self.u[i][j] + self.u[i+1][j]) * 0.5
                    v = (self.v[i][j] + self.v[i][j+1]) * 0.5

                    x -= dt * u
                    y -= dt * v
                    self.newSmoke[i][j] = self.sampleField(x,y, "s")

        # for i in range(len(self.smoke)):
        #     for j in range(len(self.smoke[i])):
        #         if self.newSmoke[i][j] != self.smoke[i][j]:
        #             print(i , j)

        self.smoke[:] = [[val for val in row] for row in self.newSmoke]

    def simulate(self, dt, gravity, numIters):
        self.integrate(dt, gravity)
        self.solveIncompressibility(numIters, 1.9, dt)
        self.extrapolate()
        self.advectVel(dt)
        self.advectSmoke(dt)

class Scene:
    def __init__(self):
        self.gravity = 0
        self.dt = 1.0 / 120.0
        self.numIters = 40
        self.frameNr = 0
        self.overRelaxation = 1.9
        self.obstacleX = 0.0
        self.obstacleY = 0.0
        self.obstacleRadius = 0.15
        self.paused = False
        self.sceneNr = 0
        self.showObstacle = False
        self.showStreamlines = False
        self.showVelocities = False
        self.showPressure = False
        self.showSmoke = True
        self.fluid = None

scene = Scene()


simHeight = 1.1;	
cScale = 200 / simHeight;
simWidth = 200 / cScale;
res = 100
domainHeight = 1.0
domainWidth = domainHeight / simHeight * simWidth
h = domainHeight / res;

numX = int(domainWidth / h);
numY = int(domainHeight / h);

density = 1000.0;

scene.fluid = Fluid(density, numX, numY, h)

inVel = 2.0
for i in range(numX):
    for j in range(numY):
        s = 1.0
        if i == 0 or j == 0 or j == numY-1:
            s = 0.0
        scene.fluid.s[i][j] = s

        if (i == 1):
            scene.fluid.u[i][j] = inVel
        

# Create figure and axis
fig, ax = plt.subplots()
smoke_img = ax.imshow(scene.fluid.smoke, origin='lower', cmap='gray', norm=mcolors.Normalize(vmin=0, vmax=1))

def update(frame):
    # Add continuous smoke at the center
    
    # Simulate the fluid dynamics
    scene.fluid.simulate(scene.dt, scene.gravity, scene.numIters)
    
    # Update the image with the new smoke values
    print("next")
    for row in scene.fluid.u:
        print(row[1])
    for row in scene.fluid.smoke:
        row[2] = 0.0
        

    smoke_img.set_data(scene.fluid.smoke)
    return smoke_img,

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

#plt.show()