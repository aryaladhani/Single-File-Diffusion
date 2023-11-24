import pygame
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Particle class to represent point particles in 1D


# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 600
NUM_PARTICLES = 5
FPS = 60
BACKGROUND_COLOR = (0, 0, 0)
COLLISION_DAMPING = 1.0
SIMULATION_TIME = 10.0  # Specify the duration of the simulation (in seconds)
TIME_STEP = 1/100000# Time step per frame
REALIZATIONS = 10
AMU = 1.66e-27
MASS = AMU
LENGTH_OF_CONTAINER = WIDTH
DT = 0.01
class Particle1D:
    def __init__(self, x, velocity):
        self.x = x
        self.velocity = velocity
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def move(self):
        self.x += self.velocity

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), 300), 5)  # Draw a small dot for point particle

def MB_CDF(v,m,T):
    """ Cumulative Distribution function of the Maxwell-Boltzmann speed distribution """
    kB = 1.38e-23
    a = np.sqrt(kB*T/m)
    return erf(v/(np.sqrt(2)*a)) - np.sqrt(2/np.pi)* v* np.exp(-v**2/(2*a**2))/a
from scipy.interpolate import interp1d as interp
def generate_velocities_position(temperature, no_of_particles,length_of_container):
    vs = np.arange(0, 3000, 0.1)
    cdf = MB_CDF(vs,MASS,temperature) # essentially y = f(x)
    inv_cdf = interp(cdf, vs)
    rand_nums = np.random.random(no_of_particles)
    speed = inv_cdf(rand_nums)
    random_generator = np.pi * np.random.randint(0, 100, speed.size)
    velocities = np.cos(random_generator)*speed
    positions = length_of_container * np.random.uniform(0, 1, no_of_particles)
    # print(positions)
    positions = np.sort(positions)
    # print
    return velocities,positions

# Create the screen
def simulation(no_of_particles, length_of_container, temperature, simulation_time, dt):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("1D Point Particle Collision Simulation")
    pygame.init()
    velocities, positions= generate_velocities_position(temperature,no_of_particles,length_of_container)
    particles = []
    for i in range (no_of_particles):
        particles.append(Particle1D(positions[i],velocities[i]))


# Main simulation loop
    clock = pygame.time.Clock()
    running = True

# Variables to track the middlemost particle and its initial position
    middle_particle = particles[no_of_particles // 2]
    initial_position = middle_particle.x
    simulation_start_time = pygame.time.get_ticks() / 1000.0
    middle_particle_positions_iterative =[]
    middle_particle_positions=[]
# List to store the position of the middle particle at different time steps
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill(BACKGROUND_COLOR)

        # Update and draw particles
        for particle in particles:
            particle.move()
            particle.draw(screen)

            # Check for collisions with walls
            if particle.x <= 0 or particle.x >= WIDTH:
                particle.velocity *= -1
        e = COLLISION_DAMPING
        # Check for particle collisions
        for i in range(no_of_particles-1):
            # for j in range(i + 1, no_of_particles):
                j = i+1
                if abs(particles[i].x - particles[j].x) <=2:  # Point particles, no radius
                    u1 = particles[i].velocity 
                    u2 = particles[j].velocity
                    v1 = ((1+e)*u2 + (1-e)*u1)/2
                    v2 = ((1+e)*u1+(1-e)*u2)/2
                    particles[i].velocity, particles[j].velocity = v1, v2

        # Update the display
        pygame.display.flip()
        clock.tick(FPS)

        # Calculate the elapsed time
        current_time = pygame.time.get_ticks() / 1000.0
        elapsed_time = current_time - simulation_start_time

        # Calculate the displacement of the middlemost particle
        square_displacement = abs (middle_particle.x - initial_position)

        # Store the position of the middle particle at this time step
        middle_particle_positions.append((elapsed_time, square_displacement))

        # Check if the simulation time has passed
        if elapsed_time >= SIMULATION_TIME:
            running = False
    pygame.quit()
    time = simulation_time
    def middle_particle_position_function(time):
        for t, x in middle_particle_positions:
            if t >= time:
                return x
    while (time>=0):
        middle_particle_positions_iterative.append(middle_particle_position_function(time-dt))
        time -=dt
    return middle_particle_positions_iterative
    
positions_realizations = []
for j in range(REALIZATIONS):
    cum_positions=[]
    positions=simulation(5,1000,0.00005,SIMULATION_TIME,0.000001)
    sum =0
    for position in positions:
        cum_positions.append(position)
    curr_pos_np = np.array(cum_positions)   
    curr_pos_np= curr_pos_np.reshape(-1,1)
    scaler.fit(curr_pos_np)
    curr_pos_np = scaler.transform(curr_pos_np)
    curr_pos_np = curr_pos_np.flatten()
    positions_realizations.append(curr_pos_np)
    if (len(curr_pos_np)!=len(np.arange(0,SIMULATION_TIME,0.000001))):
        curr_pos_np = np.array(curr_pos_np[1:])
    # plt.plot(np.arange(0,SIMULATION_TIME,0.001),curr_pos_np)
    # plt.show()
if (len(positions_realizations)!=len(np.arange(0,SIMULATION_TIME,0.000001))):

    positions_realizations = np.array(positions_realizations[1:])
else:
    positions_realizations = np.array(positions_realizations)
print(positions_realizations.shape)
positions_mean = np.mean(positions_realizations, axis =0)
print(positions_mean.shape)
plt.plot(np.arange(0,SIMULATION_TIME,0.000001),positions_mean)
plt.show()
