import numpy as np
import matplotlib.pyplot as plt
import heapq

# Constants for the simulations
L = [200.0, 100.0]  # Box lengths
N = [100, 50]       # Number of particles
temperature = 200.0
num_realizations = 500
time_step = 0.1     # Fixed time interval for recording positions
num_steps = 40
particle_indices = [49, 24]

# Function to initialize positions and velocities
def initialize(N, L, temperature):
    positions = np.linspace(0.2 * L, 0.8 * L, N)
    velocities = np.random.normal(loc=0.0, scale=np.sqrt(temperature), size=N)
    return positions, velocities

# Function to find the next collision time and partners
def next_collision(positions, velocities, N, L):
    event_queue = []
    for i in range(N):
        for j in range(i + 1, N):
            if velocities[i] != velocities[j]:
                dt = (positions[j] - positions[i]) / (velocities[i] - velocities[j])
                if dt > 0:
                    heapq.heappush(event_queue, (dt, i, j))
        # Wall collisions
        if velocities[i] > 0:
            heapq.heappush(event_queue, ((L - positions[i]) / velocities[i], i, 'right_wall'))
        elif velocities[i] < 0:
            heapq.heappush(event_queue, (positions[i] / abs(velocities[i]), i, 'left_wall'))
    return event_queue

# Function to update positions
def update_positions(positions, velocities, dt):
    return positions + velocities * dt

# Function to perform a single realization
def perform_realization(N, L, particle_index, num_steps, time_step):
    positions, velocities = initialize(N, L, temperature)
    initial_position = positions[particle_index]
    event_queue = next_collision(positions, velocities, N, L)
    msd = np.zeros(num_steps)
    current_time = 0

    for step in range(num_steps):
        
        next_event_time = event_queue[0][0] if event_queue else float('inf')
        
        while current_time < next_event_time and current_time < step * time_step:
            current_time += time_step

        if current_time >= next_event_time:
            dt, i, j = heapq.heappop(event_queue)
            positions = update_positions(positions, velocities, dt - current_time)
            if isinstance(j, str):  # Wall collision
                velocities[i] *= -1
            else:  # Particle collision
                velocities[i], velocities[j] = velocities[j], velocities[i]
            # Update event queue
            for new_event in next_collision(positions, velocities, N, L):
                heapq.heappush(event_queue, new_event)
            current_time = dt

        positions = update_positions(positions, velocities, time_step - (current_time % time_step))
        msd[step] = (positions[particle_index] - initial_position)**2

    return msd

# Perform simulations and plot results
plt.figure(figsize=(12, 6))
plt.title("Mean Square Displacement Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Mean Square Displacement (units^2)")

for sim in range(2):
    msd_avg = np.zeros(num_steps)
    for realization in range(num_realizations):
        print(f"currently conducting simulations for {realization} for {sim} particle")
        msd = perform_realization(N[sim], L[sim], particle_indices[sim], num_steps, time_step)
        msd_avg += msd
    msd_avg /= num_realizations
    plt.plot(np.arange(num_steps) * time_step, msd_avg, label=f"N={N[sim]}, L={L[sim]}, Particle {particle_indices[sim]+1}")

plt.legend()
plt.show()
