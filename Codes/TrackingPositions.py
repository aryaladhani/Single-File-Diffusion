import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf

def MB_CDF(v,m,T):
    """ Cumulative Distribution function of the Maxwell-Boltzmann speed distribution """
    kB = 1.38e-23
    a = np.sqrt(kB*T/m)
    return erf(v/(np.sqrt(2)*a)) - np.sqrt(2/np.pi)* v* np.exp(-v**2/(2*a**2))/a
from scipy.interpolate import interp1d as interp

amu = 1.66e-27
mass = 85*amu
# create CDF

def generate_velocities_position(temperature, no_of_particles,length_of_container):
    vs = np.arange(0, 3000, 0.1)
    cdf = MB_CDF(vs,mass,temperature) # essentially y = f(x)
    inv_cdf = interp(cdf, vs)
    rand_nums = np.random.random(no_of_particles)
    speed = inv_cdf(rand_nums)
    random_generator = np.pi * np.random.randint(0, 100, speed.size)
    velocities = np.cos(random_generator)*speed
    # print(velocities.si)
    velocities.reshape(velocities.size,1)
    positions = length_of_container * np.random.uniform(0, 1, no_of_particles)
    positions.reshape(positions.size,1)
    positions_series = pd.DataFrame(positions)
    velocities_series =pd.DataFrame(velocities)
    data = pd.concat([positions_series,velocities_series],axis=1)
    data.columns =["Positions", "Velocities"]
    data.sort_values("Positions", inplace=True)
    return data

def change_positions(dataframe, dt):
    velocities = dataframe['Velocities']
    positions = dataframe['Positions']
    positions = positions + dt * velocities
    dataframe['Positions'] =positions
    return dataframe

def swap(array,i):
    temp = array.iloc[i]
    array.iloc[i]=array.iloc[i+1]
    array.iloc[i+1] = temp
dataframe = generate_velocities_position(400,10,2)

def simulation(temperature,length, no_of_particles,time):
    dataframe = generate_velocities_position(temperature,no_of_particles,length)
    print(dataframe)
    collision_number =0
    info  =dataframe.copy()
    mean_sq_dist = [0]
    dist =0
    times=[]
    times.append(0)
    initial_positions = dataframe['Positions']
    timespent =0
    data = {
        'Time':[timespent],
    }
    # print(initial_positions)
    
    for i in range(no_of_particles):
        data[f'{i+1}th_particle_position']=initial_positions.iloc[i]
    # print ()
    results = pd.DataFrame(data)
    smallest_collision_time = 1000
    results.to_excel("initial_positions.xlsx")

    while(time >1e-10):

        
        time_of_next_collision = 100
        first_point_position = info.iloc[0]['Positions']
        first_point_velocity = info.iloc[0]['Velocities']
        collision_index = -1
        if (first_point_velocity<0):
            time_of_next_collision = -1 * (first_point_position/first_point_velocity)
        last_point_position = info.iloc[no_of_particles-1]['Positions']
        last_point_velocity = info.iloc[no_of_particles - 1]['Velocities']
        if (last_point_velocity>0 and (length-last_point_position)/last_point_velocity<time_of_next_collision):
            time_of_next_collision = (length-last_point_position)/last_point_velocity
            collision_index = no_of_particles -1
        for i in range (0,no_of_particles-1):
            first_velocity = info.iloc[i]['Velocities']
            second_velocity = info.iloc[i+1]['Velocities']
            first_position = info.iloc[i]['Positions']
            second_position = info.iloc[i+1]['Positions']
            collision_time = (second_position - first_position)/(first_velocity-second_velocity)
            if (collision_time>0 and first_velocity> second_velocity):
                if(collision_time<time_of_next_collision):
                    time_of_next_collision =collision_time
                    collision_index = i
        for i in range (no_of_particles):
            info.iloc[i]['Positions'] = info.iloc[i]['Positions'] + time_of_next_collision*info.iloc[i]['Velocities']
            dist += abs(time_of_next_collision*info.iloc[i]['Velocities'])

        mean_sq_dist.append((time_of_next_collision*info.iloc[i]['Velocities']) **2+mean_sq_dist[-1])

        if (collision_index>-1 and collision_index<no_of_particles-1):
            temp = info.iloc[collision_index]['Velocities']
            info.iloc[collision_index]['Velocities'] = info.iloc[collision_index+1]['Velocities']
            info.iloc[collision_index+1]['Velocities'] = temp
        else:
            if (collision_index ==-1):
                info.iloc[collision_index+1]['Velocities'] =-1 * info.iloc[collision_index+1]['Velocities']
            else:
                info.iloc[collision_index]['Velocities'] =-1 * info.iloc[collision_index]['Velocities']
        collision_number +=1
        timespent+=time_of_next_collision
        time = time - time_of_next_collision
        data2 = {
            'Time': [timespent],
        }
        for i in range(0,no_of_particles):
            data2[f'{i+1}th_particle_position']=info['Positions'].iloc[i]
        curr_results = pd.DataFrame(data2)
        results = pd.concat([results, curr_results])

        # results.to_excel(f"{collision_number}th collision.xlsx")
        # smallest_collision_time = min(time_of_next_collision,smallest_collision_time)
        # results.plot(x='Time')
        # plt.legend(loc='center left', bbox_to_anchor=(0.90, 0.5))
        # plt.title("Positions of particles varying with time")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Position (m)")
        # plt.savefig(f"Collision#{collision_number}.png")
        # plt.show()
    results.plot(x='Time')
    plt.legend(loc='center left', bbox_to_anchor=(0.90, 0.5))
    plt.title("Positions of particles varying with time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.show()

    print(dist)
    print(results)

    return results
# def simulation(temperature,length, no_of_particles,total_time,dt, iterations):
    time_steps=[]
    time_passed = 0
    while (time_passed<total_time):
        time_passed+= dt
        time_steps.append(time_passed)
    Mean_sq_data ={
        'Time':time_steps,
    }
    # p)
    Mean_sq_data_pd = pd.DataFrame(Mean_sq_data)
    # print(Mean_sq_data_pd)
    # for k in iterations:
    #     Mean_sq_data.append(f'{k+1}th_iteration':time_steps)

    for j in range(iterations):
        # Mean_sq_data2 =
        Mean_sq_data
        dataframe = generate_velocities_position(temperature,no_of_particles,length)
        collision_number =0
        info  =dataframe.copy()
        mean_sq_dist = [0]
        dist =0
        times=[]
        times.append(0)
        initial_positions = dataframe['Positions']
        initial_velocities = dataframe['Velocities']
        timespent =0
        data = {
            'Time':[timespent],
        }
        # for l in range ()
        # print(initial_positions)
        for i in range(no_of_particles):
            data[f'{i+1}th_particle_position']=initial_positions.iloc[i]
            data[f'{i+1}th_particle_velocity']=initial_velocities.iloc[i]
        time = total_time
        time_of_simulation = total_time
        results = pd.DataFrame(data)
        smallest_collision_time = 1000
        while(time >1e-10):

            
            time_of_next_collision = 100
            first_point_position = info.iloc[0]['Positions']
            first_point_velocity = info.iloc[0]['Velocities']
            collision_index = -1
            if (first_point_velocity<0):
                time_of_next_collision = -1 * (first_point_position/first_point_velocity)
            last_point_position = info.iloc[no_of_particles-1]['Positions']
            last_point_velocity = info.iloc[no_of_particles - 1]['Velocities']
            if (last_point_velocity>0 and (length-last_point_position)/last_point_velocity<time_of_next_collision):
                time_of_next_collision = (length-last_point_position)/last_point_velocity
                collision_index = no_of_particles -1
            for i in range (0,no_of_particles-1):
                first_velocity = info.iloc[i]['Velocities']
                second_velocity = info.iloc[i+1]['Velocities']
                first_position = info.iloc[i]['Positions']
                second_position = info.iloc[i+1]['Positions']
                collision_time = (second_position - first_position)/(first_velocity-second_velocity)
                if (collision_time>0 and first_velocity> second_velocity):
                    if(collision_time<time_of_next_collision):
                        time_of_next_collision =collision_time
                        collision_index = i
            for i in range (no_of_particles):
                info.iloc[i]['Positions'] = info.iloc[i]['Positions'] + time_of_next_collision*info.iloc[i]['Velocities']
                dist += abs(time_of_next_collision*info.iloc[i]['Velocities'])

            mean_sq_dist.append((time_of_next_collision*info.iloc[i]['Velocities']) **2+mean_sq_dist[-1])

            if (collision_index>-1 and collision_index<no_of_particles-1):
                temp = info.iloc[collision_index]['Velocities']
                info.iloc[collision_index]['Velocities'] = info.iloc[collision_index+1]['Velocities']
                info.iloc[collision_index+1]['Velocities'] = temp
            else:
                if (collision_index ==-1):
                    info.iloc[collision_index+1]['Velocities'] =-1 * info.iloc[collision_index+1]['Velocities']
                else:
                    info.iloc[collision_index]['Velocities'] =-1 * info.iloc[collision_index]['Velocities']
            collision_number +=1
            timespent+=time_of_next_collision
            time = time - time_of_next_collision
            data2 = {
                'Time': [timespent],
            }
            for i in range(0,no_of_particles):
                data2[f'{i+1}th_particle_position']=info['Positions'].iloc[i]
                data2[f'{i+1}th_particle_velocities']=info['Velocities'].iloc[i]
            curr_results = pd.DataFrame(data2)
            results = pd.concat([results, curr_results])


            smallest_collision_time = min(time_of_next_collision,smallest_collision_time)
        # print(f"Smallest Collision Time is {smallest_collision_time}")
        time_spent = 0
        idx = 0
        # dt = 1e-6
        distance_travelled = []
        # mean_sq_data2 = []
        mean_sq_data2 ={
            'Time': time_steps,
            f'{j+1}th_iteration': time_steps
        }
        mean_sq_data_2_pd = pd.DataFrame(mean_sq_data2)
        while (time_spent<time_of_simulation):
            if(results['Time'].iloc[idx]<=time_spent):
                idx +=1
                # print(idx)
                continue
            else: 
                distance_travelled.append((dt*results[f'{no_of_particles//2+1}th_particle_velocities'].iloc[idx])**2)
                mean_sq_data_2_pd [f'{j+1}th_iteration'].iloc[idx+1]=(dt*results[f'{no_of_particles//2+1}th_particle_velocities'].iloc[idx]**2)
                time_spent +=dt
        # print(results[f'{no_of_particles//2+1}th_particle_velocities'])
        # print(mean_sq_data_2_pd)
        series = pd.Series(mean_sq_data_2_pd[f'{j+1}th_iteration'])
        Mean_sq_data_pd = pd.concat([Mean_sq_data_pd,series],axis=1)
        # results.plot(x='Time', y = f'{no_of_particles//2+1}th_particle_velocities')
        # plt.show()


        # mean_sq = pd.DataFrame(distance_travelled[1:])

        # plt.plot()
        # print(dist)
        # print(results)
    for o in range(iterations):
        Mean_sq_data_pd.plot(x='Time', y = f'{o+1}th_iteration')
    plt.show()
    print(Mean_sq_data_pd)


simulation(temperature=400,length=1,no_of_particles=10,time=5)



