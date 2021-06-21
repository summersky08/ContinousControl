#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import time

from math import atan2

from casadi import *

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1


# In[3]:


# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step * f.flatten()

if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller by your own controller
        control = simple_controller(cur_state, cur_ref)
        #print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        #print(cur_iter)
        #print(t2-t1)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state - cur_ref)
        cur_iter = cur_iter + 1
        
        break
        
    Q = SX([[10,0],[0,10]])
    R = SX([[1,1], [1,1]])

    wmean = [0,0,0]
    wstd = np.array([0.04,0.04,0.004])
    
    t = np.linspace(0,200,2001)
    r = np.array(lissajous(t))
    r[2, :] = [atan2(r[1, i], r[0, i]) for i in range(r.shape[1])]
    
    # solving NLP part
    tau = 0.1
    q = 1
    T = 3
    Q2 = SX(np.eye(2))
    R2 = SX(np.eye(2)*0)
    p0 = SX(car_states[0] - r[:, 0])
    his = [car_states[0]]
    for ti, _ in enumerate(t[:1200]):
        d1,d2 = norm_2(p0[:-1]+r[:-1, ti] - SX([-2,-2])), norm_2(p0[:-1]+r[:-1, ti] - SX([1,2]))
        if float(d1) < float(d2): c = SX([-2,-2])
        else: c = SX([1,2])

        V = 0
        u = SX.sym("u", T*2)
        p = p0[:]
        d = SX([0]*T)
        for i in range(T):
            V += p[:2].T @ Q2 @ p[:2] + u[2*i:2*(i+1)].T @ R2 @ u[2*i:2*(i+1)]
            p[0] += u[2*i]*tau*cos(p[-1] + r[2, ti+i])
            p[1] += u[2*i]*tau*sin(p[-1] + r[2, ti+i])
            p[2] += tau * u[2*i+1]
            p += r[:, ti+i] - r[:, ti+i+1]
            d[i] = norm_2(p[:-1]+r[:-1,ti+i+1]-c)

        nlp = {'x':u, 'f':V, 'g':vertcat(d)}
        S = nlpsol('S', 'ipopt', nlp)

        res = S(x0=[0.5, 0]*(T),          lbx = [0,-1]*T, 
          ubx = [1,1]*T,
          lbg=[0.6]*T, ubg=[100000]*T)
        u_opt = res['x']

        p0[0] += u_opt[0]*tau*cos(p0[-1] + r[2, ti+i])
        p0[1] += u_opt[0]*tau*sin(p0[-1] + r[2, ti+i])
        p0[2] += tau * u_opt[1]
        p0 += r[:, ti+i] - r[:, ti+i+1]
        p0 += np.random.normal(wmean, wstd, size=3)

        his.append(np.array(DM(p0)).flatten() + r[:, ti])


    his = np.array(his)

    fig, axes = plt.subplots(3,3,figsize=(15,15))
    for i in range(1, 10):
        row, col = divmod(i-1, 3)
        axes[row, col].plot(r[0, :i*100], r[1, :i*100], label="reference")
        axes[row, col].plot(his[:i*100, 0], his[:i*100, 1], label="robot")

        C1 = patches.Circle(xy=(-2,-2), radius=0.5, ec='k', fill=False)
        C2 = patches.Circle(xy=(1,2), radius=0.5, ec='k', fill=False)
        axes[row, col].add_patch(C1)
        axes[row, col].add_patch(C2)
        axes[row, col].axis("equal")
        axes[row, col].legend()
        axes[row, col].set_title(f"t = {i * 10}")


# In[ ]:




