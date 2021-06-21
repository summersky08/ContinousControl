#!/usr/bin/env python
# coding: utf-8

# - GPI

# In[2]:


import numpy as np 
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time


def visualize(car_states, ref_traj, obstacles, t, time_step, save=False):
    init_state = car_states[0,:]
    def create_triangle(state=[0,0,0], h=0.5, w=0.25, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [cos(th), -sin(th)],
            [sin(th),  cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        return path, current_state, target_state,

    def animate(i):
        # get variables
        x = car_states[i,0]
        y = car_states[i,1]
        th = car_states[i,2]

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        #x_new = car_states[0, :, i]
        #y_new = car_states[1, :, i]
        #horizon.set_data(x_new, y_new)

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))

        # update current_target
        x_ref = ref_traj[i,0]
        y_ref = ref_traj[i,1]
        th_ref = ref_traj[i,2]
        target_state.set_xy(create_triangle([x_ref, y_ref, th_ref], update=True))

        # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)            

        return path, current_state, target_state,
    circles = []
    for obs in obstacles:
        circles.append(plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha = 0.5))
    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    min_scale_x = min(init_state[0], np.min(ref_traj[:,0])) - 1.5
    max_scale_x = max(init_state[0], np.max(ref_traj[:,0])) + 1.5
    min_scale_y = min(init_state[1], np.min(ref_traj[:,1])) - 1.5
    max_scale_y = max(init_state[1], np.max(ref_traj[:,1])) + 1.5
    ax.set_xlim(left = min_scale_x, right = max_scale_x)
    ax.set_ylim(bottom = min_scale_y, top = max_scale_y)
    for circle in circles:
        ax.add_patch(circle)
    # create lines:
    #   path
    path, = ax.plot([], [], 'k', linewidth=2)

    #   current_state
    current_triangle = create_triangle(init_state[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='r')
    current_state = current_state[0]
    #   target_state
    target_triangle = create_triangle(ref_traj[0,0:3])
    target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b')
    target_state = target_state[0]

    #   reference trajectory
    ax.scatter(ref_traj[:,0], ref_traj[:,1], marker='x')

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=time_step*100,
        blit=True,
        repeat=True
    )
    
    from IPython.display import HTML
    HTML(sim.to_jshtml())
    
    plt.show()

    if save == True:
        sim.save('./fig/animation' + str(time()) +'.gif', writer='ffmpeg', fps=15)

    return


# In[3]:


from time import time
import numpy as np
from utils import visualize

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


# In[4]:


import numpy as np
from math import atan2
tau = 0.1
t = np.linspace(0,200,2001)
r = np.array(lissajous(t))
r[2, :] = [atan2(r[1, i], r[0, i]) for i in range(r.shape[1])]


# In[5]:


def p2i(p):
    pi = np.zeros(3,dtype=int)
    pi[0] = (p[0]-(-3))//resol[0]
    pi[1] = (p[1]-(-3))//resol[1]
    pi[2] = ((p[2]-(-np.pi)) % (2*np.pi))//resol[2]
    return pi

def i2p(pi):
    p = np.zeros(3)
    p[0] = -3+(pi[0] + 0.5)*resol[0]
    p[1] = -3+(pi[1] + 0.5)*resol[1]
    p[2] = -np.pi+(pi[2] + 0.5)*resol[2]
    return p


# In[6]:


def u2i(u):
    ui = np.zeros(2, dtype=int)
    ui[0] = u[0] // resol[3]
    ui[1] = (u[1]-(-1)) // resol[4]
    return ui

def i2u(ui):
    u = np.zeros(2)
    u[0] = (ui[0] + 0.5) * resol[3]
    u[1] = -1 + (ui[1] + 0.5) * resol[4]
    return u


# In[7]:


def normal_pdf(x, m, std):
    a = np.log(1/(2*np.pi*std**2))
    b = -(x-m)**2/(2*std**2)
    return a+b


# In[34]:


Q = np.eye(2)*10
R = np.ones((2,2))
q = 1
gamma = 0.9


# In[35]:


G = lambda x, u, t, alpha: np.array([[t*np.cos(x[2] + alpha), 0], [t*np.sin(x[2] + alpha), 0], [0, t]])
wmean = [0,0,0]
wstd = np.array([0.04, 0.04, 0.004])


# In[36]:


def a2i(a):
    return int(((a-(-np.pi)) % (2*np.pi))//resol[5])

def i2a(i):
    return -np.pi+(i + 0.5)*resol[5]


# In[37]:


def calc_cost(p, u):
    return p[:-1].T @ Q @ p[:-1] + q*(1-np.cos(p[-1]))**2 + u.T @ R @ u


# In[38]:


def step(G, p, tau, wmean, wvar, u, const, const2, random=True):
    p += G(p, u, tau, const2) @ u + const
    if random:
        p += np.random.normal(wmean, wvar, size=3)
    p[-1] %= np.pi*2
    return p


# In[39]:


def V_init():
    #resol = [0.2,0.2,0.3,0.5,0.5, 0.3]
    #resol = [0.3,0.3,0.6,0.3,0.6, 0.6]
    pmax = np.array([3,3,np.pi-1e-5])
    pimax = p2i(pmax)
    umax = np.array([1,1])
    uimax = u2i(umax)
    #imax = np.hstack([pimax, uimax])

    amax = np.pi-1e-5
    aimax = a2i(amax)


    #V = np.zeros(imax+1)
    imax = np.hstack([pimax, aimax])
    V = np.zeros(imax+1)
    return V


# In[40]:


def policy_init(V):
    u = np.array([0, 0])
    ui = u2i(u)
    policy = np.empty_like(V, dtype=object)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            for k in range(policy.shape[2]):
                for ai in range(policy.shape[3]):
                    policy[i, j, k, ai] = ui.copy()
    return policy


# In[41]:


def policy_evaluate(V, policy):
    for itr in range(10):
        print(f"{itr+1}-th iteration")
        V2 = np.zeros_like(V)
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                print(i, j)
                for k in range(V.shape[2]):
                    pi = np.array([i, j, k])
                    p = i2p(pi)

                    for ai in range(V.shape[3]):
                        ui = policy[i, j, k, ai]
                        u = i2u(ui)
                        stage_cost = calc_cost(p, u)

                        a = i2a(ai)

                        p0 = step(G, p.copy(), 0.1, 0, 0, i2u(ui), 0, a, random=False)

                        # 26近傍の確率を求める
                        prob = np.full(V.shape[:-1], -np.inf)
                        for di in [-1, 0, 1]:
                            i2 = i + di
                            if not (0<=i2<V.shape[0]): continue
                            for dj in [-1, 0, 1]:
                                j2 = j + dj
                                if not (0<=j2<V.shape[1]): continue
                                for dk in [-1, 0, 1]:
                                    k2 = k + dk
                                    k2 %= V.shape[2]
                                    if i2==j2==k2==0: continue

                                    pi2 = np.array([i2,j2,k2])
                                    p2 = i2p(pi2)

                                    prob[i2, j2, k2] = 0
                                    prob[i2, j2, k2] += normal_pdf(p2[0], p0[0], 0.04)
                                    prob[i2, j2, k2] += normal_pdf(p2[1], p0[1], 0.04)
                                    prob[i2, j2, k2] += max(
                                        normal_pdf(p2[2], p0[2], 0.004),
                                        normal_pdf(p2[2]+2*np.pi, p0[2], 0.004),
                                        normal_pdf(p2[2]-2*np.pi, p0[2], 0.004)
                                    )

                                    #print(i,j,k,i2,j2,k2,prob[i2,j2,k2])

                        prob = np.exp(prob.astype(np.float128)) / np.exp(prob.astype(np.float128)).sum()
                        V2[i, j, k, ai] = stage_cost + gamma * (V[:,:,:,ai] * prob).sum()
                        #print(V2[i,j,k,ai])
                        #break
                    #break
                #break
            #break

        V = V2.copy()
        
    return V


# In[42]:


def policy_improve(V, policy):
    policy2 = np.zeros_like(policy, dtype=object)
    res = 0
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            print(i, j)
            for k in range(V.shape[2]):
                pi = np.array([i, j, k])
                p = i2p(pi)

                for ai in range(V.shape[3]):
                    a = i2a(ai)

                    mincost = np.inf
                    argmin = -1
                    for ui1 in range(uimax[0]+1):
                        for ui2 in range(uimax[1]+1):
                            ui = np.array([ui1, ui2])
                            u = i2u(ui)

                            p0 = step(G, p.copy(), 0.1, 0, 0, u, 0, a, random=False)

                            # 26近傍の確率を求める
                            prob = np.full(V.shape[:-1], -np.inf)
                            for di in [-1, 0, 1]:
                                i2 = i + di
                                if not (0<=i2<V.shape[0]): continue
                                for dj in [-1, 0, 1]:
                                    j2 = j + dj
                                    if not (0<=j2<V.shape[1]): continue
                                    for dk in [-1, 0, 1]:
                                        k2 = k + dk
                                        k2 %= V.shape[2]
                                        if i2==j2==k2==0: continue

                                        pi2 = np.array([i2,j2,k2])
                                        p2 = i2p(pi2)

                                        prob[i2, j2, k2] = 0
                                        prob[i2, j2, k2] += normal_pdf(p2[0], p0[0], 0.04)
                                        prob[i2, j2, k2] += normal_pdf(p2[1], p0[1], 0.04)
                                        prob[i2, j2, k2] += max(
                                            normal_pdf(p2[2], p0[2], 0.004),
                                            normal_pdf(p2[2]+2*np.pi, p0[2], 0.004),
                                            normal_pdf(p2[2]-2*np.pi, p0[2], 0.004)
                                        )

                                        #print(i,j,k,i2,j2,k2,prob[i2,j2,k2])

                            prob = np.exp(prob.astype(np.float128)) / np.exp(prob.astype(np.float128)).sum()

                            expected_value = (V[:, :, :, ai] * prob).sum()

                            #print(expected_value, ui)
                            if expected_value < mincost:
                                mincost = expected_value
                                argmin = ui

                    #print(mincost, argmin)
                    policy2[i,j,k,ai] = argmin
                    if (policy[i,j,k,ai] != policy2[i,j,k,ai]).any(): res += 1
                    #break
                #print(i, j, k, "policy change: ", res)
                #break
            #break
        #break
    return policy2, res


# In[43]:


def plan(policy):
    #%%time

    his = [car_states[0]]
    p = car_states[0] - r[:, 0]
    for tau_i, tau in enumerate(t[:1001]):
        i, j, k = p2i(p)

        i, j = np.minimum(pimax[:-1], [i, j])
        i, j = np.maximum([0,0], [i,j])

        ai = a2i(r[2, tau_i])
        ui = policy[i, j, k, ai]
        u = i2u(ui)

        p2 = step(G, p.copy(), 0.1, wmean, wstd, u, r[:, tau_i]-r[:, tau_i+1], r[2, tau_i])

        # collision avoid
        d1, d2 = norm(p2[:-1]+r[:-1, tau_i]-c1), norm(p2[:-1]+r[:-1, tau_i]-c2)

        if d1 < d2: c = c1
        else: c = c2

        if min(d1,d2) < 1:
            maxdist = 0
            argmax = -1
            for ui1 in range(uimax[0]+1):
                for ui2 in range(uimax[1]+1):
                    ui = np.array([ui1, ui2])
                    u = i2u(ui)

                    p0 = step(G, p.copy(), 0.1, 0, 0, u, r[:, tau_i]-r[:, tau_i+1], r[2, tau_i], random=False)

                    dist = norm(p0[:-1]+r[:-1, tau_i] - c)

                    if dist > maxdist:
                        maxdist = dist
                        argmax = u

            p2 = step(G, p.copy(), 0.1, wmean, wstd, argmax, r[:, tau_i]-r[:, tau_i+1], r[2, tau_i])

        p = p2

        his.append(p +r[:, tau_i+1])
        
    return his


# In[44]:


def plot(his):
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(15,15))
    ax = plt.axes()

    # fc = face color, ec = edge color
    C1 = patches.Circle(xy=(-2,-2), radius=0.5, ec='k', fill=False)
    C2 = patches.Circle(xy=(1,2), radius=0.5, ec='k', fill=False)
    ax.add_patch(C1)
    ax.add_patch(C2)
    ax.axis("equal")

    his = np.array(his)
    plt.plot(his[:,0], his[:,1])
    plt.plot(r[0, :len(his)], r[1, :len(his)])


# In[45]:


def norm(x):
    return (x**2).sum()**0.5


# In[ ]:


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
        
    record = {}
    res = 10**20
    resol = [0.3,0.3,0.6,0.3,0.6, 0.6]
    V = V_init()
    policy = policy_init(V)

    i = 0
    while res > 10:
        print(f"{i+1}-th iteration (Large)")
        V = policy_evaluate(V, policy)
        policy, res = policy_improve(V, policy)
        record[i] = (V.copy(), policy.copy(), res)
        i += 1
        print("policy change: ", res)
        
    resol = [0.3,0.3,0.6,0.3,0.6, 0.6]
    his = plan(policy)

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




