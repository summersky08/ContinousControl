#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import patches
from time import time

from scipy.sparse import csc_matrix


# In[17]:


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


# In[18]:


import numpy as np
from math import atan2
tau = 0.1
t = np.linspace(0,200,2001)
r = np.array(lissajous(t))
r[2, :] = [atan2(r[1, i], r[0, i]) for i in range(r.shape[1])]


# In[19]:


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


# In[20]:


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


# In[21]:


def a2i(a):
    return int(((a-(-np.pi)) % (2*np.pi))//resol[5])

def i2a(i):
    return -np.pi+(i + 0.5)*resol[5]


# In[22]:


p = car_states[0] - r[:, 0]
G = lambda x, u, t, alpha: np.array([[t*np.cos(x[2] + alpha), 0], [t*np.sin(x[2] + alpha), 0], [0, t]])
wmean = [0,0,0]
wstd = np.array([0.04,0.04,0.004])


# In[23]:


Q = np.eye(2)*10
R = np.ones((2,2))
q = 1
gamma = 0.9


# In[24]:


#resol = [0.2,0.2,0.3,0.5,0.5, 0.3]
resol = [0.3,0.3,0.6,0.3,0.6, 0.6]
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
VV = V.flatten()


# In[25]:


def calc_cost(p, u):
    return p[:-1].T @ Q @ p[:-1] + q*(1-np.cos(p[-1]))**2 + u.T @ R @ u


# In[26]:


def step(G, p, tau, wmean, wstd, u, const, const2, random=True):
    p += G(p, u, tau, const2) @ u + const
    if random:
        p += np.random.normal(wmean, wstd, size=3)
    p[-1] %= np.pi*2
    return p


# In[27]:


def normal_pdf(x, m, std):
    a = np.log(1/(2*np.pi*std**2))
    b = -(x-m)**2/(2*std**2)
    return a+b


# In[28]:


col = np.hstack([np.arange(len(VV)), np.arange(len(VV))])
col.sort()
row = np.arange(len(VV)*2)
data = np.array([(0 + 0.5) * resol[3], (0 + 0.5) * resol[4] + (-1)] * len(VV))

U = csc_matrix((data, (row, col)), shape=(len(VV)*2, len(VV)))


# In[29]:


Xi = np.array([i for i in range(V.shape[0]) for j in range(V.shape[1]) for k in range(V.shape[2]) for ai in range(V.shape[3])])
X = (Xi + 0.5) * resol[0] + (-3)
Yi = np.array([j for i in range(V.shape[0]) for j in range(V.shape[1]) for k in range(V.shape[2]) for ai in range(V.shape[3])])
Y = (Yi + 0.5) * resol[1] + (-3)

data = np.hstack([X.reshape(-1, 1),Y.reshape(-1, 1)]).flatten()

P = csc_matrix((data, (row, col)), shape=(len(VV)*2, len(VV)))


# In[30]:


col = np.hstack([np.arange(len(VV)*2), np.arange(len(VV)*2)])
col.sort()
row = np.hstack([np.arange(len(VV)*2).reshape(-1,2), np.arange(len(VV)*2).reshape(-1,2)])
row = row.flatten()

data = np.tile(Q.flatten(), reps=len(VV))
QQ = csc_matrix((data, (row, col)), shape=(len(VV)*2, len(VV)*2))

data = np.tile(R.flatten(), reps=len(VV))
RR = csc_matrix((data, (row, col)), shape=(len(VV)*2, len(VV)*2))


# In[31]:


Thi = np.array([k for k in range(V.shape[0]) for j in range(V.shape[1]) for k in range(V.shape[2]) for ai in range(V.shape[3])])
Th = (Thi + 0.5) * resol[2] + (-np.pi)

Ai = np.array([ai for i in range(V.shape[0]) for j in range(V.shape[1]) for k in range(V.shape[2]) for ai in range(V.shape[3])])
A = (Ai + 0.5)*resol[5] + (-np.pi)


# In[32]:


col = np.hstack([np.arange(len(VV)), np.arange(len(VV)),np.arange(len(VV))])
col.sort()
row = np.arange(len(VV)*3)
data = np.hstack([X.reshape([-1, 1]), Y.reshape([-1,1]), Th.reshape([-1, 1])]).flatten()

E = csc_matrix((data, (row, col)), shape=(len(VV)*3, len(VV)))


# In[33]:


col = np.hstack([np.arange(len(VV))*2, np.arange(len(VV))*2,np.arange(len(VV))*2+1])
col.sort()
row = np.arange(len(VV)*3)

tau = 0.1
a = np.cos(Th + A)
b = np.sin(Th + A)
c = np.ones(len(VV))
data = np.hstack([a.reshape([-1, 1]), b.reshape([-1, 1]), c.reshape([-1, 1])]).flatten() * tau

GG = csc_matrix((data, (row, col)), shape=(len(VV)*3, len(VV)*2))


# In[34]:


#resol = [0.2,0.2,0.3,0.5,0.5, 0.3]
resol = [0.3,0.3,0.6,0.3,0.6, 0.6]
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

VV = V.flatten()


# In[35]:


col = np.hstack([np.arange(len(VV)), np.arange(len(VV))])
col.sort()
row = np.arange(len(VV)*2)
data = np.array([(0 + 0.5) * resol[3], (0 + 0.5) * resol[4] + (-1)] * len(VV))

U = csc_matrix((data, (row, col)), shape=(len(VV)*2, len(VV)))


# In[36]:


data = np.array([((i + 0.5)*resol[3], (j + 0.5)*resol[4]+(-1)) for i in range(uimax[0]+1) for j in range(uimax[1]+1)]).flatten()

row = np.arange(len(data)*V.shape[3])
col = np.hstack([np.arange(len(data)*V.shape[3]//2), np.arange(len(data)*V.shape[3]//2)])
col.sort()

data = np.tile(data, reps=V.shape[3])

UU = csc_matrix((data, (row, col)), shape=(len(data), len(data)//2))


# In[39]:


row = np.arange(UU.shape[1]*3)
col = np.hstack([np.arange(UU.shape[1])*2, np.arange(UU.shape[1])*2,np.arange(UU.shape[1])*2+1])
col.sort()

k=0
A2 = [(ai + 0.5)*resol[5]+(-np.pi) for ai in range(V.shape[2])]*(uimax+1).prod()
Th2 = np.array([(k + 0.5) * resol[2] + (- np.pi)]*len(A2))

data = np.vstack([np.cos(Th2+A2), np.sin(Th2+A2), np.ones(len(A2))]).T.flatten()

GG2 = csc_matrix((data, (row, col)), shape=(UU.shape[1]*3, UU.shape[1]*2))


# In[40]:


#U_base = np.array([(i, j) for i in range(uimax[0]+1) for j in range(uimax[1]+1)])
U_base = np.array([((i + 0.5)*resol[3], (j + 0.5)*resol[4]+(-1)) for i in range(uimax[0]+1) for j in range(uimax[1]+1)])


# In[42]:


Ui_base = np.array([(i, j) for i in range(uimax[0]+1) for j in range(uimax[1]+1)])


# In[45]:


def plan(policy, num_point = 1001):
    #%%time

    his = [car_states[0]]
    p = car_states[0] - r[:, 0]
    for tau_i, tau in enumerate(t[:num_point]):
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


# In[46]:


def plot(his, num_point = 1001):
    num_point = min(num_point, len(his))
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
    plt.plot(his[:num_point,0], his[:num_point,1])
    plt.plot(r[0, :num_point], r[1, :num_point])


# In[47]:


def norm(x):
    return (x**2).sum()**0.5

c1,c2 = np.array([-2,-2]), np.array([1,2])


# In[48]:


def policy_evaluate(V, U, num_iter = 20):
    global Xi,Yi,Thi,Ai
    for itr in range(num_iter):
        V2 = np.zeros_like(V)

        stage_cost = (P.T @ QQ @ P).diagonal() + q*(1-np.cos(Th))**2 + (U.T @ RR @ U).diagonal()

        E_next = E + GG @ U

        # 遷移
        col = np.hstack([np.arange(len(VV)), np.arange(len(VV)),np.arange(len(VV))])
        col.sort()
        row = np.arange(len(VV)*3)

        res = np.array(E_next[row, col]).reshape(-1, 3)
        X2, Y2, Th2 = res[:,0], res[:,1], res[:,2]
        X2 = (X2 - (-3)) // resol[0]
        Y2 = (Y2 - (-3)) // resol[1]
        Th2 = (Th2 - (-np.pi)) // resol[2]

        X2 = np.minimum(V.shape[0]-1, X2)
        X2 = np.maximum(0, X2)
        Y2 = np.minimum(V.shape[1]-1, Y2)
        Y2 = np.maximum(0, Y2)
        #Th2 = np.minimum(V.shape[2]-1, Th2)
        #Th2 = np.maximum(0, Th2)
        Th2 %= V.shape[2]

        Xi,Yi,Thi,Ai,X2,Y2,Th2 = map(lambda x: x.astype(int), (Xi,Yi,Thi,Ai,X2,Y2,Th2))

        V2[Xi, Yi, Thi, Ai] += stage_cost + gamma * V[X2, Y2, Th2, Ai]

        V = V2.copy()
    return V


# In[49]:


# インデクシングに0.5を書き加えたバージョン

def policy_improve(V, U):
    udata = np.array([0, 0] * len(VV))
    policy2 = np.zeros(len(VV))
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                p = np.array([(i + 0.5)*resol[0]+(-3), (j + 0.5)*resol[1]+(-3), (k + 0.5)*resol[2]+(-np.pi)])
                col = np.hstack([np.arange(UU.shape[1]), np.arange(UU.shape[1]),np.arange(UU.shape[1])])
                col.sort()
                row = np.arange(UU.shape[1]*3)
                data = np.tile(p, reps=UU.shape[1])

                E2 = csc_matrix((data, (row, col)), shape=(UU.shape[1]*3, UU.shape[1]))

                row = np.arange(UU.shape[1]*3)
                col = np.hstack([np.arange(UU.shape[1])*2, np.arange(UU.shape[1])*2,np.arange(UU.shape[1])*2+1])
                col.sort()

                Ai2 = np.array([ai for ai in range(V.shape[2])]*(uimax+1).prod())
                Ai2.sort()
                A2 = (Ai2 + 0.5)*resol[5]+(-np.pi)
                Th2 = np.array([(k + 0.5)*resol[2]+(-np.pi)]*len(A2))

                #data = np.vstack([np.cos(Th2+A2), np.sin(Th2+A2), np.ones(len(A2))]).T.flatten() * tau
                
                data = np.vstack([np.cos(Th2+A2), np.sin(Th2+A2), np.ones(len(A2))]).T.flatten()

                GG2 = csc_matrix((data, (row, col)), shape=(UU.shape[1]*3, UU.shape[1]*2))

                E_next = E2 + GG2 @ UU

                # 遷移
                col = np.hstack([np.arange(UU.shape[1]), np.arange(UU.shape[1]),np.arange(UU.shape[1])])
                col.sort()
                row = np.arange(UU.shape[1]*3)

                res = np.array(E_next[row, col]).reshape(-1, 3)
                X2, Y2, Th2 = res[:,0], res[:,1], res[:,2]
                X2 = (X2 - (-3)) // resol[0]
                Y2 = (Y2 - (-3)) // resol[1]
                Th2 = (Th2 - (-np.pi)) // resol[2]

                X2 = np.minimum(V.shape[0]-1, X2)
                X2 = np.maximum(0, X2)
                Y2 = np.minimum(V.shape[1]-1, Y2)
                Y2 = np.maximum(0, Y2)
                Th2 %= V.shape[2]

                #print(Ai2)
                #Ai2.sort()
                #print(Ai2)
                Ai2,X2,Y2,Th2 = map(lambda x: x.astype(int), (Ai2,X2,Y2,Th2))

                costs = V[X2, Y2, Th2, Ai2].reshape([V.shape[3], -1])

                argmin = costs.argmin(1)

                u = U_base[argmin]

                idx = i*(V.shape[1]*V.shape[2]) + j*V.shape[2] + k
                udata[idx*V.shape[3]*2:(idx+1)*V.shape[3]*2] = u.flatten()

                policy2[idx*V.shape[3]:(idx+1)*V.shape[3]] = argmin


    col = np.hstack([np.arange(len(VV)), np.arange(len(VV))])
    col.sort()
    row = np.arange(len(VV)*2)

    U = csc_matrix((udata.flatten(), (row, col)), shape=(len(VV)*2, len(VV)))
    return U, policy2


# In[ ]:


get_ipython().run_cell_magic('time', '', 'q = 1\nfor i in range(200):\n    V = policy_evaluate(V, U, num_iter=10)\n    U, argmin2 = policy_improve(V, U)\n    if i:\n        change = (argmin != argmin2).sum()\n        print(change)\n    argmin = argmin2.copy()\n    if i and change < 10: break')


# In[51]:


Policy = Ui_base[argmin.astype(int)].reshape(list(V.shape)+[2])


# In[ ]:


his = plan(Policy)
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




