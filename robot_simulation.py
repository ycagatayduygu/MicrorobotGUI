import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.ndimage import binary_dilation, distance_transform_edt

# ---------- PARAMETERS ---------- #
np.random.seed(42)
ENV_SIZE = 512
START, GOAL = (10, 10), (460, 470)

N_OBS, R_MIN, R_MAX = 8, 25, 60          # obstacle count & radii
BUFFER_PX          = 7                  # safety buffer (pixels) ← tweak this!

# ---------- GENERATE OBSTACLES ---------- #
obs_map = np.zeros((ENV_SIZE, ENV_SIZE), dtype=bool)
obstacles = []
YY, XX = np.ogrid[:ENV_SIZE, :ENV_SIZE]
for _ in range(N_OBS):
    cx, cy = np.random.randint(60, ENV_SIZE-60, size=2)
    r = np.random.randint(R_MIN, R_MAX)
    mask = (XX-cx)**2 + (YY-cy)**2 <= r**2
    obs_map[mask] = True
    obstacles.append((cx, cy, r))

# inflate by BUFFER_PX to forbid cells that violate clearance
occ_buffer = binary_dilation(obs_map, iterations=BUFFER_PX)

# distance‑transform for soft penalty (optional)
dist_to_obs = distance_transform_edt(~obs_map)

# ---------- A* WITH BUFFER ---------- #
def heuristic(a,b): return np.hypot(a[0]-b[0], a[1]-b[1])

def astar(start, goal, occupancy, dist_map, buf=BUFFER_PX, k_pen=1.0):
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    move_cost = [1,1,1,1,np.sqrt(2)]*4
    h, w = occupancy.shape
    open_set = [(heuristic(start,goal),0,start,None)]
    came_from, gscore = {}, {start:0}
    while open_set:
        f,g,cur,parent = heapq.heappop(open_set)
        if cur in came_from: continue
        came_from[cur]=parent
        if cur==goal: break
        for (dx,dy),mc in zip(moves,move_cost):
            nx, ny = cur[0]+dy, cur[1]+dx
            if not (0<=nx<h and 0<=ny<w): continue
            if occupancy[nx,ny]: continue   # inside buffer → forbidden
            # soft penalty if within 3×buffer
            d = dist_map[nx,ny]
            soft_pen = k_pen * max(0, (buf*2 - d))**2 / (buf**2) if d<buf*3 else 0
            tg = g + mc + soft_pen
            neigh = (nx,ny)
            if tg < gscore.get(neigh, np.inf):
                gscore[neigh]=tg
                heapq.heappush(open_set,(tg+heuristic(neigh,goal),tg,neigh,cur))
    # reconstruct
    if goal not in came_from: return None
    path=[]; node=goal
    while node is not None:
        path.append(node); node=came_from[node]
    return path[::-1]

path = astar(START, GOAL, occ_buffer, dist_to_obs)
assert path is not None, "No path"

# ---------- WAYPOINTS ---------- #
def extract_waypoints(path, step=40):
    wpts=[path[0]]; acc=0
    for i in range(1,len(path)):
        acc += np.linalg.norm(np.subtract(path[i],path[i-1]))
        if acc>=step: wpts.append(path[i]); acc=0
    wpts.append(path[-1]); return wpts
waypoints = extract_waypoints(path)

# ---------- SIMPLE TRACKER ---------- #
K, dt = 2.508, 0.1
max_turn, v_bounds = np.deg2rad(45), (0.1,2.0)
def step_dynamics(pos, h, Wm, th_dot):
    V=K*Wm; h+=th_dot*dt
    return (pos[0]+V*np.sin(h)*dt, pos[1]+V*np.cos(h)*dt), h
def ctrl(pos, h, wp):
    vec = (wp[0]-pos[0], wp[1]-pos[1])
    des_h = np.arctan2(vec[0], vec[1])
    h_err = (des_h - h + np.pi)%(2*np.pi)-np.pi
    th_dot = np.clip(h_err/dt, -max_turn, max_turn)
    Wm = np.clip(0.8+1.2*(abs(h_err)<0.3), *v_bounds)
    return Wm, th_dot

sim=[START]; heading=0; idx=1
for _ in np.arange(0,300,dt):
    wp=waypoints[idx]; pos=sim[-1]
    if np.linalg.norm(np.subtract(wp,pos))<3:
        if idx<len(waypoints)-1: idx+=1; wp=waypoints[idx]
        else: break
    Wm,td=ctrl(pos,heading,wp)
    pos,heading = step_dynamics(pos,heading,Wm,td)
    sim.append(pos)
sim=np.array(sim)

# ---------- PLOT ---------- #
fig,ax=plt.subplots(figsize=(6,6)); ax.set_aspect('equal'); ax.set_xlim(0,ENV_SIZE); ax.set_ylim(0,ENV_SIZE)
# original obstacles
for cx,cy,r in obstacles:
    ax.add_patch(plt.Circle((cx,cy), r, color='k', alpha=.25))
# buffer contour (outline)
for cx,cy,r in obstacles:
    ax.add_patch(plt.Circle((cx,cy), r+BUFFER_PX, fill=False, ls='--', color='k', alpha=.4))
ax.plot([p[1] for p in path],[p[0] for p in path], lw=1, label='A* path', alpha=.5, color='orange')
ax.plot(sim[:,1],sim[:,0],'-r',lw=2,label='Sim trajectory')
ax.scatter([w[1] for w in waypoints],[w[0] for w in waypoints],c='g',s=32,label='Waypoints')
ax.scatter(START[1],START[0],c='b',s=40,label='Start')
ax.scatter(GOAL[1],GOAL[0],c='m',s=40,label='Goal')
ax.legend(); ax.set_title(f"A* with {BUFFER_PX}px Buffer"); ax.invert_yaxis(); plt.tight_layout(); plt.show()
