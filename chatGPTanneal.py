import numpy as np
import matplotlib.pyplot as plt

# ---------- Quaternion utilities ----------

def axis_angle_to_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    w = np.cos(half)
    xyz = np.sin(half) * axis
    return np.hstack([w, xyz])  # (w, x, y, z)

def quat_conj(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([w, -x, -y, -z], axis=-1)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def quat_angle(q):
    w = np.clip(np.abs(q[..., 0]), 0.0, 1.0)
    return 2 * np.arccos(w)

def random_quats(n):
    u1, u2, u3 = np.random.rand(3, n)
    q = np.empty((n, 4))
    q[:, 1] = np.sqrt(1-u1) * np.sin(2*np.pi*u2)
    q[:, 2] = np.sqrt(1-u1) * np.cos(2*np.pi*u2)
    q[:, 3] = np.sqrt(u1)   * np.sin(2*np.pi*u3)
    q[:, 0] = np.sqrt(u1)   * np.cos(2*np.pi*u3)
    return q  # (n,4) w,x,y,z

def hex_sym_quats():
    ops = []
    # 6-fold around c-axis
    for k in range(6):
        ops.append(axis_angle_to_quat([0,0,1], k*np.pi/3))
    # 6 two-fold in basal plane
    for k in range(6):
        theta = k*np.pi/6
        ops.append(axis_angle_to_quat([np.cos(theta), np.sin(theta), 0], np.pi))
    return np.array(ops)

SYM_OPS = hex_sym_quats()  # (12,4)

def misori_angle(q1, q2):
    """
    Minimum misorientation angle (rad) between two unit quats with HCP symmetry.
    q1, q2 shape (...,4)
    """
    q1_ext = q1[None, ...]  # (1,...,4)
    q2_conj = quat_conj(q2)[None, ...]
    prod = quat_mul(SYM_OPS[:, None, :], q1_ext)
    delta = quat_mul(prod, q2_conj)
    ang = quat_angle(delta)  # (12,...)
    return ang.min(axis=0)

# ---------- Neighbourhood utilities ----------

def neighbour_table(N, use26=True):
    rng = [-1, 0, 1]
    offsets = []
    for dx in rng:
        for dy in rng:
            for dz in rng:
                if dx == dy == dz == 0:
                    continue
                if not use26 and (abs(dx)+abs(dy)+abs(dz) > 1):
                    continue
                offsets.append((dx, dy, dz))
    idx = np.arange(N**3).reshape(N, N, N)
    nbrs = [[] for _ in range(N**3)]
    for x in range(N):
        for y in range(N):
            for z in range(N):
                here = idx[x, y, z]
                for dx, dy, dz in offsets:
                    i, j, k = x+dx, y+dy, z+dz
                    if 0 <= i < N and 0 <= j < N and 0 <= k < N:
                        nbrs[here].append(idx[i, j, k])
    return [np.array(n, dtype=np.int32) for n in nbrs]

# ---------- Energy functions ----------

def local_energy(qs, node, nbrs):
    q_node = qs[node]
    q_nei = qs[nbrs[node]]
    return misori_angle(q_node, q_nei).sum()

def total_energy(qs, nbrs):
    return sum(local_energy(qs, i, nbrs) for i in range(len(qs))) / 2.0

# ---------- Simulated annealing with corrected ΔE ----------

def anneal(N=10, steps=10_000, maximise=False, seed=42):
    rng = np.random.default_rng(seed)
    nbrs = neighbour_table(N, use26=False)
    qs = random_quats(N**3)
    energy_trace = np.empty(steps, dtype=float)
    
    T0, Tf = 5.0, 1e-3
    alpha = (Tf / T0) ** (1/steps)
    T = T0
    E_total = total_energy(qs, nbrs)
    
    for s in range(steps):
        a, b = rng.integers(0, N**3, size=2)
        Ea1 = local_energy(qs, a, nbrs)
        Eb1 = local_energy(qs, b, nbrs)
        
        qs[a], qs[b] = qs[b].copy(), qs[a].copy()
        
        Ea2 = local_energy(qs, a, nbrs)
        Eb2 = local_energy(qs, b, nbrs)
        
        dE = ((Ea2 + Eb2) - (Ea1 + Eb1)) / 2.0  # correct for double counting
        if maximise:
            dE = -dE
        
        if dE < 0 or rng.random() < np.exp(-dE / T):
            E_total += dE
        else:
            qs[a], qs[b] = qs[b], qs[a]  # revert
        
        energy_trace[s] = E_total
        T *= alpha
    
    return energy_trace

# ---------- Run and plot ----------

# energy = anneal(N=10, steps=30_000, maximise=False)

# plt.figure(figsize=(7,4))
# plt.plot(energy)
# plt.xlabel("Iteration")
# plt.ylabel("Total misorientation (rad)")
# plt.title("Total misorientation vs iteration (corrected ΔE)")
# plt.tight_layout()
# plt.show()

def qmul(a, b):
    """Hamilton product a ⊗ b for quaternions a = [w,x,y,z]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def rotate_vec_by_quat(q, v):
    q = q / np.linalg.norm(q)
    vq = np.concatenate([[0,0],v])
    return qmul(qmul(q,vq),quat_conj(q))[1:]

i_hat = np.array([1.0,0.0,0.0])

rotated = np.array([rotate_vec_by_quat(q,i_hat) for q in hex_sym_quats()])

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(rotated[:,0],rotated[:,1],rotated[:,2], s=40, alpha=0.8)
ax.quiver(0,0,0,1,0,0,length=1)
plt.show()
