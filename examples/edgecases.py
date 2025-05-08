#%%
# edge_case_affinity_demo.py  –  run with   python edge_case_affinity_demo.py
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for projection='3d')

# ---------------------------- constants -----------------------------
ALPHA, BETA, GAMMA, ETA, KAPPA, LAMBDA = 4, 0.05, 0.05, 0.5, 0.00, 0.0
L0 = 0.03

# ---------------------- analytic helper funcs -----------------------
def seg(p, v, length):
    """return (p1,p2) centred on p, direction v (unit), given length"""
    v = v/np.linalg.norm(v)
    return p - v*length/2.0, p + v*length/2.0

def geom_terms(a1, a2, b1, b2):
    """θ , d_min , δ , r"""
    # angle
    v1, v2 = (a2-a1), (b2-b1)
    v1 /= np.linalg.norm(v1); v2 /= np.linalg.norm(v2)
    theta = np.arccos(np.clip(np.abs(v1@v2), 0, 1))

    # shortest line-to-line distance (both directions)
    def pt_line(p,q1,q2):
        v = q2-q1
        return np.linalg.norm(np.cross(v, p-q1))/np.linalg.norm(v)
    d_min = min(pt_line(a1,b1,b2), pt_line(a2,b1,b2),
                pt_line(b1,a1,a2), pt_line(b2,a1,a2))

    # mean end-point-to-segment distance
    def pt_seg(p,q1,q2):
        v, w = q2-q1, p-q1
        t = np.clip((w@v)/(v@v), 0, 1)
        proj = q1 + t*v
        return np.linalg.norm(p-proj)
    delta = (pt_seg(a1,b1,b2)+pt_seg(a2,b1,b2)+
             pt_seg(b1,a1,a2)+pt_seg(b2,a1,a2))/4.0

    # length ratio
    l1, l2 = np.linalg.norm(a2-a1), np.linalg.norm(b2-b1)
    r = min(l1,l2)/max(l1,l2)
    return theta, d_min, delta, r, l1, l2

def affinity(p1,p2,q1,q2, g=0.0, d_px=0.0):
    θ,dmin,δ,r,l1,l2 = geom_terms(p1,p2,q1,q2)
    angle  = np.exp(-ALPHA*θ*θ)
    off    = np.exp(-BETA *dmin*dmin)
    ends   = np.exp(-GAMMA*δ*δ)
    length = np.exp(-ETA  *(1-r))
    edge   = np.exp(-KAPPA*g*g)
    space  = np.exp(-LAMBDA*d_px*d_px)
    return angle*off*ends*length*edge*space, (θ,dmin,δ,r,l1,l2)

# -------------------------- edge scenarios --------------------------
cases = {
    "1  · same line":      seg(np.r_[0,0,0],  np.r_[1,0,0], 2),
    "2  · V-shape 60°":    (seg(np.r_[-1,0,0],np.r_[1,0,0],2)[0],
                            seg(np.r_[-1,0,0],np.r_[1,0,0],2)[1],
                            seg(np.r_[ 1,0,0],np.r_[0.5,0.866,0],2)[0],
                            seg(np.r_[ 1,0,0],np.r_[0.5,0.866,0],2)[1]),
    "3  · tiny spur":      (seg(np.r_[0,0,0], np.r_[1,0,0],4)[0],
                            seg(np.r_[0,0,0], np.r_[1,0,0],4)[1],
                            seg(np.r_[1,0,0], np.r_[0,1,0],0.5)[0],
                            seg(np.r_[1,0,0], np.r_[0,1,0],0.5)[1]),
    "4  · parallel far":   seg(np.r_[0,0,0],  np.r_[1,0,0], 2)[:2] +
                            seg(np.r_[0,5,0], np.r_[1,0,0], 2),
    "5  · X-cross":        seg(np.r_[-1,0,0], np.r_[1,0,0], 2)[:2] +
                            seg(np.r_[ 0,-1,0],np.r_[0,1,0], 2),
    "6  · parallel close": seg(np.r_[0,0,0],  np.r_[1,0,0], 2)[:2] +
                            seg(np.r_[0,0.4,0],np.r_[1,0,0], 2),
}

# ---------------------------- plotting ------------------------------
fig = plt.figure(figsize=(15,10))
for k,(name,coords) in enumerate(cases.items(),1):
    if len(coords)==2: p1,p2 = coords;          q1,q2 = coords  # same line case
    else:              p1,p2,q1,q2 = coords
    aff, (θ,dmin,δ,r,l1,l2) = affinity(p1,p2,q1,q2)

    ax = fig.add_subplot(2,3,k,projection="3d")
    ax.plot(*zip(p1,p2),lw=3);  ax.plot(*zip(q1,q2),lw=3,color='orange')
    ax.set_title(name)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_zlim(-3,3)
    ax.set_xlabel("L"); ax.set_ylabel("a"); ax.set_zlabel("b")

    txt = (f"θ = {np.degrees(θ):.1f}°\n"
           f"d_min = {dmin:.2f}\n"
           f"δ = {δ:.2f}\n"
           f"r = {r:.2f}\n"
           f"ℓ₁ = {l1:.2f}  ℓ₂ = {l2:.2f}\n"
           f"A = {aff:.3f}")
    ax.text2D(0.02,0.05,txt,transform=ax.transAxes,fontsize=8,
              bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=0.7))

plt.suptitle("Principal-colour line segments – affinity edge cases",fontsize=16)
plt.tight_layout();  plt.show()
# %%
