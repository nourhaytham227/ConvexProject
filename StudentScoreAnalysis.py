import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

np.random.seed(1)
n = 50 #students
m=3  #subjects

original_scores = np.random.normal(loc=75, scale=10, size=(n, m)) #generating scores using normal distribution
original_scores = np.clip(original_scores, 0, 100) 

x= cp.Variable((n, m)) #solution vector (optimized scores)

cost= cp.sum_squares(x - original_scores) #cost function  

constraints = [
    #each subject constraints
    x[:,0]>=60, 
    x[:,1]>=20,
    x[:,2]>=45,
    cp.var(original_scores) - cp.var(x) >= 10,   
]

prob= cp.Problem(cp.Minimize(cost), constraints)
prob.solve()

optimized_scores = x.value

print("original mean: ", np.mean(original_scores))
print("original variance: ", np.var(original_scores))
print("------------------------------")
print("optimized mean: ", np.mean(optimized_scores))
print("optimized value ", prob.value)
print("optimized variance: ", np.var(optimized_scores))

H = 2 * np.eye(n * m) #The hessian matrix for our problem
eigenvalues = np.linalg.eigvals(H)
if np.all(eigenvalues >= 0):
    print("All eigenvalues of hessian are nonnegative so the function is convex")
else:
    print("some eigenvalues of hessian are negative so the function is not convex")
    
print("Problem.is_dcp():", prob.is_dcp())
print("since the problem is DCP and hessian is positive semidefinite, the optimization problem is convex.")
print("------------------------------")
    
data = optimized_scores.copy()
hull = ConvexHull(data)
#modifying convexity using non linear transformation

non_convex_scores = optimized_scores +  0.05 * (optimized_scores - 75)**3


#restoring convexity by adding constraints

restored_scores = non_convex_scores.copy()

restored_scores = np.clip(restored_scores, 50, 95)
restored_scores[:,0] = np.clip(restored_scores[:,0], 60, 95)  # Subject 1
restored_scores[:,1] = np.clip(restored_scores[:,1], 20, 90)  # Subject 2
restored_scores[:,2] = np.clip(restored_scores[:,2], 45, 95)  #Subject 3

print("Original Convex:")
print("Mean:", np.mean(optimized_scores))
print("Variance:", np.var(optimized_scores))
print("------------------------------")
print("Modified Convex:")
print("Mean:", np.mean(non_convex_scores))
print("Variance:", np.var(non_convex_scores))
print("------------------------------")
print("Restored Convex:")
print("Mean:", np.mean(restored_scores))
print("Variance:", np.var(restored_scores))


# Optimized dataset
fig1 = plt.figure(figsize=(7, 6))
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(optimized_scores[:,0], optimized_scores[:,1], optimized_scores[:,2], c='blue', s=50)
faces = [optimized_scores[simplex] for simplex in ConvexHull(optimized_scores).simplices]
poly3d = Poly3DCollection(faces, alpha=0.2, facecolor='red')
ax1.add_collection3d(poly3d)

ax1.set_xlabel("Subject 1")
ax1.set_ylabel("Subject 2")
ax1.set_zlabel("Subject 3")
ax1.set_title("Optimized Dataset Convex Hull")
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.set_zlim(0, 100)
plt.tight_layout()
plt.show(block=False)

# Restored dataset
fig2 = plt.figure(figsize=(7, 6))
ax2 = fig2.add_subplot(111, projection='3d')

ax2.scatter(restored_scores[:,0], restored_scores[:,1], restored_scores[:,2], c='blue', s=50)
faces = [restored_scores[simplex] for simplex in ConvexHull(restored_scores).simplices]
poly3d = Poly3DCollection(faces, alpha=0.2, facecolor='red')
ax2.add_collection3d(poly3d)

ax2.set_xlabel("Subject 1")
ax2.set_ylabel("Subject 2")
ax2.set_zlabel("Subject 3")
ax2.set_title("Restored Dataset Convex Hull")
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.set_zlim(0, 100)
plt.tight_layout()
plt.show(block=False)

# plot 
fig2 = plt.figure(figsize=(18, 5))

for i, (scores, title) in enumerate(zip(
    [optimized_scores, non_convex_scores, restored_scores],
    ["Optimized (Convex)", "Modified (Non-Convex)", "Restored (Convex)"]
)):
    ax = fig2.add_subplot(1, 3, i+1, projection='3d')
    ax.scatter(scores[:,0], scores[:,1], scores[:,2], c='blue', s=50)
    ax.set_xlabel('Subject 1')
    ax.set_ylabel('Subject 2')
    ax.set_zlabel('Subject 3')
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

plt.show()
