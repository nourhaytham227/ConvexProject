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
    cp.std(original_scores) - cp.std(x) >= 1,   
]

prob= cp.Problem(cp.Minimize(cost), constraints)
prob.solve()

optimized_scores = x.value

print("original mean: ", np.mean(original_scores))
print("original std: ", np.std(original_scores))
print("------------------------------")
print("optimized mean: ", np.mean(optimized_scores))
print("optimized value ", prob.value)
print("optimized std: ", np.std(optimized_scores))

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
print("Optimized: points:", len(data), "hull vertices:", len(hull.vertices))
if len(hull.vertices) < len(data):
    print("most points dont lie on the hull, so the dataset is not convex.")
print("------------------------------")
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
print("Standard Deviation:", np.std(optimized_scores))
print("------------------------------")
print("Modified Convex:")
print("Mean:", np.mean(non_convex_scores))
print("Standard Deviation:", np.std(non_convex_scores))
print("------------------------------")
print("Restored Convex:")
print("Mean:", np.mean(restored_scores))
print("Standard Deviation:", np.std(restored_scores))


data = optimized_scores.copy()
hull = ConvexHull(data)

fig1 = plt.figure(figsize=(7, 6))
ax = fig1.add_subplot(111, projection='3d')

# Scatter points
ax.scatter(data[:,0], data[:,1], data[:,2], c='blue', s=50)


faces = [data[simplex] for simplex in hull.simplices]
poly3d = Poly3DCollection(faces, alpha=0.2, facecolor='red')
ax.add_collection3d(poly3d)

ax.set_xlabel("Subject 1")
ax.set_ylabel("Subject 2")
ax.set_zlabel("Subject 3")
ax.set_title("Optimized Dataset Convex Hull\nHull Vertices: " + str(len(hull.vertices)) + "/" + str(len(data)))

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

plt.tight_layout()
plt.show(block=False)

# plot 
fig2 = plt.figure(figsize=(18, 5))

for i, (scores, title) in enumerate(zip(
    [optimized_scores, non_convex_scores, restored_scores],
    ["Original (Convex)", "Modified (Non-Convex)", "Restored (Convex)"]
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


