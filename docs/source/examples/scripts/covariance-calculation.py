import matplotlib.pyplot as plt

from kite import Scene

# Assume we have a existing kite.Scene with defined quadtree parametrized
scene = Scene.load("acquila_2016.yml")

ax = plt.gca()
# Inspect the noise data which is used to calculate the covariance
ax.imshow(scene.covariance.noise_data)
plt.show()

# Inspect the focal-point (quick mode) covariance matrix
ax.imshow(scene.covariance.covariance_matrix_focal)

# Inspect the full covariance matrix
ax.imshow(scene.covariance.covariance_matrix)

# Get the full weight matrix
ax.imshow(scene.covariance.weight_matrix)

# Get the covariance and weight between two leafes
leaf_1 = scene.quadtree.leaves[0]
leaf_2 = scene.quadtree.leaves[0]

scene.covariance.getLeafCovariance(leaf_1, leaf_2)
scene.covariance.getLeafWeight(leaf_1, leaf_2)
