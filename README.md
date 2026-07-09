# Mujoco_BLISS

# Loading Model into MuJoCo
To load the model and simulate the dynamics without any specific ``controller'', run the following in a terminal:
```
python -m mujoco.viewer --mjcf={PATH_TO_REPO}/Mujoco_BLISS/STLs/scene.xml
```
Note that you will need to pause the simulation, reset the simulation, and then move the `base_z` joint position to some positive value to view it properly.

