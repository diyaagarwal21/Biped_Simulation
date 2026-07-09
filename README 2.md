# Mujoco-Example
This repository demonstrates the python bindings for MuJoCo (documentation: https://mujoco.readthedocs.io/en/stable/python.html)

## Installation
To install this on your computer, you should only have to run the following
``` 
pip install mujoco
```
Note: DO NOT try and install `mujoco_py`, this is an outdated and no-longer-maintained version of mujoco python.

After installing `mujoco`, you can test your installation by opening the standalone app:
```
python -m mujoco.viewer
```
## Running the example script
To test a specific model using the native viewer (no controller will be run, just the physics simulation), run the script
```
python mujoco-test.py
```
Note: If you have a Mac, you will need to run ``mjpython mujoco-test.py`` instead.

## Loading in a new robot model
Check out the example models provided through MuJoCo: https://mujoco.readthedocs.io/en/stable/models.html

Try downloading any of these models and replacing the path in line 6 of `mujoco-test.py`.

## Create your own robot model
To simulate the template xml run the python script `mujoco-custom.py`. You can then experiment with editing this xml to see how robots are defined in MuJoCo.