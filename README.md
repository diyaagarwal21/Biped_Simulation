# I. Simulation Code for BILLIE

## 0. Create virtual environment
TODO: make .yml package

## 1. Clone Github
```
git clone https://github.com/diyaagarwal21/Biped_Simulation.git
```

## 2. Run simulation
Data used in the pipeline was generated with an offline trajectory optimization framework. All data can be found in the Data folder and has shape (N, nx), with N being the number of optimization nodes and nx the state dimension.
Note that there is existing data already in the folder, which can be used. 

Simulation can be run with:
```
python -m billie_implementation
```

# II. Quick overview of files
- "Controllers" contains all code and information needed for the biped controllers.
- "Controllers/URDF" contains the biped urdfs (urdf = Unified Robot Description Format). These are used to build a pinocchio model of the biped, which is being used by the controller (in simulation and hardware).
- "Controllers/pd_controller.py", "Controllers/linear_quadratic_regulizer_controller.py", "Controllers/feedback_linearization_controller.py" contain the different controller classes. The one being deployed on the hardware and was most reliable was the PD controller.
- "Data" contains the reference trajecotries
- "STLs" contains the .xml files needed to build the biped model in MuJoCo.
- "billie_implementation.py" this is the main code, it initializes the controller, MuJoCo model, imports the data (applies some transformations) and runs the simulation.

# III. Current state of the code base
The current code base has some known issues, the biggest one is the mismatch between the pinocchio model and the real life biped. This has consequences for the accuracy of the generated reference trajectory as well as for the robustness of the controller (mostly for FL and LQR). Issues were partially ommited by not using the reference torques during deployment. In future, an adaptation of the code base from a pinned model to a free floating base model is needed to ensure reliable sim to real transfer.
