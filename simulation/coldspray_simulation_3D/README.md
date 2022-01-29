# Cold Spray Deposition Simulation
The simulation followed the method proposed in Nault et al. *Multi-axis tool path optimization and deposition modeling for cold spray additive manufacturing*.

## How to run?
Run *simulation.m* in matlab

## Method

### Deposition Modeling

The method assume that the deposition follows a Gaussian distribution.

![deposition model](imgs/deposition_distribution.png)

It then used the assumption to "grow" the points of meshes of the model.

![eq1](imgs/eq_1.png)

![eq2](imgs/eq_2.png)

### Assumption (for now)
1. The robot motions are moveL with constant speed.
2. All part of molds are below the nozzle. (Holds for a small mold but probably not big object like a blade.)

## Result

### The mold

The test mold looks like below

![mold](imgs/mold.png)

### Simulation Result with Robot path

Robot path and mesh (cross section)

![robot_path](imgs/mold_and_robot_path_cross_section.png)

Simulation result

![result](imgs/coldspray_simulation_result.png)

Calculation time

Mean: 7.23e-04
Std: 0.0022

![duration](imgs/duration.png)

