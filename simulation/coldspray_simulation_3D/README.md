# Cold Spray Deposition Simulation
The simulation followed the method proposed in Nault et al. *Multi-axis tool path optimization and deposition modeling for cold spray additive manufacturing*.

## How to run?
Run *simulation.m* in matlab

## Method

### Deposition Modeling

The method assume that the deposition follows a Gaussian distribution.

<img src="imgs/deposition_distribution.png">

It then used the assumption to "grow" the points of meshes of the model.

<img src="imgs/eq_1.png">

<img src="imgs/eq_1.png">


### Assumption (for now)
1. The robot motions are moveL with constant speed.
2. All part of molds are below the nozzle. (Holds for a small mold but probably not big object like a blade.)

## Result

### The mold

The test mold looks like below

<img src="imgs/mold.png" height="300">

### Simulation Result with Robot path

Robot path and mesh (cross section)

<img src="imgs/mold_and_robot_path_cross_section.png" height="300">

Simulation result

<img src="imgs/coldspray_simulation_result.png" height="300">

Calculation time

Mean: 7.23e-04
Std: 0.0022

<img src="imgs/duration.png" height="300">

