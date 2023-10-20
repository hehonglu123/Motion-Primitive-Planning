# Motion Optimization of Motoman

Motion optimization scripts and results for Motoman robots.

## Motion Capture System

The algorithms and codes are designed to be independent of the specific motion capture system being used. Currently, we are using Optitrack.

### Optitrack

#### RR Driver
Please refer to [this repository](https://github.com/robotraconteur-contrib/optitrack_mocap_robotraconteur_driver) for the RR driver.

### Phasesoace

#### RR Driver
Please refer to [this repository](https://github.com/robotraconteur-contrib/phasespace_mocap_robotraconteur_driver) for the RR driver.

## Run motion program updates

### Motion Optimization using tool sensing (motion capture systems)

```
python3 final_grad_real_mocap.py
```

### Motion Optimization using calibrated kinematic model

```
python3 final_grad_real_mocap_verified.py
```