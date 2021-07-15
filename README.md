# Motion-Primitive-Planning
To achieve high-precision high-speed collision free robot manipulator trajectory planning using robot motion primitives only.

## Multi-dimension piece-wise fitting
`pwlfmd.py`
A curve from turbine blade is extracted to `Curve.csv`, 3D piecewise fitting is performed on the curve.
![Fitting Results](result.png)
* Fit with given break points: `fit_with_breaks(breaks)`
* Find break points automatically based on slope change: `break_slope(data_range,min_threshold,max_threshold,step_size)`
* Fit automatically with above funtions given maximum tolerance threshold: `fit_under_error(max_error)`

