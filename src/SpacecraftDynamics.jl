"""
    SpacecraftDynamics

Spacecraft dynamics simulator for differential game benchmarking.
Implements dynamics models following Schaub & Junkins, "Analytical Mechanics 
of Space Systems" (4th ed).

Primary state representation: Hill-Clohessy-Wiltshire (HCW) frame
Attitude parameterization: Modified Rodrigues Parameters (MRPs)

# Features
- Linear and nonlinear HCW relative dynamics
- MRP-based attitude kinematics and dynamics
- Configurable thruster and reaction wheel actuators
- Integration with DifferentialGamesBase.jl
- Reference orbit management (real or virtual chief)

# Assumptions
- Near-circular reference orbits (eccentricity ≈ 0)
- Point-mass gravity (J2+ perturbations available as future extension)
- Rigid body spacecraft (no flexible modes)

# References
Schaub, H., & Junkins, J. L. (2014). Analytical Mechanics of Space Systems 
(4th ed.). AIAA Education Series.
"""
module SpacecraftDynamics

using LinearAlgebra
using StaticArrays
using Rotations  # For DCM utilities where needed
using SatelliteToolbox: R0, M0_TO_SEC  # Physical constants

# Re-export commonly used LinearAlgebra functions
using LinearAlgebra: norm, dot, cross, I

# Core types
export Spacecraft, SpacecraftState
export ReferenceOrbit, CircularOrbit
export ThrusterConfiguration, ReactionWheelConfiguration
export ActuatorConfiguration

# Dynamics types
export AbstractSpacecraftDynamics
export LinearHCWDynamics, NonlinearHCWDynamics
export CoupledDynamics, DecoupledDynamics

# Callbacks 
export create_mrp_switching_callback, create_mrp_continuous_switching_callback
export mrp_shadow_switch!

# Attitude utilities
export mrp_kinematics, mrp_to_dcm, dcm_to_mrp
export mrp_shadow_switch, mrp_norm
export skew

# Orbital utilities
export hcw_mean_motion, hcw_state_matrix, hcw_control_matrix

# Dynamics functions
export spacecraft_dynamics!, spacecraft_dynamics
export translational_dynamics, rotational_dynamics, attitude_dynamics

# Game integration
export create_separable_dynamics

# Physical constants (using SatelliteToolbox conventions)
const μ_EARTH = 3.986004418e14  # Earth gravitational parameter [m³/s²]
const R_EARTH = 6378.137e3      # Earth equatorial radius [m]

include("attitude.jl")
include("frames.jl")
include("orbital.jl")
include("actuators.jl")
include("spacecraft.jl")
include("reference_orbits.jl")
include("dynamics_integration.jl")
include("callbacks.jl")


end # module