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
using DiffEqCallbacks  
import SciMLBase 
using StaticArrays
using Rotations  
using SatelliteToolbox
using DifferentialGamesBase

# Re-export commonly used LinearAlgebra functions
using LinearAlgebra: norm, dot, cross, I

# Includes 
include("attitude.jl")
include("frames.jl")
include("orbital.jl")
include("actuators.jl")
include("spacecraft.jl")
include("reference_orbits.jl")
include("dynamics_integration.jl")
include("callbacks.jl")

# Physical constants (using SatelliteToolbox conventions)
const μ_EARTH = 3.986004418e14  # Earth gravitational parameter [m³/s²]
const R_EARTH = 6378.137e3      # Earth equatorial radius [m]

# Export constants
export μ_EARTH, R_EARTH

# Core types
export Spacecraft, SpacecraftState
export ReferenceOrbit, CircularOrbit
export ThrusterConfiguration, ReactionWheelConfiguration
export ActuatorConfiguration
export VirtualChief, RealChief
export GameScenarioSetup

# Dynamics types
export AbstractSpacecraftDynamics
export LinearHCWDynamics, NonlinearHCWDynamics
export CoupledDynamics, DecoupledDynamics
export SpacecraftGameDynamics

# Attitude utilities
export mrp_kinematics, mrp_to_dcm, dcm_to_mrp
export mrp_shadow_switch, mrp_shadow_switch!, should_switch_shadow
export mrp_norm
export skew
export attitude_dynamics

# Orbital utilities
export hcw_mean_motion, hcw_state_matrix, hcw_control_matrix
export mean_motion, orbital_period, mean_anomaly_at_time, state_at_time
export two_body_acceleration

# Frame transformations
export eci_to_hcw, hcw_to_eci, trajectory_hcw_to_eci
export body_to_hcw, hcw_to_body
export orbital_frame_rate

# Actuator utilities
export default_research_thrusters, default_research_wheels
export default_research_actuators
export compute_thruster_wrench, compute_wheel_torque, compute_wrench
export control_dimension

# Spacecraft utilities
export default_research_spacecraft
export state_dimension, to_vector
export translational_dynamics, rotational_dynamics

# Dynamics functions
export spacecraft_dynamics!, spacecraft_dynamics, spacecraft_dynamics_wrapper

# Game integration
export create_separable_dynamics, create_game_parameters
export player_state_dimension, player_control_dimension
export total_state_dimension, total_control_dimension
export state_partition, control_partition
export stack_states, unstack_states, stack_controls, unstack_controls
export extract_player_trajectory

# Scenario creation
export create_pursuit_evasion_scenario

# Callbacks
export create_mrp_switching_callback, create_mrp_continuous_switching_callback


end # module