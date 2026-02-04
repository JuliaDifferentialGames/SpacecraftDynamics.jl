"""
    spacecraft.jl

Spacecraft type and configuration for differential game simulations.

Combines physical properties (mass, inertia), actuator configurations,
and orbital dynamics models into a unified spacecraft representation.

# Key Type
- `Spacecraft`: Complete spacecraft specification
- `SpacecraftState`: State representation (position, velocity, attitude, angular velocity)

# Design Philosophy
Spacecraft acts as a container for all physical parameters needed to evaluate
dynamics. The dynamics functions take Spacecraft as a parameter along with
current state and control inputs.
"""

"""
    SpacecraftState{T}

Complete state of a spacecraft in the HCW frame.

# Fields
- `r::SVector{3,T}`: Position in HCW frame [m]
- `v::SVector{3,T}`: Velocity in HCW frame [m/s]
- `σ::SVector{3,T}`: Attitude (MRP) representing rotation from HCW to body
- `ω::SVector{3,T}`: Angular velocity in body frame [rad/s]

# Notes
Total state dimension: 12 (3 + 3 + 3 + 3)

For decoupled translational dynamics (attitude_enabled=false), only r and v are used.
"""
struct SpacecraftState{T<:Real}
    r::SVector{3,T}    # Position in HCW [m]
    v::SVector{3,T}    # Velocity in HCW [m/s]
    σ::SVector{3,T}    # MRP (HCW to body rotation)
    ω::SVector{3,T}    # Angular velocity in body [rad/s]
end

"""
    SpacecraftState(r, v, σ, ω)

Construct spacecraft state from components.

# Arguments
- `r`: Position [m] (3-element)
- `v`: Velocity [m/s] (3-element)
- `σ`: MRP attitude (3-element)
- `ω`: Angular velocity [rad/s] (3-element)
"""
function SpacecraftState(
    r::AbstractVector{T},
    v::AbstractVector{T},
    σ::AbstractVector{T},
    ω::AbstractVector{T}
) where T<:Real
    return SpacecraftState(
        SVector{3,T}(r),
        SVector{3,T}(v),
        SVector{3,T}(σ),
        SVector{3,T}(ω)
    )
end

"""
    SpacecraftState(x::AbstractVector)

Construct from state vector [r; v; σ; ω].

# Arguments
- `x::AbstractVector`: 12-element state vector
"""
function SpacecraftState(x::AbstractVector{T}) where T<:Real
    @assert length(x) == 12 "State vector must be 12-dimensional"
    
    return SpacecraftState(
        SVector{3,T}(x[1:3]),   # r
        SVector{3,T}(x[4:6]),   # v
        SVector{3,T}(x[7:9]),   # σ
        SVector{3,T}(x[10:12])  # ω
    )
end

"""
    to_vector(state::SpacecraftState)

Convert state to vector representation [r; v; σ; ω].

# Returns
- `::SVector{12}`: State as vector
"""
function to_vector(state::SpacecraftState{T}) where T
    return SVector{12,T}(state.r..., state.v..., state.σ..., state.ω...)
end

"""
    Spacecraft{T}

Complete spacecraft specification for dynamics simulation.

# Fields
- `mass::T`: Spacecraft mass [kg]
- `inertia::Union{SMatrix{3,3,T}, Diagonal{T}}`: Inertia tensor in body frame [kg⋅m²]
- `actuators::ActuatorConfiguration{T}`: Thruster and wheel configuration
- `attitude_enabled::Bool`: Whether to propagate attitude dynamics
- `orbital_dynamics::Symbol`: Dynamics model (:linear_hcw or :nonlinear_hcw)

# Dynamics Models
- `:linear_hcw`: Linear Hill-Clohessy-Wiltshire (S&J Eq 13.95-13.97)
- `:nonlinear_hcw`: Nonlinear relative motion via two-body propagation

# State Dimension
- attitude_enabled=true: 12 (r, v, σ, ω)
- attitude_enabled=false: 6 (r, v only)

# Control Dimension
Determined by actuator configuration. See `control_dimension(actuators)`.
"""
struct Spacecraft{T<:Real}
    mass::T
    inertia::Union{SMatrix{3,3,T}, Diagonal{T,SVector{3,T}}}
    actuators::ActuatorConfiguration{T}
    attitude_enabled::Bool
    orbital_dynamics::Symbol
    
    function Spacecraft{T}(
        mass::T,
        inertia::Union{SMatrix{3,3,T}, Diagonal{T,SVector{3,T}}},
        actuators::ActuatorConfiguration{T},
        attitude_enabled::Bool,
        orbital_dynamics::Symbol
    ) where T<:Real
        @assert mass > 0 "Mass must be positive"
        @assert orbital_dynamics ∈ (:linear_hcw, :nonlinear_hcw) "Invalid dynamics model"
        
        # Validate inertia is positive definite
        if inertia isa SMatrix
            eigvals = eigen(inertia).values
            @assert all(eigvals .> 0) "Inertia tensor must be positive definite"
        else
            @assert all(inertia.diag .> 0) "Inertia diagonal must be positive"
        end
        
        new{T}(mass, inertia, actuators, attitude_enabled, orbital_dynamics)
    end
end

"""
    Spacecraft(; mass, inertia, actuators, attitude_enabled=true, 
               orbital_dynamics=:linear_hcw)

Construct spacecraft with keyword arguments.

# Keyword Arguments
- `mass::Real`: Spacecraft mass [kg]
- `inertia`: Inertia tensor (3×3 matrix or Diagonal) [kg⋅m²]
- `actuators::ActuatorConfiguration`: Actuator configuration
- `attitude_enabled::Bool=true`: Whether to include attitude dynamics
- `orbital_dynamics::Symbol=:linear_hcw`: Dynamics model

# Example
```julia
sc = Spacecraft(
    mass = 100.0,
    inertia = Diagonal([10.0, 12.0, 8.0]),
    actuators = default_research_actuators(),
    attitude_enabled = true,
    orbital_dynamics = :linear_hcw
)
```
"""
function Spacecraft(;
    mass::Real,
    inertia::Union{AbstractMatrix, Diagonal},
    actuators::ActuatorConfiguration{T},
    attitude_enabled::Bool=true,
    orbital_dynamics::Symbol=:linear_hcw
) where T<:Real
    mass_T = convert(T, mass)
    
    # Convert inertia to appropriate type
    if inertia isa Diagonal
        inertia_T = Diagonal(SVector{3,T}(inertia.diag))
    else
        @assert size(inertia) == (3,3) "Inertia must be 3×3"
        inertia_T = SMatrix{3,3,T}(inertia)
    end
    
    return Spacecraft{T}(mass_T, inertia_T, actuators, attitude_enabled, orbital_dynamics)
end

"""
    state_dimension(sc::Spacecraft)

Get state dimension for this spacecraft.

# Returns
- `n_x::Int`: State dimension (12 if attitude enabled, 6 otherwise)
"""
function state_dimension(sc::Spacecraft)
    return sc.attitude_enabled ? 12 : 6
end

"""
    control_dimension(sc::Spacecraft)

Get control dimension for this spacecraft.

# Returns
- `n_u::Int`: Control dimension (from actuator configuration)
"""
function control_dimension(sc::Spacecraft)
    return control_dimension(sc.actuators)
end

"""
    default_research_spacecraft(; mass=100.0, inertia_scale=10.0,
                                thruster_force=1.0, wheel_torque=0.1,
                                attitude_enabled=true, 
                                orbital_dynamics=:linear_hcw)

Create a default "research satellite" for game-theoretic studies.

Combines default actuators with simple mass properties. Suitable for
theoretical analysis and rapid prototyping.

# Keyword Arguments
- `mass::Real=100.0`: Spacecraft mass [kg]
- `inertia_scale::Real=10.0`: Diagonal inertia values [kg⋅m²]
- `thruster_force::Real=1.0`: Max thrust per thruster [N]
- `wheel_torque::Real=0.1`: Max torque per wheel [N⋅m]
- `attitude_enabled::Bool=true`: Include attitude dynamics
- `orbital_dynamics::Symbol=:linear_hcw`: Dynamics model

# Returns
- `Spacecraft`: Default research spacecraft

# Example
```julia
# Simple 6-DOF spacecraft for pursuit-evasion
pursuer = default_research_spacecraft(mass=80.0, thruster_force=2.0)
evader = default_research_spacecraft(mass=100.0, thruster_force=1.5)
```
"""
function default_research_spacecraft(;
    mass::Real=100.0,
    inertia_scale::Real=10.0,
    thruster_force::Real=1.0,
    wheel_torque::Real=0.1,
    attitude_enabled::Bool=true,
    orbital_dynamics::Symbol=:linear_hcw
)
    # Simple diagonal inertia (spherical approximation)
    inertia = Diagonal([inertia_scale, inertia_scale, inertia_scale])
    
    # Default actuators
    actuators = default_research_actuators(
        thruster_force=thruster_force,
        wheel_torque=wheel_torque
    )
    
    return Spacecraft(
        mass=mass,
        inertia=inertia,
        actuators=actuators,
        attitude_enabled=attitude_enabled,
        orbital_dynamics=orbital_dynamics
    )
end

"""
    translational_dynamics(sc::Spacecraft, r, v, F_hcw, n)

Compute translational dynamics: [ṙ; v̇]

# Arguments
- `sc::Spacecraft`: Spacecraft specification
- `r::SVector{3}`: Position in HCW [m]
- `v::SVector{3}`: Velocity in HCW [m/s]
- `F_hcw::SVector{3}`: Force in HCW frame [N]
- `n::Real`: Mean motion of reference orbit [rad/s]

# Returns
- `ṙ::SVector{3}`: Position derivative [m/s]
- `v̇::SVector{3}`: Velocity derivative [m/s²]

# Notes
Implements linear HCW equations (S&J Eq 13.95-13.97):
```
ẍ - 2nẏ - 3n²x = Fₓ/m
ÿ + 2nẋ = Fᵧ/m  
z̈ + n²z = Fᵧ/m
```
"""
function translational_dynamics(
    sc::Spacecraft{T},
    r::SVector{3,T},
    v::SVector{3,T},
    F_hcw::SVector{3,T},
    n::T
) where T
    # Position derivative (trivial)
    ṙ = v
    
    # Acceleration from HCW equations
    x, y, z = r
    vx, vy, vz = v
    Fx, Fy, Fz = F_hcw
    
    n² = n^2
    inv_m = 1 / sc.mass
    
    # S&J Eq 13.95-13.97
    ax = 3n²*x + 2n*vy + Fx*inv_m
    ay = -2n*vx + Fy*inv_m
    az = -n²*z + Fz*inv_m
    
    v̇ = SVector{3,T}(ax, ay, az)
    
    return ṙ, v̇
end

"""
    rotational_dynamics(sc::Spacecraft, σ, ω, τ_body)

Compute rotational dynamics: [σ̇; ω̇]

# Arguments
- `sc::Spacecraft`: Spacecraft specification
- `σ::SVector{3}`: MRP attitude
- `ω::SVector{3}`: Angular velocity in body frame [rad/s]
- `τ_body::SVector{3}`: Torque in body frame [N⋅m]

# Returns
- `σ̇::SVector{3}`: MRP derivative [1/s]
- `ω̇::SVector{3}`: Angular acceleration [rad/s²]

# Notes
Combines MRP kinematics (S&J Eq 3.162) with Euler's equation (S&J Eq 4.11).
"""
function rotational_dynamics(
    sc::Spacecraft{T},
    σ::SVector{3,T},
    ω::SVector{3,T},
    τ_body::SVector{3,T}
) where T
    # MRP kinematics
    σ̇ = mrp_kinematics(σ, ω)
    
    # Euler's rotational equation
    ω̇ = attitude_dynamics(ω, τ_body, sc.inertia)
    
    return σ̇, ω̇
end

"""
    spacecraft_dynamics!(ẋ, x, u, p, t)

In-place spacecraft dynamics for DifferentialEquations.jl integration.

# Arguments
- `ẋ::AbstractVector`: Output state derivative (modified in-place)
- `x::AbstractVector`: Current state
- `u::AbstractVector`: Control input
- `p`: Parameters (NamedTuple with spacecraft, n, etc.)
- `t::Real`: Time

# Parameter Structure
`p` should be a NamedTuple with fields:
- `spacecraft::Spacecraft`: Spacecraft specification
- `n::Real`: Mean motion [rad/s] (for linear HCW)
- Additional fields as needed for specific scenarios

# Notes
This is the standard DifferentialEquations.jl signature for controlled systems.
Use with `ODEProblem` and callbacks for game simulations.
"""
function spacecraft_dynamics!(
    ẋ::AbstractVector,
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Real
)
    sc = p.spacecraft
    n = p.n
    
    if sc.attitude_enabled
        # Full 12-state dynamics
        @assert length(x) == 12 "State must be 12-dimensional for attitude-enabled dynamics"
        
        r = SVector{3}(x[1:3])
        v = SVector{3}(x[4:6])
        σ = SVector{3}(x[7:9])
        ω = SVector{3}(x[10:12])
        
        # Compute wrench from actuators
        F_body, τ_body = compute_wrench(sc.actuators, u)
        
        # Transform force from body to HCW frame
        F_hcw = body_to_hcw(F_body, σ)
        
        # Translational dynamics
        ṙ, v̇ = translational_dynamics(sc, r, v, F_hcw, n)
        
        # Rotational dynamics
        σ̇, ω̇ = rotational_dynamics(sc, σ, ω, τ_body)
        
        # Pack derivatives
        ẋ[1:3] .= ṙ
        ẋ[4:6] .= v̇
        ẋ[7:9] .= σ̇
        ẋ[10:12] .= ω̇
        
    else
        # Translational-only 6-state dynamics
        @assert length(x) == 6 "State must be 6-dimensional for translational-only dynamics"
        
        r = SVector{3}(x[1:3])
        v = SVector{3}(x[4:6])
        
        # For translational-only, assume control is force in HCW frame
        # This requires thrusters only (no wheels) and assumes they're commanded
        # in HCW frame or that attitude control keeps body aligned with HCW
        @assert isnothing(sc.actuators.wheels) "Translational-only mode requires no reaction wheels"
        
        # Compute thrust (assuming directions aligned with HCW for simplicity)
        F_thrust, _ = compute_thruster_wrench(sc.actuators.thrusters, u)
        
        # Translational dynamics
        ṙ, v̇ = translational_dynamics(sc, r, v, F_thrust, n)
        
        # Pack derivatives
        ẋ[1:3] .= ṙ
        ẋ[4:6] .= v̇
    end
    
    return nothing
end

"""
    spacecraft_dynamics(x, u, p, t)

Out-of-place spacecraft dynamics (returns ẋ instead of modifying in-place).

# Arguments
- `x::AbstractVector`: Current state
- `u::AbstractVector`: Control input
- `p`: Parameters (NamedTuple)
- `t::Real`: Time

# Returns
- `ẋ::SVector`: State derivative

# Notes
Functional (out-of-place) version for use with StaticArrays and game solvers
that expect functional dynamics.
"""
function spacecraft_dynamics(
    x::AbstractVector{T},
    u::AbstractVector,
    p,
    t::Real=0.0
) where T
    n_x = length(x)
    ẋ = Vector{T}(undef, n_x)
    spacecraft_dynamics!(ẋ, x, u, p, t)
    return SVector{n_x,T}(ẋ)
end