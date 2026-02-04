"""
    actuators.jl

Thruster and reaction wheel actuator models for spacecraft control.

# Actuator Types
- **Thrusters**: Produce force, continuous or impulsive
- **Reaction Wheels**: Produce torque via momentum exchange

# Design Philosophy
Actuators are defined by their configuration (placement, orientation) and
operational characteristics (thrust levels, saturation, etc.). The spacecraft
dynamics integrate these configurations to compute total forces and torques.

# References
S&J Chapter 4 (Rigid Body Dynamics with Actuators)
Wie, B. (2008). Space Vehicle Dynamics and Control (2nd ed.). AIAA.
"""

"""
    ThrusterConfiguration{T,N}

Configuration of N thrusters on a spacecraft.

Each thruster is defined by:
- Position in body frame (lever arm for torque)
- Direction in body frame (unit vector)
- Maximum thrust magnitude

# Type Parameters
- `T<:Real`: Numeric type
- `N::Int`: Number of thrusters

# Fields
- `positions::SVector{N, SVector{3,T}}`: Thruster positions in body frame [m]
- `directions::SVector{N, SVector{3,T}}`: Thrust direction unit vectors (body frame)
- `max_thrust::SVector{N,T}`: Maximum thrust per thruster [N]
- `min_impulse_bit::T`: Minimum impulse bit for impulsive thrusters [N⋅s] (0 for continuous)
- `continuous::Bool`: True for continuous thrust, false for impulsive

# Notes
For continuous thrusters, control input is thrust fraction ∈ [0, 1] per thruster.
For impulsive thrusters, control is bang-bang: 0 or 1 (future implementation).
"""
struct ThrusterConfiguration{T<:Real, N}
    positions::SVector{N, SVector{3,T}}      # [m]
    directions::SVector{N, SVector{3,T}}     # unit vectors
    max_thrust::SVector{N, T}                 # [N]
    min_impulse_bit::T                        # [N⋅s]
    continuous::Bool
    
    function ThrusterConfiguration(
        positions::SVector{N, SVector{3,T}},
        directions::SVector{N, SVector{3,T}},
        max_thrust::SVector{N,T},
        min_impulse_bit::T,
        continuous::Bool
    ) where {T<:Real, N}
        # Validate direction vectors are unit vectors
        for (i, d) in enumerate(directions)
            d_norm = norm(d)
            if abs(d_norm - 1.0) > 1e-6
                @warn "Thruster $i direction not normalized: |d| = $d_norm. Normalizing."
                directions = setindex(directions, d / d_norm, i)
            end
        end
        
        new{T,N}(positions, directions, max_thrust, min_impulse_bit, continuous)
    end
end

"""
    ThrusterConfiguration(positions, directions, max_thrust; 
                         min_impulse_bit=0.0, continuous=true)

Construct thruster configuration with validation.

# Arguments
- `positions`: Vector of thruster positions in body frame
- `directions`: Vector of thrust direction unit vectors
- `max_thrust`: Vector of maximum thrust values (or scalar for uniform)
- `min_impulse_bit::Real=0.0`: Minimum impulse bit [N⋅s]
- `continuous::Bool=true`: Continuous vs impulsive mode
"""
function ThrusterConfiguration(
    positions::AbstractVector{<:AbstractVector{T}},
    directions::AbstractVector{<:AbstractVector{T}},
    max_thrust::Union{AbstractVector{T}, T};
    min_impulse_bit::T=zero(T),
    continuous::Bool=true
) where T<:Real
    N = length(positions)
    @assert length(directions) == N "positions and directions must have same length"
    
    pos_sv = SVector{N}([SVector{3,T}(p) for p in positions])
    dir_sv = SVector{N}([SVector{3,T}(d) for d in directions])
    
    if max_thrust isa AbstractVector
        @assert length(max_thrust) == N "max_thrust length must match number of thrusters"
        thrust_sv = SVector{N,T}(max_thrust)
    else
        thrust_sv = SVector{N,T}(fill(max_thrust, N))
    end
    
    return ThrusterConfiguration(pos_sv, dir_sv, thrust_sv, min_impulse_bit, continuous)
end

"""
    default_research_thrusters(; max_thrust=1.0, continuous=true)

Create default "research satellite" thruster configuration.

Six thrusters at center of mass, aligned with body axes (±x, ±y, ±z).
Suitable for theoretical studies with full 3-DOF translational control authority.

# Keyword Arguments
- `max_thrust::Real=1.0`: Maximum thrust per thruster [N]
- `continuous::Bool=true`: Continuous vs impulsive mode

# Returns
- `ThrusterConfiguration{Float64,6}`: Six-thruster configuration

# Notes
This is a simplified configuration for game-theoretic studies. Real spacecraft
have thrusters offset from CoM and canted to provide both force and torque.
"""
function default_research_thrusters(; max_thrust::Real=1.0, continuous::Bool=true)
    T = typeof(float(max_thrust))
    
    # All thrusters at center of mass (no torque coupling)
    positions = [
        @SVector(zeros(T, 3)),
        @SVector(zeros(T, 3)),
        @SVector(zeros(T, 3)),
        @SVector(zeros(T, 3)),
        @SVector(zeros(T, 3)),
        @SVector(zeros(T, 3))
    ]
    
    # Aligned with body axes
    directions = [
        @SVector([1.0, 0.0, 0.0]),   # +x
        @SVector([-1.0, 0.0, 0.0]),  # -x
        @SVector([0.0, 1.0, 0.0]),   # +y
        @SVector([0.0, -1.0, 0.0]),  # -y
        @SVector([0.0, 0.0, 1.0]),   # +z
        @SVector([0.0, 0.0, -1.0])   # -z
    ]
    
    return ThrusterConfiguration(
        positions, directions, max_thrust;
        min_impulse_bit=zero(T), continuous=continuous
    )
end

"""
    compute_thruster_wrench(config::ThrusterConfiguration, u_thrusters)

Compute total force and torque from thruster firing commands.

# Arguments
- `config::ThrusterConfiguration{T,N}`: Thruster configuration
- `u_thrusters::AbstractVector`: Thruster commands (N-element, values ∈ [0,1])

# Returns
- `F_total::SVector{3,T}`: Total force in body frame [N]
- `τ_total::SVector{3,T}`: Total torque in body frame [N⋅m]

# Notes
For continuous thrusters: u[i] ∈ [0, 1] is thrust fraction
Force from thruster i: F_i = u[i] * max_thrust[i] * direction[i]
Torque from thruster i: τ_i = position[i] × F_i
"""
function compute_thruster_wrench(
    config::ThrusterConfiguration{T,N},
    u_thrusters::AbstractVector
) where {T,N}
    @assert length(u_thrusters) == N "Control dimension mismatch: expected $N, got $(length(u_thrusters))"
    
    F_total = @SVector zeros(T, 3)
    τ_total = @SVector zeros(T, 3)
    
    for i in 1:N
        # Clamp control to [0, 1]
        u_i = clamp(u_thrusters[i], zero(T), one(T))
        
        # Thrust magnitude and force vector
        thrust_mag = u_i * config.max_thrust[i]
        F_i = thrust_mag * config.directions[i]
        
        # Torque from this thruster
        τ_i = cross(config.positions[i], F_i)
        
        # Accumulate
        F_total += F_i
        τ_total += τ_i
    end
    
    return F_total, τ_total
end

"""
    ReactionWheelConfiguration{T,N}

Configuration of N reaction wheels on a spacecraft.

Each wheel is defined by:
- Spin axis in body frame (unit vector)
- Maximum torque output
- Maximum momentum storage

# Type Parameters
- `T<:Real`: Numeric type
- `N::Int`: Number of reaction wheels

# Fields
- `axes::SVector{N, SVector{3,T}}`: Wheel spin axes (unit vectors, body frame)
- `max_torque::SVector{N,T}`: Maximum torque per wheel [N⋅m]
- `max_momentum::SVector{N,T}`: Maximum momentum storage [N⋅m⋅s]
- `momentum::SVector{N,T}`: Current stored momentum (state) [N⋅m⋅s]

# Notes
Reaction wheels exchange angular momentum with spacecraft:
τ_wheel_on_body = -ḣ_wheel

For now, we ignore saturation (max_momentum) to simplify game formulations.
Future: implement momentum management and desaturation logic.
"""
struct ReactionWheelConfiguration{T<:Real, N}
    axes::SVector{N, SVector{3,T}}     # unit vectors
    max_torque::SVector{N,T}           # [N⋅m]
    max_momentum::SVector{N,T}         # [N⋅m⋅s]
    
    function ReactionWheelConfiguration(
        axes::SVector{N, SVector{3,T}},
        max_torque::SVector{N,T},
        max_momentum::SVector{N,T}
    ) where {T<:Real, N}
        # Validate axes are unit vectors
        for (i, a) in enumerate(axes)
            a_norm = norm(a)
            if abs(a_norm - 1.0) > 1e-6
                @warn "Reaction wheel $i axis not normalized: |a| = $a_norm. Normalizing."
                axes = setindex(axes, a / a_norm, i)
            end
        end
        
        new{T,N}(axes, max_torque, max_momentum)
    end
end

"""
    ReactionWheelConfiguration(axes, max_torque, max_momentum)

Construct reaction wheel configuration with validation.

# Arguments
- `axes`: Vector of spin axis unit vectors
- `max_torque`: Vector of maximum torques (or scalar for uniform)
- `max_momentum`: Vector of maximum momenta (or scalar for uniform)
"""
function ReactionWheelConfiguration(
    axes::AbstractVector{<:AbstractVector{T}},
    max_torque::Union{AbstractVector{T}, T},
    max_momentum::Union{AbstractVector{T}, T}
) where T<:Real
    N = length(axes)
    
    axes_sv = SVector{N}([SVector{3,T}(a) for a in axes])
    
    if max_torque isa AbstractVector
        @assert length(max_torque) == N "max_torque length must match number of wheels"
        torque_sv = SVector{N,T}(max_torque)
    else
        torque_sv = SVector{N,T}(fill(max_torque, N))
    end
    
    if max_momentum isa AbstractVector
        @assert length(max_momentum) == N "max_momentum length must match number of wheels"
        momentum_sv = SVector{N,T}(max_momentum)
    else
        momentum_sv = SVector{N,T}(fill(max_momentum, N))
    end
    
    return ReactionWheelConfiguration(axes_sv, torque_sv, momentum_sv)
end

"""
    default_research_wheels(; max_torque=0.1, max_momentum=10.0)

Create default "research satellite" reaction wheel configuration.

Three wheels aligned with body axes (x, y, z), providing full 3-DOF
rotational control authority.

# Keyword Arguments
- `max_torque::Real=0.1`: Maximum torque per wheel [N⋅m]
- `max_momentum::Real=10.0`: Maximum momentum storage [N⋅m⋅s]

# Returns
- `ReactionWheelConfiguration{Float64,3}`: Three-wheel configuration

# Notes
Saturation limits are defined but not enforced in current implementation.
"""
function default_research_wheels(; max_torque::Real=0.1, max_momentum::Real=10.0)
    T = typeof(float(max_torque))
    
    axes = [
        @SVector([1.0, 0.0, 0.0]),  # x-axis wheel
        @SVector([0.0, 1.0, 0.0]),  # y-axis wheel
        @SVector([0.0, 0.0, 1.0])   # z-axis wheel
    ]
    
    return ReactionWheelConfiguration(axes, max_torque, max_momentum)
end

"""
    compute_wheel_torque(config::ReactionWheelConfiguration, u_wheels)

Compute total torque from reaction wheel commands.

# Arguments
- `config::ReactionWheelConfiguration{T,N}`: Wheel configuration
- `u_wheels::AbstractVector`: Wheel torque commands (N-element, values ∈ [-1,1])

# Returns
- `τ_total::SVector{3,T}`: Total torque in body frame [N⋅m]

# Notes
For reaction wheels: u[i] ∈ [-1, 1] is torque fraction (bidirectional)
Torque from wheel i: τ_i = u[i] * max_torque[i] * axis[i]
"""
function compute_wheel_torque(
    config::ReactionWheelConfiguration{T,N},
    u_wheels::AbstractVector
) where {T,N}
    @assert length(u_wheels) == N "Control dimension mismatch: expected $N, got $(length(u_wheels))"
    
    τ_total = @SVector zeros(T, 3)
    
    for i in 1:N
        # Clamp control to [-1, 1]
        u_i = clamp(u_wheels[i], -one(T), one(T))
        
        # Torque from this wheel
        τ_i = u_i * config.max_torque[i] * config.axes[i]
        
        # Accumulate
        τ_total += τ_i
    end
    
    return τ_total
end

"""
    ActuatorConfiguration{T}

Combined actuator configuration for a spacecraft.

Wraps both thruster and reaction wheel configurations. Provides unified
interface for computing total wrench (force + torque) from control inputs.

# Fields
- `thrusters::Union{ThrusterConfiguration{T}, Nothing}`: Thruster config (if equipped)
- `wheels::Union{ReactionWheelConfiguration{T}, Nothing}`: Wheel config (if equipped)

# Control Input Structure
If both thrusters and wheels equipped:
    u = [u_thrusters..., u_wheels...]
If only thrusters:
    u = [u_thrusters...]
If only wheels:
    u = [u_wheels...]
"""
struct ActuatorConfiguration{T<:Real}
    thrusters::Union{ThrusterConfiguration{T}, Nothing}
    wheels::Union{ReactionWheelConfiguration{T}, Nothing}
    
    function ActuatorConfiguration{T}(
        thrusters::Union{ThrusterConfiguration{T}, Nothing},
        wheels::Union{ReactionWheelConfiguration{T}, Nothing}
    ) where T<:Real
        if isnothing(thrusters) && isnothing(wheels)
            error("Must have at least thrusters or reaction wheels")
        end
        new{T}(thrusters, wheels)
    end
end

"""
    ActuatorConfiguration(; thrusters=nothing, wheels=nothing)

Construct actuator configuration.

# Keyword Arguments
- `thrusters::Union{ThrusterConfiguration, Nothing}=nothing`
- `wheels::Union{ReactionWheelConfiguration, Nothing}=nothing`

At least one of thrusters or wheels must be provided.
"""
function ActuatorConfiguration(;
    thrusters::Union{ThrusterConfiguration{T}, Nothing}=nothing,
    wheels::Union{ReactionWheelConfiguration{T}, Nothing}=nothing
) where T<:Real
    return ActuatorConfiguration{T}(thrusters, wheels)
end

# Helper to infer type for mixed construction
function ActuatorConfiguration(
    thrusters::Union{ThrusterConfiguration{T1}, Nothing},
    wheels::Union{ReactionWheelConfiguration{T2}, Nothing}
) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)
    return ActuatorConfiguration{T}(thrusters, wheels)
end

"""
    control_dimension(config::ActuatorConfiguration)

Get total control input dimension.

# Returns
- `n_u::Int`: Total number of control inputs
"""
function control_dimension(config::ActuatorConfiguration)
    n_u = 0
    
    if !isnothing(config.thrusters)
        N_thrusters = length(config.thrusters.max_thrust)
        n_u += N_thrusters
    end
    
    if !isnothing(config.wheels)
        N_wheels = length(config.wheels.max_torque)
        n_u += N_wheels
    end
    
    return n_u
end

"""
    compute_wrench(config::ActuatorConfiguration, u)

Compute total force and torque from all actuators.

# Arguments
- `config::ActuatorConfiguration{T}`: Actuator configuration
- `u::AbstractVector`: Full control input vector

# Returns
- `F::SVector{3,T}`: Total force in body frame [N]
- `τ::SVector{3,T}`: Total torque in body frame [N⋅m]

# Notes
Splits control vector appropriately between thrusters and wheels.
"""
function compute_wrench(config::ActuatorConfiguration{T}, u::AbstractVector) where T
    F = @SVector zeros(T, 3)
    τ = @SVector zeros(T, 3)
    
    idx = 1
    
    # Thrusters contribute both force and torque
    if !isnothing(config.thrusters)
        N_thrusters = length(config.thrusters.max_thrust)
        u_thrusters = u[idx:idx+N_thrusters-1]
        
        F_thrust, τ_thrust = compute_thruster_wrench(config.thrusters, u_thrusters)
        F += F_thrust
        τ += τ_thrust
        
        idx += N_thrusters
    end
    
    # Reaction wheels contribute only torque
    if !isnothing(config.wheels)
        N_wheels = length(config.wheels.max_torque)
        u_wheels = u[idx:idx+N_wheels-1]
        
        τ_wheels = compute_wheel_torque(config.wheels, u_wheels)
        τ += τ_wheels
        
        idx += N_wheels
    end
    
    return F, τ
end

"""
    default_research_actuators(; thruster_force=1.0, wheel_torque=0.1, 
                               wheel_momentum=10.0, continuous=true)

Create default "research satellite" actuator configuration.

Combines default thrusters and reaction wheels for full 6-DOF control.

# Keyword Arguments
- `thruster_force::Real=1.0`: Max thrust per thruster [N]
- `wheel_torque::Real=0.1`: Max torque per wheel [N⋅m]
- `wheel_momentum::Real=10.0`: Max momentum per wheel [N⋅m⋅s]
- `continuous::Bool=true`: Continuous vs impulsive thrusters

# Returns
- `ActuatorConfiguration`: Combined configuration (6 thrusters + 3 wheels)
"""
function default_research_actuators(;
    thruster_force::Real=1.0,
    wheel_torque::Real=0.1,
    wheel_momentum::Real=10.0,
    continuous::Bool=true
)
    thrusters = default_research_thrusters(;
        max_thrust=thruster_force,
        continuous=continuous
    )
    
    wheels = default_research_wheels(;
        max_torque=wheel_torque,
        max_momentum=wheel_momentum
    )
    
    return ActuatorConfiguration(thrusters=thrusters, wheels=wheels)
end