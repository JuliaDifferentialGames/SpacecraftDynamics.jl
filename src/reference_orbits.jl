"""
    reference_orbits.jl

Reference orbit management for HCW frame definition.

The HCW frame is defined by a "chief" spacecraft orbit, which can be either:
- A real spacecraft (deputy in formation with chief)
- A virtual reference point (no physical chief, just defines coordinate frame)

# Key Types
- `ReferenceOrbit`: Abstract type for reference orbit specifications
- `CircularOrbit`: Circular reference orbit (most common for HCW)
- `VirtualChief`: Virtual reference point with prescribed circular orbit
- `RealChief`: Physical chief spacecraft (future: coupled dynamics)

# References
S&J Chapter 13, Section 13.4 (Relative Motion Coordinate Frames)
"""

"""
    ReferenceOrbit{T}

Abstract type for reference orbit specifications.

Subtypes must implement:
- `mean_motion(ref::ReferenceOrbit)`: Get mean motion n [rad/s]
- `state_at_time(ref::ReferenceOrbit, t)`: Get ECI state at time t
"""
abstract type ReferenceOrbit{T<:Real} end

"""
    CircularOrbit{T}

Circular reference orbit specification.

Defines a circular orbit by semi-major axis and orbital elements.
This is the standard reference for linear HCW dynamics.

# Fields
- `a::T`: Semi-major axis [m]
- `i::T`: Inclination [rad]
- `Ω::T`: Right ascension of ascending node (RAAN) [rad]
- `ω::T`: Argument of periapsis [rad] (arbitrary for circular)
- `M0::T`: Mean anomaly at epoch [rad]
- `epoch::T`: Reference epoch [s]
- `μ::T`: Gravitational parameter [m³/s²]

# Derived Quantities
- Mean motion: n = √(μ/a³)
- Eccentricity: e = 0 (circular)
- Period: T = 2π/n

# Notes
For circular orbits, ω (argument of periapsis) is undefined. We set it to 0
by convention. The true anomaly ν equals the mean anomaly M for circular orbits.
"""
struct CircularOrbit{T<:Real} <: ReferenceOrbit{T}
    a::T       # Semi-major axis [m]
    i::T       # Inclination [rad]
    Ω::T       # RAAN [rad]
    ω::T       # Argument of periapsis [rad]
    M0::T      # Mean anomaly at epoch [rad]
    epoch::T   # Epoch time [s]
    μ::T       # Gravitational parameter [m³/s²]
    
    function CircularOrbit{T}(a, i, Ω, ω, M0, epoch, μ) where T<:Real
        @assert a > 0 "Semi-major axis must be positive"
        @assert 0 ≤ i ≤ π "Inclination must be in [0, π]"
        @assert μ > 0 "Gravitational parameter must be positive"
        
        new{T}(a, i, Ω, ω, M0, epoch, μ)
    end
end

"""
    CircularOrbit(; altitude, inclination=0.0, raan=0.0, 
                  mean_anomaly_0=0.0, epoch=0.0, μ=μ_EARTH)

Construct circular orbit from altitude and orbital elements.

# Keyword Arguments
- `altitude::Real`: Altitude above Earth's surface [m]
- `inclination::Real=0.0`: Inclination [rad]
- `raan::Real=0.0`: Right ascension of ascending node [rad]
- `mean_anomaly_0::Real=0.0`: Mean anomaly at epoch [rad]
- `epoch::Real=0.0`: Reference epoch [s]
- `μ::Real=μ_EARTH`: Gravitational parameter [m³/s²]

# Returns
- `CircularOrbit`: Circular orbit specification

# Example
```julia
# 400 km circular LEO, equatorial
orbit = CircularOrbit(altitude=400e3, inclination=0.0)

# 600 km SSO (sun-synchronous, i ≈ 97.8°)
orbit_sso = CircularOrbit(altitude=600e3, inclination=deg2rad(97.8))
```
"""
function CircularOrbit(;
    altitude::Real,
    inclination::Real=0.0,
    raan::Real=0.0,
    mean_anomaly_0::Real=0.0,
    epoch::Real=0.0,
    μ::Real=μ_EARTH
)
    T = promote_type(typeof(altitude), typeof(inclination), typeof(μ))
    
    a = T(R_EARTH + altitude)
    i = T(inclination)
    Ω = T(raan)
    ω = zero(T)  # Undefined for circular; set to 0
    M0 = T(mean_anomaly_0)
    t0 = T(epoch)
    μ_T = T(μ)
    
    return CircularOrbit{T}(a, i, Ω, ω, M0, t0, μ_T)
end

"""
    mean_motion(orbit::CircularOrbit)

Compute mean motion from circular orbit parameters.

# Arguments
- `orbit::CircularOrbit`: Orbit specification

# Returns
- `n::Real`: Mean motion [rad/s]
"""
function mean_motion(orbit::CircularOrbit)
    return sqrt(orbit.μ / orbit.a^3)
end

"""
    orbital_period(orbit::CircularOrbit)

Compute orbital period.

# Arguments
- `orbit::CircularOrbit`: Orbit specification

# Returns
- `T::Real`: Orbital period [s]
"""
function orbital_period(orbit::CircularOrbit)
    n = mean_motion(orbit)
    return 2π / n
end

"""
    mean_anomaly_at_time(orbit::CircularOrbit, t)

Compute mean anomaly at time t.

For circular orbits: M(t) = M₀ + n(t - t₀)

# Arguments
- `orbit::CircularOrbit`: Orbit specification
- `t::Real`: Time [s]

# Returns
- `M::Real`: Mean anomaly at time t [rad]
"""
function mean_anomaly_at_time(orbit::CircularOrbit, t::Real)
    n = mean_motion(orbit)
    M = orbit.M0 + n * (t - orbit.epoch)
    return mod2pi(M)  # Wrap to [0, 2π)
end

"""
    state_at_time(orbit::CircularOrbit, t)

Compute ECI position and velocity at time t.

Uses classical orbital elements to ECI transformation (S&J Chapter 13).
For circular orbits, this is simplified since e = 0.

# Arguments
- `orbit::CircularOrbit`: Orbit specification
- `t::Real`: Time [s]

# Returns
- `r_eci::SVector{3}`: Position in ECI frame [m]
- `v_eci::SVector{3}`: Velocity in ECI frame [m/s]

# Notes
Implements the transformation from orbital elements to ECI (S&J Eq 13.56-13.59).
For circular orbits, true anomaly ν = mean anomaly M.

# Reference
S&J Section 13.3, pages 710-714
"""
function state_at_time(orbit::CircularOrbit{T}, t::Real) where T
    # Mean anomaly at time t
    M = mean_anomaly_at_time(orbit, t)
    
    # For circular orbit, true anomaly = mean anomaly
    ν = M
    
    # Semi-major axis and mean motion
    a = orbit.a
    n = mean_motion(orbit)
    
    # Position and velocity in orbital plane (perifocal frame)
    # For circular orbit: r = a, v = na
    r_pf = a
    
    # Position in perifocal coordinates
    x_pf = r_pf * cos(ν)
    y_pf = r_pf * sin(ν)
    z_pf = zero(T)
    
    r_perifocal = SVector{3,T}(x_pf, y_pf, z_pf)
    
    # Velocity in perifocal coordinates
    # v = na for circular orbit
    v_mag = n * a
    vx_pf = -v_mag * sin(ν)
    vy_pf = v_mag * cos(ν)
    vz_pf = zero(T)
    
    v_perifocal = SVector{3,T}(vx_pf, vy_pf, vz_pf)
    
    # Rotation matrix from perifocal to ECI (S&J Eq 13.59)
    # R_ECI_PF = R₃(-Ω) R₁(-i) R₃(-ω)
    # For circular orbit, ω = 0, so: R_ECI_PF = R₃(-Ω) R₁(-i)
    
    cos_Ω = cos(orbit.Ω)
    sin_Ω = sin(orbit.Ω)
    cos_i = cos(orbit.i)
    sin_i = sin(orbit.i)
    
    # Combined rotation matrix
    R_ECI_PF = @SMatrix [
        cos_Ω  -sin_Ω*cos_i  sin_Ω*sin_i;
        sin_Ω   cos_Ω*cos_i -cos_Ω*sin_i;
        0.0     sin_i         cos_i
    ]
    
    # Transform to ECI
    r_eci = R_ECI_PF * r_perifocal
    v_eci = R_ECI_PF * v_perifocal
    
    return r_eci, v_eci
end

"""
    VirtualChief{T}

Virtual chief spacecraft for HCW frame definition.

A virtual chief is not a physical spacecraft, but a prescribed reference
trajectory that defines the HCW coordinate frame. Useful for:
- Formation flying where no single spacecraft is designated "chief"
- Trajectory tracking where reference is a desired path
- Theoretical studies with idealized reference orbits

# Fields
- `orbit::CircularOrbit{T}`: Reference orbit specification

# State Implications
For dynamics integration, virtual chief state is computed from orbit at each
time step. No deputy-chief coupling (chief unaffected by deputies).
"""
struct VirtualChief{T<:Real}
    orbit::CircularOrbit{T}
end

"""
    VirtualChief(; altitude, inclination=0.0, raan=0.0, 
                 mean_anomaly_0=0.0, epoch=0.0, μ=μ_EARTH)

Construct virtual chief from orbital parameters.

# Keyword Arguments
Same as `CircularOrbit` constructor.

# Example
```julia
# Virtual chief in 500 km LEO
chief = VirtualChief(altitude=500e3)
```
"""
function VirtualChief(;
    altitude::Real,
    inclination::Real=0.0,
    raan::Real=0.0,
    mean_anomaly_0::Real=0.0,
    epoch::Real=0.0,
    μ::Real=μ_EARTH
)
    orbit = CircularOrbit(
        altitude=altitude,
        inclination=inclination,
        raan=raan,
        mean_anomaly_0=mean_anomaly_0,
        epoch=epoch,
        μ=μ
    )
    
    return VirtualChief(orbit)
end

"""
    mean_motion(chief::VirtualChief)

Get mean motion of virtual chief orbit.
"""
mean_motion(chief::VirtualChief) = mean_motion(chief.orbit)

"""
    state_at_time(chief::VirtualChief, t)

Get virtual chief ECI state at time t.
"""
state_at_time(chief::VirtualChief, t::Real) = state_at_time(chief.orbit, t)

"""
    RealChief{T}

Real (physical) chief spacecraft in formation.

For scenarios where the chief is an actual spacecraft that can maneuver,
affecting the HCW frame definition. This introduces coupling between
chief and deputy dynamics.

# Fields
- `spacecraft::Spacecraft{T}`: Chief spacecraft specification
- `initial_orbit::CircularOrbit{T}`: Nominal orbit at t=0

# State Implications
Chief state must be propagated alongside deputy states. The HCW frame
rotates and translates based on actual chief motion, not a prescribed orbit.

# Notes
**NOT YET FULLY IMPLEMENTED**: Currently, chief dynamics are assumed
unaffected by deputies (one-way coupling). Full two-way coupling for
multi-agent games requires extending the state vector to include all agents
and carefully handling frame definitions.

This is a **future extension** for scenarios like:
- Cooperative formation reconfiguration (all agents maneuver)
- Multi-pursuer single-evader games (pursuers affect each other's frames)
"""
struct RealChief{T<:Real}
    spacecraft::Spacecraft{T}
    initial_orbit::CircularOrbit{T}
end

"""
    RealChief(spacecraft::Spacecraft, orbit::CircularOrbit)

Construct real chief from spacecraft and initial orbit.

# Arguments
- `spacecraft::Spacecraft`: Chief spacecraft specification
- `orbit::CircularOrbit`: Initial orbit at epoch

# Notes
For coupled multi-agent dynamics, chief state becomes part of the
overall system state vector.
"""
function RealChief(spacecraft::Spacecraft{T}, orbit::CircularOrbit{T}) where T
    return RealChief{T}(spacecraft, orbit)
end

"""
    GameScenarioSetup{T}

Complete setup for a differential game scenario.

Combines reference orbit, spacecraft configurations, and initial conditions
for multi-agent differential games.

# Fields
- `reference::Union{VirtualChief{T}, RealChief{T}}`: Chief/reference definition
- `deputies::Vector{Spacecraft{T}}`: Deputy spacecraft (players)
- `initial_states::Vector{SpacecraftState{T}}`: Initial states for each deputy
- `time_horizon::T`: Game time horizon [s]

# Notes
This is a convenience structure for setting up game problems. Not required
for dynamics evaluation, but useful for organizing benchmark scenarios.
"""
struct GameScenarioSetup{T<:Real}
    reference::Union{VirtualChief{T}, RealChief{T}}
    deputies::Vector{Spacecraft{T}}
    initial_states::Vector{SpacecraftState{T}}
    time_horizon::T
    
    function GameScenarioSetup{T}(
        reference::Union{VirtualChief{T}, RealChief{T}},
        deputies::Vector{Spacecraft{T}},
        initial_states::Vector{SpacecraftState{T}},
        time_horizon::T
    ) where T<:Real
        n_deputies = length(deputies)
        @assert length(initial_states) == n_deputies "Must provide initial state for each deputy"
        @assert time_horizon > 0 "Time horizon must be positive"
        
        new{T}(reference, deputies, initial_states, time_horizon)
    end
end

"""
    GameScenarioSetup(reference, deputies, initial_states, time_horizon)

Construct game scenario setup.

# Arguments
- `reference`: Virtual or real chief
- `deputies::Vector{Spacecraft}`: Deputy spacecraft
- `initial_states::Vector{SpacecraftState}`: Initial conditions
- `time_horizon::Real`: Game duration [s]
"""
function GameScenarioSetup(
    reference::Union{VirtualChief{T}, RealChief{T}},
    deputies::Vector{Spacecraft{T}},
    initial_states::Vector{SpacecraftState{T}},
    time_horizon::Real
) where T<:Real
    return GameScenarioSetup{T}(reference, deputies, initial_states, T(time_horizon))
end

"""
    create_pursuit_evasion_scenario(; altitude=500e3, pursuer_thrust=2.0,
                                    evader_thrust=1.5, separation=1000.0,
                                    time_horizon=600.0)

Create a standard pursuit-evasion game scenario.

Pursuer starts behind evader, both in HCW frame of virtual chief.

# Keyword Arguments
- `altitude::Real=500e3`: Reference orbit altitude [m]
- `pursuer_thrust::Real=2.0`: Pursuer max thrust [N]
- `evader_thrust::Real=1.5`: Evader max thrust [N]
- `separation::Real=1000.0`: Initial along-track separation [m]
- `time_horizon::Real=600.0`: Game duration [s]

# Returns
- `GameScenarioSetup`: Complete scenario specification

# Example
```julia
scenario = create_pursuit_evasion_scenario(
    altitude=400e3,
    pursuer_thrust=3.0,
    evader_thrust=2.0
)
```
"""
function create_pursuit_evasion_scenario(;
    altitude::Real=500e3,
    pursuer_thrust::Real=2.0,
    evader_thrust::Real=1.5,
    separation::Real=1000.0,
    time_horizon::Real=600.0
)
    T = Float64
    
    # Virtual chief
    chief = VirtualChief(altitude=altitude)
    
    # Pursuer and evader spacecraft
    pursuer = default_research_spacecraft(
        mass=100.0,
        thruster_force=pursuer_thrust,
        attitude_enabled=false  # Simplified dynamics for pursuit-evasion
    )
    
    evader = default_research_spacecraft(
        mass=100.0,
        thruster_force=evader_thrust,
        attitude_enabled=false
    )
    
    deputies = [pursuer, evader]
    
    # Initial states: evader ahead along y-axis (along-track)
    # Pursuer at origin
    pursuer_state = SpacecraftState(
        @SVector(zeros(T, 3)),      # r
        @SVector(zeros(T, 3)),      # v
        @SVector(zeros(T, 3)),      # σ (unused)
        @SVector(zeros(T, 3))       # ω (unused)
    )
    
    # Evader ahead in along-track direction
    evader_state = SpacecraftState(
        @SVector([0.0, separation, 0.0]),  # r
        @SVector(zeros(T, 3)),              # v
        @SVector(zeros(T, 3)),              # σ (unused)
        @SVector(zeros(T, 3))               # ω (unused)
    )
    
    initial_states = [pursuer_state, evader_state]
    
    return GameScenarioSetup(chief, deputies, initial_states, time_horizon)
end


