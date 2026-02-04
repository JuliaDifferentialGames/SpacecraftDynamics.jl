"""
    orbital.jl

Hill-Clohessy-Wiltshire (HCW) relative orbital dynamics.

Implements both linear and nonlinear relative motion equations for spacecraft
formation flying and proximity operations.

# Dynamics Models
- **Linear HCW**: Assumes circular reference orbit, linearized about chief
- **Nonlinear**: Full two-body dynamics with relative state differencing
- **Future**: Tschauner-Hempel for eccentric reference orbits

# References
S&J Chapter 13, Section 13.5 (Linear Relative Motion)
Clohessy, W. H., & Wiltshire, R. S. (1960). Terminal guidance system for 
satellite rendezvous. Journal of the Aerospace Sciences, 27(9), 653-658.
"""

"""
    hcw_mean_motion(a; μ=μ_EARTH)

Compute mean motion n for a circular orbit of semi-major axis a.

Following Kepler's third law: n = √(μ/a³)

# Arguments
- `a::Real`: Semi-major axis of reference orbit [m]
- `μ::Real=μ_EARTH`: Gravitational parameter [m³/s²]

# Returns
- `n::Real`: Mean motion [rad/s]

# Reference
S&J Eq 13.95, page 731
"""
function hcw_mean_motion(a::Real; μ::Real=μ_EARTH)
    return sqrt(μ / a^3)
end

"""
    hcw_mean_motion(r_chief::AbstractVector; μ=μ_EARTH)

Compute mean motion from chief position vector (assumes circular orbit).

# Arguments
- `r_chief::AbstractVector`: Chief position in ECI [m]
- `μ::Real=μ_EARTH`: Gravitational parameter [m³/s²]

# Returns
- `n::Real`: Mean motion [rad/s]
"""
function hcw_mean_motion(r_chief::AbstractVector; μ::Real=μ_EARTH)
    a = norm(r_chief)  # For circular orbit, a ≈ |r|
    return hcw_mean_motion(a; μ=μ)
end

"""
    hcw_state_matrix(n)

Construct the continuous-time state matrix A for linear HCW dynamics.

The linearized HCW equations are: ẋ = Ax + Bu

Following S&J Eq 13.95-13.97:
```
ẍ - 2nẏ - 3n²x = Fₓ/m
ÿ + 2nẋ = Fᵧ/m  
z̈ + n²z = Fᵧ/m
```

State vector: [x, y, z, ẋ, ẏ, ż]ᵀ

# Arguments
- `n::Real`: Mean motion [rad/s]

# Returns
- `A::SMatrix{6,6}`: Continuous-time state matrix

# Reference
S&J Section 13.5.1, Eq 13.95-13.97, page 731
"""
function hcw_state_matrix(n::T) where T<:Real
    # State: [x, y, z, vx, vy, vz]
    # Dynamics in block form:
    # [ṙ]   [0₃ₓ₃  I₃ₓ₃] [r]
    # [v̇] = [A₂₁   A₂₂ ] [v]
    
    n² = n^2
    
    A = @SMatrix [
        0.0  0.0  0.0   1.0   0.0   0.0;
        0.0  0.0  0.0   0.0   1.0   0.0;
        0.0  0.0  0.0   0.0   0.0   1.0;
        3n²  0.0  0.0   0.0   2n    0.0;
        0.0  0.0  0.0  -2n    0.0   0.0;
        0.0  0.0 -n²    0.0   0.0   0.0
    ]
    
    return A
end

"""
    hcw_control_matrix(m)

Construct the continuous-time control matrix B for linear HCW dynamics.

The control input is the force in HCW frame: u = [Fₓ, Fᵧ, Fᵧ]ᵀ

# Arguments
- `m::Real`: Spacecraft mass [kg]

# Returns
- `B::SMatrix{6,3}`: Continuous-time control matrix

# Notes
B = [0₃ₓ₃; (1/m)I₃ₓ₃] maps forces to accelerations.
"""
function hcw_control_matrix(m::T) where T<:Real
    inv_m = 1 / m
    
    B = @SMatrix [
        0.0   0.0   0.0;
        0.0   0.0   0.0;
        0.0   0.0   0.0;
        inv_m 0.0   0.0;
        0.0   inv_m 0.0;
        0.0   0.0   inv_m
    ]
    
    return B
end

"""
    LinearHCWDynamics{T}

Linear Hill-Clohessy-Wiltshire dynamics for relative motion.

Assumes circular reference orbit with mean motion n.

# Fields
- `n::T`: Mean motion [rad/s]
- `m::T`: Spacecraft mass [kg]
- `A::SMatrix{6,6,T}`: State matrix
- `B::SMatrix{6,3,T}`: Control matrix

# State Vector
[x, y, z, vₓ, vᵧ, vᵧ]ᵀ where positions in [m], velocities in [m/s]

# Control Vector
[Fₓ, Fᵧ, Fᵧ]ᵀ in HCW frame [N]
"""
struct LinearHCWDynamics{T<:Real}
    n::T                    # Mean motion [rad/s]
    m::T                    # Mass [kg]
    A::SMatrix{6,6,T}       # State matrix
    B::SMatrix{6,3,T}       # Control matrix
    
    function LinearHCWDynamics(n::T, m::T) where T<:Real
        A = hcw_state_matrix(n)
        B = hcw_control_matrix(m)
        new{T}(n, m, A, B)
    end
end

"""
    LinearHCWDynamics(; altitude, mass, μ=μ_EARTH)

Construct linear HCW dynamics from orbital altitude.

# Keyword Arguments
- `altitude::Real`: Altitude above Earth's surface [m]
- `mass::Real`: Spacecraft mass [kg]
- `μ::Real=μ_EARTH`: Gravitational parameter [m³/s²]
"""
function LinearHCWDynamics(; altitude::Real, mass::Real, μ::Real=μ_EARTH)
    a = R_EARTH + altitude
    n = hcw_mean_motion(a; μ=μ)
    return LinearHCWDynamics(n, mass)
end

"""
    (dyn::LinearHCWDynamics)(x, u, t)

Evaluate linear HCW dynamics: ẋ = Ax + Bu

# Arguments
- `x::AbstractVector`: State [x, y, z, vₓ, vᵧ, vᵧ] (6-element)
- `u::AbstractVector`: Control [Fₓ, Fᵧ, Fᵧ] (3-element)
- `t::Real`: Time (unused for time-invariant dynamics)

# Returns
- `ẋ::SVector{6}`: State derivative
"""
function (dyn::LinearHCWDynamics)(x::AbstractVector, u::AbstractVector, t::Real=0.0)
    x_sv = SVector{6}(x)
    u_sv = SVector{3}(u)
    return dyn.A * x_sv + dyn.B * u_sv
end

"""
    NonlinearHCWDynamics{T}

Nonlinear relative dynamics via full two-body propagation and differencing.

Propagates chief and deputy separately under two-body dynamics, then computes
relative state. More accurate than linear HCW but computationally expensive.

# Fields
- `μ::T`: Gravitational parameter [m³/s²]
- `m::T`: Deputy spacecraft mass [kg]

# State Vector
[x, y, z, vₓ, vᵧ, vᵧ, xc, yc, zc, vxc, vyc, vzc]ᵀ
First 6 elements: deputy relative state in HCW
Elements 7-12: chief absolute state in ECI (for frame definition)

# Control Vector
[Fₓ, Fᵧ, Fᵧ]ᵀ in HCW frame [N]

# Notes
This is more expensive than linear HCW but exact for two-body dynamics.
For games requiring high fidelity over extended horizons, use this.
For rapid prototyping and theoretical analysis, use LinearHCWDynamics.
"""
struct NonlinearHCWDynamics{T<:Real}
    μ::T    # Gravitational parameter [m³/s²]
    m::T    # Deputy mass [kg]
    
    function NonlinearHCWDynamics(μ::T, m::T) where T<:Real
        new{T}(μ, m)
    end
end

"""
    NonlinearHCWDynamics(; mass, μ=μ_EARTH)

Construct nonlinear HCW dynamics.

# Keyword Arguments
- `mass::Real`: Deputy spacecraft mass [kg]
- `μ::Real=μ_EARTH`: Gravitational parameter [m³/s²]
"""
function NonlinearHCWDynamics(; mass::Real, μ::Real=μ_EARTH)
    return NonlinearHCWDynamics(μ, mass)
end

"""
    two_body_acceleration(r, μ)

Compute two-body gravitational acceleration: a = -μr/|r|³

# Arguments
- `r::SVector{3}`: Position vector [m]
- `μ::Real`: Gravitational parameter [m³/s²]

# Returns
- `a::SVector{3}`: Acceleration [m/s²]

# Reference
S&J Eq 13.1, page 700
"""
function two_body_acceleration(r::SVector{3,T}, μ::T) where T
    r_norm = norm(r)
    return -(μ / r_norm^3) * r
end

"""
    (dyn::NonlinearHCWDynamics)(x, u, t)

Evaluate nonlinear relative dynamics.

Propagates both deputy and chief under two-body gravity, applies control
to deputy, then computes relative state dynamics in rotating HCW frame.

# Arguments
- `x::AbstractVector`: State [r_rel (HCW), v_rel (HCW), r_chief (ECI), v_chief (ECI)] (12-element)
- `u::AbstractVector`: Control [Fₓ, Fᵧ, Fᵧ] in HCW frame (3-element)
- `t::Real`: Time (unused)

# Returns
- `ẋ::SVector{12}`: State derivative

# Notes
This implements the exact nonlinear relative dynamics by:
1. Transforming relative state to absolute ECI
2. Applying two-body gravity to both spacecraft
3. Applying control force to deputy
4. Computing relative acceleration in rotating HCW frame
"""
function (dyn::NonlinearHCWDynamics)(x::AbstractVector, u::AbstractVector, t::Real=0.0)
    # Extract states
    r_rel = SVector{3}(x[1:3])      # Relative position in HCW
    v_rel = SVector{3}(x[4:6])      # Relative velocity in HCW
    r_chief = SVector{3}(x[7:9])    # Chief position in ECI
    v_chief = SVector{3}(x[10:12])  # Chief velocity in ECI
    
    u_sv = SVector{3}(u)            # Control in HCW frame
    
    # Convert relative state to absolute deputy state in ECI
    r_deputy, v_deputy = hcw_to_eci(r_rel, v_rel, r_chief, v_chief)
    
    # Two-body acceleration for both spacecraft
    a_chief_gravity = two_body_acceleration(r_chief, dyn.μ)
    a_deputy_gravity = two_body_acceleration(r_deputy, dyn.μ)
    
    # Control force in ECI frame (transform from HCW using current attitude)
    # Note: u is assumed in HCW frame for now
    # For body-frame thrusters, this will be handled in spacecraft_dynamics
    x_hat = r_chief / norm(r_chief)
    h = cross(r_chief, v_chief)
    z_hat = h / norm(h)
    y_hat = cross(z_hat, x_hat)
    
    R_ECI_HCW = @SMatrix [
        x_hat[1] y_hat[1] z_hat[1];
        x_hat[2] y_hat[2] z_hat[2];
        x_hat[3] y_hat[3] z_hat[3]
    ]
    
    F_hcw = u_sv
    F_eci = R_ECI_HCW * F_hcw
    a_deputy_control = F_eci / dyn.m
    
    # Total acceleration
    a_chief = a_chief_gravity
    a_deputy = a_deputy_gravity + a_deputy_control
    
    # Relative acceleration in HCW frame
    # Need to account for frame rotation (Coriolis and centrifugal terms)
    ω_frame = orbital_frame_rate(r_chief, v_chief)
    
    # Transform accelerations to HCW frame
    R_HCW_ECI = R_ECI_HCW'
    a_rel_eci = a_deputy - a_chief
    
    # In rotating frame: ẍ_rel = R(a_rel - 2ω×v_rel - ω×(ω×r_rel) - ω̇×r_rel)
    # For circular orbit, ω̇ ≈ 0
    a_coriolis = 2 * cross(ω_frame, R_ECI_HCW * v_rel)
    a_centrifugal = cross(ω_frame, cross(ω_frame, R_ECI_HCW * r_rel))
    
    a_rel_hcw = R_HCW_ECI * (a_rel_eci - a_coriolis - a_centrifugal)
    
    # Derivative of chief state (two-body motion)
    ṙ_chief = v_chief
    v̇_chief = a_chief
    
    # Derivative of relative state
    ṙ_rel = v_rel
    v̇_rel = a_rel_hcw
    
    return SVector{12}(ṙ_rel..., v̇_rel..., ṙ_chief..., v̇_chief...)
end