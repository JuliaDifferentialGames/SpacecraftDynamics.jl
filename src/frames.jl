"""
    frames.jl

Coordinate frame definitions and transformations for spacecraft dynamics.

# Coordinate Frames
- **ECI (Earth-Centered Inertial)**: Inertial frame with origin at Earth's center
- **HCW (Hill-Clohessy-Wiltshire) / LVLH (Local-Vertical Local-Horizontal)**:
  - Origin at chief spacecraft (or virtual reference point)
  - x̂: radial (away from Earth)
  - ŷ: along-track (direction of orbital motion)
  - ẑ: cross-track (normal to orbital plane)
- **Body Frame**: Principal axes of spacecraft, origin at center of mass

# Key Functions
- `eci_to_hcw`: Transform state from ECI to HCW frame
- `hcw_to_eci`: Transform state from HCW to ECI frame
- `body_to_hcw`: Transform vectors from body to HCW frame using attitude

# References
S&J Chapter 13, Section 13.4 (Relative Motion Frames)
"""

"""
    eci_to_hcw(r_deputy, v_deputy, r_chief, v_chief)

Transform deputy spacecraft state from ECI to HCW frame.

The HCW frame is defined by the chief's orbital motion:
- x̂: radial direction (r̂_chief)
- ẑ: angular momentum direction (ĥ = r × v)
- ŷ: completes right-handed frame (ŷ = ẑ × x̂)

# Arguments
- `r_deputy::SVector{3}`: Deputy position in ECI [m]
- `v_deputy::SVector{3}`: Deputy velocity in ECI [m/s]
- `r_chief::SVector{3}`: Chief position in ECI [m]
- `v_chief::SVector{3}`: Chief velocity in ECI [m/s]

# Returns
- `r_rel::SVector{3}`: Relative position in HCW frame [m]
- `v_rel::SVector{3}`: Relative velocity in HCW frame [m/s]

# Notes
Assumes near-circular chief orbit (eccentricity ≈ 0). For eccentric orbits,
consider using true anomaly-based frame definitions.

# Reference
S&J Section 13.4, Eq 13.68-13.70
"""
function eci_to_hcw(r_deputy::SVector{3,T}, v_deputy::SVector{3,T},
                    r_chief::SVector{3,T}, v_chief::SVector{3,T}) where T
    # Construct HCW frame unit vectors in ECI
    x_hat = r_chief / norm(r_chief)  # Radial
    
    h = cross(r_chief, v_chief)      # Angular momentum
    z_hat = h / norm(h)               # Normal to orbit plane
    
    y_hat = cross(z_hat, x_hat)      # Along-track (completes right-hand frame)
    
    # Rotation matrix from ECI to HCW
    R_HCW_ECI = @SMatrix [
        x_hat[1] x_hat[2] x_hat[3];
        y_hat[1] y_hat[2] y_hat[3];
        z_hat[1] z_hat[2] z_hat[3]
    ]
    
    # Relative position and velocity in ECI
    Δr_eci = r_deputy - r_chief
    Δv_eci = v_deputy - v_chief
    
    # Transform to HCW frame
    r_rel = R_HCW_ECI * Δr_eci
    
    # Velocity transformation includes frame rotation rate
    # v_HCW = R(v_ECI - ω × r_ECI) where ω = h/|r|²
    ω_orbit = h / dot(r_chief, r_chief)
    v_rel = R_HCW_ECI * (Δv_eci - cross(ω_orbit, Δr_eci))
    
    return r_rel, v_rel
end

"""
    hcw_to_eci(r_rel, v_rel, r_chief, v_chief)

Transform deputy spacecraft state from HCW to ECI frame.

Inverse of `eci_to_hcw`.

# Arguments
- `r_rel::SVector{3}`: Relative position in HCW frame [m]
- `v_rel::SVector{3}`: Relative velocity in HCW frame [m/s]
- `r_chief::SVector{3}`: Chief position in ECI [m]
- `v_chief::SVector{3}`: Chief velocity in ECI [m/s]

# Returns
- `r_deputy::SVector{3}`: Deputy position in ECI [m]
- `v_deputy::SVector{3}`: Deputy velocity in ECI [m/s]
"""
function hcw_to_eci(r_rel::SVector{3,T}, v_rel::SVector{3,T},
                    r_chief::SVector{3,T}, v_chief::SVector{3,T}) where T
    # Construct HCW frame (same as eci_to_hcw)
    x_hat = r_chief / norm(r_chief)
    
    h = cross(r_chief, v_chief)
    z_hat = h / norm(h)
    
    y_hat = cross(z_hat, x_hat)
    
    # Rotation matrix from HCW to ECI (transpose of ECI to HCW)
    R_ECI_HCW = @SMatrix [
        x_hat[1] y_hat[1] z_hat[1];
        x_hat[2] y_hat[2] z_hat[2];
        x_hat[3] y_hat[3] z_hat[3]
    ]
    
    # Transform position
    Δr_eci = R_ECI_HCW * r_rel
    r_deputy = r_chief + Δr_eci
    
    # Transform velocity (including frame rotation)
    ω_orbit = h / dot(r_chief, r_chief)
    Δv_eci = R_ECI_HCW * v_rel + cross(ω_orbit, Δr_eci)
    v_deputy = v_chief + Δv_eci
    
    return r_deputy, v_deputy
end

"""
    body_to_hcw(v_body, σ)

Transform a vector from body frame to HCW frame using MRP attitude.

# Arguments
- `v_body::SVector{3}`: Vector in body frame
- `σ::SVector{3}`: MRP representing rotation from HCW to body

# Returns
- `v_hcw::SVector{3}`: Vector in HCW frame

# Notes
Uses the DCM computed from MRP: v_hcw = R(σ)ᵀ v_body
Since R(σ) rotates from HCW to body, Rᵀ rotates from body to HCW.
"""
function body_to_hcw(v_body::SVector{3,T}, σ::SVector{3,T}) where T
    R = mrp_to_dcm(σ)  # HCW to body
    return R' * v_body  # Transpose for body to HCW
end

"""
    hcw_to_body(v_hcw, σ)

Transform a vector from HCW frame to body frame using MRP attitude.

# Arguments
- `v_hcw::SVector{3}`: Vector in HCW frame
- `σ::SVector{3}`: MRP representing rotation from HCW to body

# Returns
- `v_body::SVector{3}`: Vector in body frame

# Notes
Uses the DCM computed from MRP: v_body = R(σ) v_hcw
"""
function hcw_to_body(v_hcw::SVector{3,T}, σ::SVector{3,T}) where T
    R = mrp_to_dcm(σ)  # HCW to body
    return R * v_hcw
end

"""
    orbital_frame_rate(r_chief, v_chief)

Compute the angular velocity of the HCW frame relative to ECI.

For a near-circular orbit, this is approximately:
ω = h / r² ≈ n ẑ

where n is the mean motion and ẑ is normal to orbital plane.

# Arguments
- `r_chief::SVector{3}`: Chief position in ECI [m]
- `v_chief::SVector{3}`: Chief velocity in ECI [m/s]

# Returns
- `ω_frame::SVector{3}`: Angular velocity of HCW frame in ECI [rad/s]

# Notes
This is used in velocity transformations between frames.
"""
function orbital_frame_rate(r_chief::SVector{3,T}, v_chief::SVector{3,T}) where T
    h = cross(r_chief, v_chief)
    r_squared = dot(r_chief, r_chief)
    return h / r_squared
end