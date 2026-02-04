"""
    attitude.jl

Attitude kinematics and dynamics using Modified Rodrigues Parameters (MRPs).
All equations follow Schaub & Junkins (4th ed), Chapter 3.

# Key Functions
- `mrp_kinematics`: MRP differential equation σ̇ = G(σ)ω
- `mrp_to_dcm`: Convert MRP to Direction Cosine Matrix
- `dcm_to_mrp`: Convert DCM to MRP
- `mrp_shadow_switch`: Numerically stable MRP switching
- `attitude_dynamics`: Rigid body rotational dynamics Jω̇ + ω × Jω = τ
"""

"""
    skew(v::AbstractVector)

Compute skew-symmetric matrix from vector: [v×]

Following S&J notation where [v×]w = v × w

# Arguments
- `v::AbstractVector{T}`: 3-element vector

# Returns
- `::SMatrix{3,3,T}`: Skew-symmetric matrix

# Example
```julia
v = @SVector [1.0, 2.0, 3.0]
v_skew = skew(v)
# v_skew * w ≈ cross(v, w)
```
"""
@inline function skew(v::SVector{3,T}) where T
    return @SMatrix [
        zero(T)  -v[3]     v[2];
        v[3]      zero(T) -v[1];
       -v[2]      v[1]     zero(T)
    ]
end

@inline function skew(v::AbstractVector{T}) where T
    @assert length(v) == 3 "Vector must be 3-dimensional"
    return skew(SVector{3,T}(v))
end

"""
    mrp_kinematics(σ, ω)

Compute MRP kinematic differential equation: σ̇ = G(σ)ω

Following S&J Eq 3.162:
```
G(σ) = (1/4)[(1-σᵀσ)I₃ + 2[σ×] + 2σσᵀ]
```

# Arguments
- `σ::SVector{3,T}`: Modified Rodrigues Parameters (dimensionless)
- `ω::SVector{3,T}`: Angular velocity in body frame [rad/s]

# Returns
- `::SVector{3,T}`: MRP time derivative σ̇ [1/s]

# Reference
S&J Section 3.7, Eq 3.162, page 131
"""
function mrp_kinematics(σ::SVector{3,T}, ω::SVector{3,T}) where T
    σ_norm_sq = dot(σ, σ)
    
    # G(σ) matrix (S&J Eq 3.162)
    # G = (1/4)[(1-σᵀσ)I₃ + 2[σ×] + 2σσᵀ]
    G = 0.25 * ((1 - σ_norm_sq) * I(3) + 2 * skew(σ) + 2 * (σ * σ'))
    
    return G * ω
end

"""
    mrp_norm(σ)

Compute the norm of MRP: |σ| = √(σᵀσ)

# Arguments
- `σ::AbstractVector`: Modified Rodrigues Parameters

# Returns
- `::Real`: Norm of MRP
"""
mrp_norm(σ::AbstractVector) = sqrt(dot(σ, σ))

"""
    mrp_shadow_switch(σ; threshold=1.0)

Apply shadow set switching when |σ| > threshold for numerical stability.

Following S&J Section 3.7.1, page 133:
When σᵀσ > threshold², switch to σ_shadow = -σ/(σᵀσ)

The shadow set provides an alternate parameterization of the same physical
rotation, avoiding numerical issues when |σ| becomes large.

# Arguments
- `σ::SVector{3,T}`: Current MRP
- `threshold::Real=1.0`: Switching threshold (typically 1.0)

# Returns
- `σ_new::SVector{3,T}`: MRP after potential shadow switch
- `switched::Bool`: Whether switching occurred

# Reference
S&J Section 3.7.1, page 133

# Example
```julia
σ = @SVector [1.5, 0.5, 0.3]  # |σ| > 1
σ_new, switched = mrp_shadow_switch(σ)
# σ_new will have |σ_new| < 1
```
"""
function mrp_shadow_switch(σ::SVector{3,T}; threshold::Real=1.0) where T
    σ_norm_sq = dot(σ, σ)
    
    if σ_norm_sq > threshold^2
        return -σ / σ_norm_sq, true
    else
        return σ, false
    end
end

"""
    mrp_shadow_switch!(σ::AbstractVector; threshold=1.0)

In-place shadow set switching. Modifies σ if switching occurs.

# Arguments
- `σ::AbstractVector`: Current MRP (modified in-place)
- `threshold::Real=1.0`: Switching threshold

# Returns
- `switched::Bool`: Whether switching occurred

# Example
```julia
σ = [1.5, 0.5, 0.3]
switched = mrp_shadow_switch!(σ)
# σ is now modified to shadow set
```
"""
function mrp_shadow_switch!(σ::AbstractVector{T}; threshold::Real=1.0) where T
    σ_norm_sq = dot(σ, σ)
    
    if σ_norm_sq > threshold^2
        σ .*= -1 / σ_norm_sq
        return true
    else
        return false
    end
end

"""
    should_switch_shadow(σ; threshold=1.0)

Check if MRP should be switched to shadow set.

# Arguments
- `σ::AbstractVector`: Current MRP
- `threshold::Real=1.0`: Switching threshold

# Returns
- `Bool`: True if |σ| > threshold
"""
function should_switch_shadow(σ::AbstractVector; threshold::Real=1.0)
    return dot(σ, σ) > threshold^2
end

"""
    mrp_to_dcm(σ)

Convert Modified Rodrigues Parameters to Direction Cosine Matrix (DCM).

Following S&J Eq 3.159:
```
R(σ) = I + (8[σ×]² - 4(1-σᵀσ)[σ×]) / (1+σᵀσ)²
```

The DCM represents the rotation from reference frame to body frame.

# Arguments
- `σ::SVector{3,T}`: Modified Rodrigues Parameters

# Returns
- `::SMatrix{3,3,T}`: Direction Cosine Matrix (rotation from reference to body)

# Properties
- R is orthogonal: RᵀR = I
- det(R) = 1 (proper rotation)

# Reference
S&J Eq 3.159, page 132
"""
function mrp_to_dcm(σ::SVector{3,T}) where T
    σ_norm_sq = dot(σ, σ)
    denom = (1 + σ_norm_sq)^2
    
    σ_skew = skew(σ)
    σ_skew_sq = σ_skew * σ_skew
    
    # S&J Eq 3.159
    R = I(3) + (8 * σ_skew_sq - 4 * (1 - σ_norm_sq) * σ_skew) / denom
    
    return SMatrix{3,3,T}(R')
end

"""
    dcm_to_mrp(R; tolerance=1e-10)

Convert Direction Cosine Matrix to Modified Rodrigues Parameters.

Uses the principal rotation approach via eigendecomposition to extract
the rotation axis and angle, then computes MRP via S&J Eq 3.157.

# Arguments
- `R::AbstractMatrix`: Direction Cosine Matrix (3×3, orthogonal)
- `tolerance::Real=1e-10`: Tolerance for near-zero rotation detection

# Returns
- `::SVector{3}`: Modified Rodrigues Parameters (short rotation, |σ| ≤ 1)

# Notes
- Returns the short rotation MRP (|σ| ≤ 1)
- For rotations near 180°, MRP approaches singularity; consider quaternions
- For near-zero rotations, returns σ ≈ 0

# Reference
S&J Eq 3.157 (MRP from principal rotation), page 130
S&J Eq 3.22 (principal rotation angle from trace), page 91
"""
function dcm_to_mrp(R::AbstractMatrix{T}; tolerance::Real=1e-10) where T
    # Principal rotation angle from trace (S&J Eq 3.22)
    trace_R = tr(R')
    cos_Φ = (trace_R - 1) / 2
    
    # Clamp to handle numerical errors
    cos_Φ = clamp(cos_Φ, -one(T), one(T))
    Φ = acos(cos_Φ)
    
    # Handle near-zero rotation
    if abs(Φ) < tolerance
        return @SVector zeros(T, 3)
    end
    
    # Handle 180° rotation (MRP singularity)
    if abs(Φ - π) < tolerance
        # Use quaternion intermediate (more stable near 180°)
        # Find the column with largest diagonal element
        i = argmax([R[1,1], R[2,2], R[3,3]])
        
        # Extract rotation axis (eigenvector corresponding to eigenvalue 1)
        if i == 1
            e = @SVector [
                sqrt((R[1,1] + 1) / 2),
                R[2,1] / (2 * sqrt((R[1,1] + 1) / 2)),
                R[3,1] / (2 * sqrt((R[1,1] + 1) / 2))
            ]
        elseif i == 2
            e = @SVector [
                R[1,2] / (2 * sqrt((R[2,2] + 1) / 2)),
                sqrt((R[2,2] + 1) / 2),
                R[3,2] / (2 * sqrt((R[2,2] + 1) / 2))
            ]
        else
            e = @SVector [
                R[1,3] / (2 * sqrt((R[3,3] + 1) / 2)),
                R[2,3] / (2 * sqrt((R[3,3] + 1) / 2)),
                sqrt((R[3,3] + 1) / 2)
            ]
        end
        
        # MRP for 180° rotation approaches infinity; return large but finite value
        # This indicates proximity to singularity
        return e * tan(Φ / 4) 
    end
    
    # Standard case: extract principal rotation axis
    # From R - Rᵀ = 2sin(Φ)[e×] (S&J Eq 3.24)
    sin_Φ = sin(Φ)
    e_skew = (R - R') / (2 * sin_Φ)
    
    # Extract axis from skew-symmetric matrix
    e = @SVector [e_skew[3,2], e_skew[1,3], e_skew[2,1]]
    
    # Normalize (should already be unit, but ensure numerical stability)
    e = e / norm(e)
    
    # Compute MRP from principal rotation (S&J Eq 3.157)
    σ = e * tan(Φ / 4)
    
    return σ
end

"""
    attitude_dynamics(ω, τ, J)

Compute rigid body rotational dynamics: ω̇ = J⁻¹(τ - ω × Jω)

Following S&J Eq 4.11 (Euler's rotational equations):
```
Jω̇ + ω × (Jω) = τ
```

# Arguments
- `ω::SVector{3,T}`: Angular velocity in body frame [rad/s]
- `τ::SVector{3,T}`: Applied torque in body frame [N⋅m]
- `J::SMatrix{3,3,T}`: Inertia tensor in body frame [kg⋅m²]

# Returns
- `::SVector{3,T}`: Angular acceleration ω̇ [rad/s²]

# Reference
S&J Section 4.2, Eq 4.11, page 166
"""
function attitude_dynamics(ω::SVector{3,T}, τ::SVector{3,T}, 
                          J::SMatrix{3,3,T}) where T
    # Euler's equation: Jω̇ = τ - ω × Jω
    # Therefore: ω̇ = J⁻¹(τ - ω × Jω)
    
    Jω = J * ω
    ω̇ = J \ (τ - cross(ω, Jω))
    
    return ω̇
end

"""
    attitude_dynamics(ω, τ, J::Diagonal)

Specialized version for diagonal inertia tensor (principal axes aligned).

More efficient computation when J is diagonal (common case for spacecraft
with symmetric geometry).

# Arguments
- `ω::SVector{3,T}`: Angular velocity in body frame [rad/s]
- `τ::SVector{3,T}`: Applied torque in body frame [N⋅m]  
- `J::Diagonal{T}`: Diagonal inertia tensor [kg⋅m²]

# Returns
- `::SVector{3,T}`: Angular acceleration ω̇ [rad/s²]
"""
function attitude_dynamics(ω::SVector{3,T}, τ::SVector{3,T}, 
                          J::Diagonal{T}) where T
    Jx, Jy, Jz = J.diag[1], J.diag[2], J.diag[3]
    
    # Euler's equations for principal axes (S&J Eq 4.13)
    ω̇x = (τ[1] - (Jz - Jy) * ω[2] * ω[3]) / Jx
    ω̇y = (τ[2] - (Jx - Jz) * ω[3] * ω[1]) / Jy
    ω̇z = (τ[3] - (Jy - Jx) * ω[1] * ω[2]) / Jz
    
    return @SVector [ω̇x, ω̇y, ω̇z]
end