"""
    callbacks.jl

Callback functions for ODE integration with proper event handling.

Implements shadow set switching for MRP attitude representation during
numerical integration.
"""

"""
    create_mrp_switching_callback(; threshold=1.0, attitude_indices=7:9)

Create a callback for automatic MRP shadow set switching during integration.

This callback monitors the MRP magnitude during integration and switches
to the shadow set when |σ| > threshold to maintain numerical stability.

# Keyword Arguments
- `threshold::Real=1.0`: Switching threshold (typically 1.0)
- `attitude_indices::UnitRange=7:9`: Indices of MRP in state vector

# Returns
- `DiscreteCallback`: Callback for use with DifferentialEquations.jl

# Usage with DifferentialEquations.jl
```julia
using DifferentialEquations

# Create callback
cb = create_mrp_switching_callback(attitude_indices=7:9)

# Solve with callback
prob = ODEProblem(dynamics!, x0, tspan, p)
sol = solve(prob, Tsit5(), callback=cb)
```

# Notes
- Uses DiscreteCallback for efficient event detection
- Only activates when attitude dynamics are enabled
- Can be combined with other callbacks using CallbackSet

# Reference
S&J Section 3.7.1, pages 133-134
"""
function create_mrp_switching_callback(; threshold::Real=1.0, attitude_indices::UnitRange=7:9)
    
    # Condition: check if |σ| > threshold at end of each timestep
    function condition(u, t, integrator)
        σ = @view u[attitude_indices]
        σ_norm_sq = dot(σ, σ)
        return σ_norm_sq > threshold^2
    end
    
    # Affect: switch to shadow set
    function affect!(integrator)
        σ = @view integrator.u[attitude_indices]
        σ_norm_sq = dot(σ, σ)
        
        # Apply shadow set transformation: σ_new = -σ/(σᵀσ)
        @. σ = -σ / σ_norm_sq
        
        # Optional: log switching event
        # @info "MRP shadow switching at t=$(integrator.t), |σ|=$(sqrt(σ_norm_sq))"
    end
    
    # Create discrete callback (checked at end of each timestep)
    return DiscreteCallback(condition, affect!)
end

"""
    create_mrp_continuous_switching_callback(; threshold=1.0, attitude_indices=7:9)

Create a continuous callback for MRP shadow set switching.

Unlike the discrete version, this uses root-finding to detect exactly when
|σ| = threshold and switches at that precise moment.

# Keyword Arguments
- `threshold::Real=1.0`: Switching threshold
- `attitude_indices::UnitRange=7:9`: Indices of MRP in state vector

# Returns
- `ContinuousCallback`: Callback with exact root-finding

# Notes
More accurate than discrete callback but with higher computational cost.
Use when precise switching timing is critical.

# Reference
S&J Section 3.7.1, pages 133-134
"""
function create_mrp_continuous_switching_callback(; threshold::Real=1.0, attitude_indices::UnitRange=7:9)
    
    # Condition: |σ|² - threshold² = 0 (detect crossing)
    function condition(u, t, integrator)
        σ = @view u[attitude_indices]
        σ_norm_sq = dot(σ, σ)
        return σ_norm_sq - threshold^2
    end
    
    # Affect: switch to shadow set
    function affect!(integrator)
        σ = @view integrator.u[attitude_indices]
        σ_norm_sq = dot(σ, σ)
        
        # Apply shadow set transformation
        @. σ = -σ / σ_norm_sq
        
        # Optional: log switching event
        # @info "MRP shadow switching (continuous) at t=$(integrator.t)"
    end
    
    # Create continuous callback with root-finding
    # affect_neg! not needed since we only care about crossing from below
    return ContinuousCallback(condition, affect!, nothing)
end