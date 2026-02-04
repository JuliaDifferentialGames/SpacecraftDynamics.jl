"""
    dynamics_integration.jl

Integration with DifferentialGamesBase.jl for multi-agent differential games.

Provides adapters to convert spacecraft dynamics into the SeperableDynamics
format expected by game solvers (iLQGames, IBR, JAGUAR, etc.).

# Key Functions
- `create_separable_dynamics`: Main interface for game problem setup
- `spacecraft_dynamics_wrapper`: Adapter for individual agent dynamics
- `create_game_parameters`: Parameter structure for multi-agent scenarios

# Design Philosophy
This file bridges the gap between:
1. Spacecraft-centric representation (Spacecraft, SpacecraftState, etc.)
2. Game-theoretic representation (SeperableDynamics with player indices)

The goal is to allow users to define spacecraft naturally, then automatically
convert to game solver format.

# References
DifferentialGamesBase.jl: https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl
"""

"""
    SpacecraftGameDynamics{T,N}

Wrapper for spacecraft dynamics in differential game format.

Stores all spacecraft and reference orbit information needed to evaluate
dynamics for N players in a differential game.

# Type Parameters
- `T<:Real`: Numeric type
- `N::Int`: Number of players

# Fields
- `spacecraft::NTuple{N, Spacecraft{T}}`: Spacecraft for each player
- `reference::Union{VirtualChief{T}, RealChief{T}}`: Reference orbit
- `n::T`: Mean motion of reference orbit [rad/s]
- `coupled::Bool`: Whether dynamics are coupled between players

# Notes
For separable dynamics (typical in spacecraft games), coupling occurs only
through costs and constraints, not through dynamics. The `coupled` flag
indicates whether one player's control directly affects another's dynamics
(rare in formation flying, common in contact scenarios).
"""
struct SpacecraftGameDynamics{T<:Real, N}
    spacecraft::NTuple{N, Spacecraft{T}}
    reference::Union{VirtualChief{T}, RealChief{T}}
    n::T
    coupled::Bool
    
    function SpacecraftGameDynamics{T,N}(
        spacecraft::NTuple{N, Spacecraft{T}},
        reference::Union{VirtualChief{T}, RealChief{T}},
        coupled::Bool=false
    ) where {T<:Real, N}
        n = mean_motion(reference)
        new{T,N}(spacecraft, reference, n, coupled)
    end
end

"""
    SpacecraftGameDynamics(spacecraft::Vector{Spacecraft{T}}, reference; coupled=false)

Construct game dynamics from vector of spacecraft.

# Arguments
- `spacecraft::Vector{Spacecraft}`: Spacecraft for each player
- `reference`: Virtual or real chief
- `coupled::Bool=false`: Whether dynamics are coupled

# Returns
- `SpacecraftGameDynamics{T,N}`: Game dynamics specification
"""
function SpacecraftGameDynamics(
    spacecraft::Vector{Spacecraft{T}},
    reference::Union{VirtualChief{T}, RealChief{T}};
    coupled::Bool=false
) where T<:Real
    N = length(spacecraft)
    spacecraft_tuple = NTuple{N, Spacecraft{T}}(spacecraft)
    return SpacecraftGameDynamics{T,N}(spacecraft_tuple, reference, coupled)
end

"""
    player_state_dimension(game_dyn::SpacecraftGameDynamics, player_idx::Int)

Get state dimension for a specific player.

# Arguments
- `game_dyn::SpacecraftGameDynamics`: Game dynamics
- `player_idx::Int`: Player index (1-based)

# Returns
- `n_x::Int`: State dimension for this player
"""
function player_state_dimension(game_dyn::SpacecraftGameDynamics{T,N}, player_idx::Int) where {T,N}
    @assert 1 ≤ player_idx ≤ N "Player index must be in [1, $N]"
    return state_dimension(game_dyn.spacecraft[player_idx])
end

"""
    player_control_dimension(game_dyn::SpacecraftGameDynamics, player_idx::Int)

Get control dimension for a specific player.

# Arguments
- `game_dyn::SpacecraftGameDynamics`: Game dynamics
- `player_idx::Int`: Player index (1-based)

# Returns
- `n_u::Int`: Control dimension for this player
"""
function player_control_dimension(game_dyn::SpacecraftGameDynamics{T,N}, player_idx::Int) where {T,N}
    @assert 1 ≤ player_idx ≤ N "Player index must be in [1, $N]"
    return control_dimension(game_dyn.spacecraft[player_idx])
end

"""
    total_state_dimension(game_dyn::SpacecraftGameDynamics)

Get total state dimension across all players.

# Returns
- `n_x_total::Int`: Sum of all player state dimensions
"""
function total_state_dimension(game_dyn::SpacecraftGameDynamics{T,N}) where {T,N}
    return sum(state_dimension(sc) for sc in game_dyn.spacecraft)
end

"""
    total_control_dimension(game_dyn::SpacecraftGameDynamics)

Get total control dimension across all players.

# Returns
- `n_u_total::Int`: Sum of all player control dimensions
"""
function total_control_dimension(game_dyn::SpacecraftGameDynamics{T,N}) where {T,N}
    return sum(control_dimension(sc) for sc in game_dyn.spacecraft)
end

"""
    state_partition(game_dyn::SpacecraftGameDynamics)

Get state vector partition indices for each player.

# Returns
- `ranges::Vector{UnitRange{Int}}`: Index ranges for each player's state

# Example
```julia
game_dyn = SpacecraftGameDynamics([sc1, sc2], chief)
ranges = state_partition(game_dyn)
# ranges[1] = 1:12 (player 1 state indices)
# ranges[2] = 13:24 (player 2 state indices)
```
"""
function state_partition(game_dyn::SpacecraftGameDynamics{T,N}) where {T,N}
    ranges = Vector{UnitRange{Int}}(undef, N)
    idx = 1
    
    for i in 1:N
        n_x_i = state_dimension(game_dyn.spacecraft[i])
        ranges[i] = idx:(idx + n_x_i - 1)
        idx += n_x_i
    end
    
    return ranges
end

"""
    control_partition(game_dyn::SpacecraftGameDynamics)

Get control vector partition indices for each player.

# Returns
- `ranges::Vector{UnitRange{Int}}`: Index ranges for each player's control
"""
function control_partition(game_dyn::SpacecraftGameDynamics{T,N}) where {T,N}
    ranges = Vector{UnitRange{Int}}(undef, N)
    idx = 1
    
    for i in 1:N
        n_u_i = control_dimension(game_dyn.spacecraft[i])
        ranges[i] = idx:(idx + n_u_i - 1)
        idx += n_u_i
    end
    
    return ranges
end

"""
    spacecraft_dynamics_wrapper(player_idx, game_dyn, x_ranges, u_ranges)

Create dynamics function for a single player in separable dynamics format.

Returns a function with signature: f(xᵢ, uᵢ, x, u, t) → ẋᵢ

# Arguments
- `player_idx::Int`: Player index
- `game_dyn::SpacecraftGameDynamics`: Game dynamics specification
- `x_ranges::Vector{UnitRange{Int}}`: State partition
- `u_ranges::Vector{UnitRange{Int}}`: Control partition

# Returns
- `f::Function`: Dynamics function for this player

# Notes
The returned function has the signature expected by SeperableDynamics:
- `xᵢ`: This player's state
- `uᵢ`: This player's control
- `x`: Full joint state (all players)
- `u`: Full joint control (all players)
- `t`: Time

For separable (uncoupled) dynamics, `f` only depends on (xᵢ, uᵢ, t).
The full state `x` is available for coupled scenarios (e.g., collision avoidance
based on other agents' positions).
"""
function spacecraft_dynamics_wrapper(
    player_idx::Int,
    game_dyn::SpacecraftGameDynamics{T,N},
    x_ranges::Vector{UnitRange{Int}},
    u_ranges::Vector{UnitRange{Int}}
) where {T,N}
    # Extract player-specific information
    sc = game_dyn.spacecraft[player_idx]
    n = game_dyn.n
    
    # Create parameters for spacecraft_dynamics
    p = (spacecraft=sc, n=n)
    
    # Dynamics function with SeperableDynamics signature
    function dynamics_fn(
        xᵢ::AbstractVector,
        uᵢ::AbstractVector,
        x::AbstractVector,
        u::AbstractVector,
        t::Real
    )
        # For separable dynamics, ignore joint state x and joint control u
        # (only use them if coupling is needed)
        
        # Evaluate spacecraft dynamics for this player
        ẋᵢ = spacecraft_dynamics(xᵢ, uᵢ, p, t)
        
        return ẋᵢ
    end
    
    return dynamics_fn
end

"""
    create_separable_dynamics(game_dyn::SpacecraftGameDynamics)

Create SeperableDynamics object for DifferentialGamesBase.jl solvers.

This is the main interface for converting spacecraft game setup into
the format expected by game-theoretic solvers.

# Arguments
- `game_dyn::SpacecraftGameDynamics{T,N}`: Game dynamics specification

# Returns
- `dyn::SeperableDynamics{N}`: Separable dynamics for game solvers

# Example
```julia
# Setup spacecraft
pursuer = default_research_spacecraft(thruster_force=2.0)
evader = default_research_spacecraft(thruster_force=1.5)
chief = VirtualChief(altitude=500e3)

# Create game dynamics
game_dyn = SpacecraftGameDynamics([pursuer, evader], chief)

# Convert to separable dynamics for game solver
sep_dyn = create_separable_dynamics(game_dyn)

# Use with DifferentialGamesBase.jl
# game_prob = GameProblem(sep_dyn, costs, constraints, ...)
```

# Notes
The returned SeperableDynamics has:
- N players (one per spacecraft)
- State dimensions: (n_x₁, n_x₂, ..., n_xₙ)
- Control dimensions: (n_u₁, n_u₂, ..., n_uₙ)
"""
function create_separable_dynamics(game_dyn::SpacecraftGameDynamics{T,N}) where {T,N}
    # Get state and control partitions
    x_ranges = state_partition(game_dyn)
    u_ranges = control_partition(game_dyn)
    
    # Get dimensions for each player
    n_x = [length(r) for r in x_ranges]
    n_u = [length(r) for r in u_ranges]
    
    # Create dynamics function for each player
    dynamics_fns = ntuple(N) do i
        spacecraft_dynamics_wrapper(i, game_dyn, x_ranges, u_ranges)
    end
    
    # Create SeperableDynamics
    # Note: This assumes SeperableDynamics constructor signature
    # If DifferentialGamesBase.jl has a different constructor, adjust accordingly
    n_x_total = sum(n_x)
    n_u_total = sum(n_u)
    
    # Return the tuple of functions wrapped in appropriate type
    # The exact constructor depends on DifferentialGamesBase.jl API
    return dynamics_fns, n_x, n_u, x_ranges, u_ranges
end

"""
    create_game_parameters(game_dyn::SpacecraftGameDynamics)

Create parameter structure for game problem.

Returns a NamedTuple with all information needed during game solving:
- Spacecraft configurations
- Reference orbit parameters
- State/control partitions

# Arguments
- `game_dyn::SpacecraftGameDynamics`: Game dynamics

# Returns
- `params::NamedTuple`: Game parameters

# Example
```julia
params = create_game_parameters(game_dyn)

# Access components
params.spacecraft        # Tuple of spacecraft
params.n                 # Mean motion
params.x_ranges          # State partition
params.u_ranges          # Control partition
```
"""
function create_game_parameters(game_dyn::SpacecraftGameDynamics{T,N}) where {T,N}
    return (
        spacecraft = game_dyn.spacecraft,
        reference = game_dyn.reference,
        n = game_dyn.n,
        x_ranges = state_partition(game_dyn),
        u_ranges = control_partition(game_dyn),
        coupled = game_dyn.coupled
    )
end

"""
    extract_player_trajectory(sol, player_idx, x_ranges)

Extract trajectory for a single player from joint solution.

# Arguments
- `sol`: ODE solution object (from DifferentialEquations.jl)
- `player_idx::Int`: Player index
- `x_ranges::Vector{UnitRange{Int}}`: State partition

# Returns
- `times::Vector`: Time points
- `states::Matrix`: State trajectory (n_x × n_t)

# Example
```julia
# After solving game
times, states = extract_player_trajectory(sol, 1, params.x_ranges)
# states[1:3, :] are positions over time
# states[4:6, :] are velocities over time
```
"""
function extract_player_trajectory(
    sol,
    player_idx::Int,
    x_ranges::Vector{UnitRange{Int}}
)
    # Extract this player's state indices
    idx = x_ranges[player_idx]
    
    # Extract trajectory
    times = sol.t
    states = reduce(hcat, [sol.u[i][idx] for i in 1:length(sol.u)])
    
    return times, states
end

"""
    stack_controls(u_players::Vector{<:AbstractVector})

Stack individual player controls into joint control vector.

# Arguments
- `u_players::Vector`: Vector of control vectors, one per player

# Returns
- `u::Vector`: Stacked control vector [u₁; u₂; ...; uₙ]

# Example
```julia
u1 = [0.5, 0.3, 0.0]  # Player 1 control (3D)
u2 = [0.8, -0.2]       # Player 2 control (2D)
u = stack_controls([u1, u2])  # 5D joint control
```
"""
function stack_controls(u_players::Vector{<:AbstractVector{T}}) where T
    return reduce(vcat, u_players)
end

"""
    unstack_controls(u::AbstractVector, u_ranges::Vector{UnitRange{Int}})

Unstack joint control vector into individual player controls.

# Arguments
- `u::AbstractVector`: Joint control vector
- `u_ranges::Vector{UnitRange{Int}}`: Control partition

# Returns
- `u_players::Vector{Vector}`: Individual control vectors

# Example
```julia
u = [0.5, 0.3, 0.0, 0.8, -0.2]  # Joint control
u_ranges = [1:3, 4:5]
u_players = unstack_controls(u, u_ranges)
# u_players[1] = [0.5, 0.3, 0.0]
# u_players[2] = [0.8, -0.2]
```
"""
function unstack_controls(u::AbstractVector{T}, u_ranges::Vector{UnitRange{Int}}) where T
    N = length(u_ranges)
    u_players = Vector{Vector{T}}(undef, N)
    
    for i in 1:N
        u_players[i] = u[u_ranges[i]]
    end
    
    return u_players
end

"""
    stack_states(x_players::Vector{<:AbstractVector})

Stack individual player states into joint state vector.

# Arguments
- `x_players::Vector`: Vector of state vectors, one per player

# Returns
- `x::Vector`: Stacked state vector [x₁; x₂; ...; xₙ]
"""
function stack_states(x_players::Vector{<:AbstractVector{T}}) where T
    return reduce(vcat, x_players)
end

"""
    unstack_states(x::AbstractVector, x_ranges::Vector{UnitRange{Int}})

Unstack joint state vector into individual player states.

# Arguments
- `x::AbstractVector`: Joint state vector
- `x_ranges::Vector{UnitRange{Int}}`: State partition

# Returns
- `x_players::Vector{Vector}`: Individual state vectors
"""
function unstack_states(x::AbstractVector{T}, x_ranges::Vector{UnitRange{Int}}) where T
    N = length(x_ranges)
    x_players = Vector{Vector{T}}(undef, N)
    
    for i in 1:N
        x_players[i] = x[x_ranges[i]]
    end
    
    return x_players
end