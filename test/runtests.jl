using Test
using SpacecraftDynamics
using LinearAlgebra
using StaticArrays

@testset "SpacecraftDynamics.jl" begin
    include("test_attitude.jl")
    # include("test_frames.jl")
    # include("test_orbital.jl")
    # include("test_actuators.jl")
    # include("test_spacecraft.jl")
    # include("test_reference_orbits.jl")
    # include("test_dynamics_integration.jl")
    # include("test_game_scenarios.jl")
end