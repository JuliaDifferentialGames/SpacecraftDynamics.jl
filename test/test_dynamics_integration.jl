@testset "Dynamics Integration with DifferentialGamesBase" begin

    """
    Helper to create translational-only spacecraft (no reaction wheels)
    """
    function create_translational_spacecraft(;
        mass::Real=100.0,
        thruster_force::Real=1.0
    )
        thrusters = default_research_thrusters(max_thrust=thruster_force)
        actuators = ActuatorConfiguration(thrusters=thrusters)
        
        return Spacecraft(
            mass=mass,
            inertia=Diagonal([10.0, 12.0, 8.0]),
            actuators=actuators,
            attitude_enabled=false,
            orbital_dynamics=:linear_hcw
        )
    end
    
    @testset "SpacecraftGameDynamics construction" begin
        # Create spacecraft
        sc1 = default_research_spacecraft(mass=100.0, thruster_force=2.0)
        sc2 = default_research_spacecraft(mass=120.0, thruster_force=1.5)
        
        spacecraft = [sc1, sc2]
        chief = VirtualChief(altitude=500e3)
        
        game_dyn = SpacecraftGameDynamics(spacecraft, chief)
        
        @test length(game_dyn.spacecraft) == 2
        @test game_dyn.n > 0
        @test game_dyn.coupled == false
    end
    
    @testset "Player dimensions" begin
        # Create spacecraft with different configurations
        # Player 1: attitude-enabled (12D state, 9 controls)
        sc1 = default_research_spacecraft(attitude_enabled=true)
        
        # Player 2: translational-only - need to remove wheels!
        thrusters_only = ActuatorConfiguration(
            thrusters=default_research_thrusters()
        )
        sc2 = Spacecraft(
            mass=100.0,
            inertia=Diagonal([10.0, 12.0, 8.0]),
            actuators=thrusters_only,
            attitude_enabled=false,
            orbital_dynamics=:linear_hcw
        )
        
        chief = VirtualChief(altitude=500e3)
        game_dyn = SpacecraftGameDynamics([sc1, sc2], chief)
        
        # Player 1: attitude-enabled (12D state, 9 controls)
        @test player_state_dimension(game_dyn, 1) == 12
        @test player_control_dimension(game_dyn, 1) == 9
        
        # Player 2: translational-only (6D state, 6 controls - thrusters only)
        @test player_state_dimension(game_dyn, 2) == 6
        @test player_control_dimension(game_dyn, 2) == 6
    end

    @testset "Total dimensions" begin
        # Three spacecraft with varied configurations
        sc1 = default_research_spacecraft(attitude_enabled=true)  # 12D, 9 controls
        
        # Translational-only spacecraft (6D, 6 controls)
        thrusters_only = ActuatorConfiguration(
            thrusters=default_research_thrusters()
        )
        sc2 = Spacecraft(
            mass=100.0,
            inertia=Diagonal([10.0, 12.0, 8.0]),
            actuators=thrusters_only,
            attitude_enabled=false,
            orbital_dynamics=:linear_hcw
        )
        
        sc3 = default_research_spacecraft(attitude_enabled=true)  # 12D, 9 controls
        
        chief = VirtualChief(altitude=500e3)
        game_dyn = SpacecraftGameDynamics([sc1, sc2, sc3], chief)
        
        # Total state: 12 + 6 + 12 = 30
        @test total_state_dimension(game_dyn) == 30
        
        # Total control: 9 + 6 + 9 = 24
        @test total_control_dimension(game_dyn) == 24
    end
    
    @testset "State partition" begin
        sc1 = default_research_spacecraft(attitude_enabled=true)   # 12D
        sc2 = default_research_spacecraft(attitude_enabled=false)  # 6D
        sc3 = default_research_spacecraft(attitude_enabled=true)   # 12D
        
        chief = VirtualChief(altitude=500e3)
        game_dyn = SpacecraftGameDynamics([sc1, sc2, sc3], chief)
        
        ranges = state_partition(game_dyn)
        
        @test length(ranges) == 3
        @test ranges[1] == 1:12
        @test ranges[2] == 13:18
        @test ranges[3] == 19:30
        
        # Verify total coverage
        @test ranges[3].stop == total_state_dimension(game_dyn)
    end
    
    @testset "Control partition" begin
        sc1 = default_research_spacecraft()  # 9 controls
        sc2 = default_research_spacecraft()  # 9 controls
        
        chief = VirtualChief(altitude=500e3)
        game_dyn = SpacecraftGameDynamics([sc1, sc2], chief)
        
        ranges = control_partition(game_dyn)
        
        @test length(ranges) == 2
        @test ranges[1] == 1:9
        @test ranges[2] == 10:18
    end
    
    @testset "Stack and unstack states" begin
        x1 = @SVector randn(12)
        x2 = @SVector randn(6)
        x3 = @SVector randn(12)
        
        x_players = [Vector(x1), Vector(x2), Vector(x3)]
        x_stacked = stack_states(x_players)
        
        @test length(x_stacked) == 30
        @test x_stacked[1:12] ≈ x1
        @test x_stacked[13:18] ≈ x2
        @test x_stacked[19:30] ≈ x3
        
        # Unstack
        x_ranges = [1:12, 13:18, 19:30]
        x_unstacked = unstack_states(x_stacked, x_ranges)
        
        @test length(x_unstacked) == 3
        @test x_unstacked[1] ≈ Vector(x1)
        @test x_unstacked[2] ≈ Vector(x2)
        @test x_unstacked[3] ≈ Vector(x3)
    end
    
    @testset "Stack and unstack controls" begin
        u1 = @SVector randn(9)
        u2 = @SVector randn(6)
        
        u_players = [Vector(u1), Vector(u2)]
        u_stacked = stack_controls(u_players)
        
        @test length(u_stacked) == 15
        @test u_stacked[1:9] ≈ u1
        @test u_stacked[10:15] ≈ u2
        
        # Unstack
        u_ranges = [1:9, 10:15]
        u_unstacked = unstack_controls(u_stacked, u_ranges)
        
        @test length(u_unstacked) == 2
        @test u_unstacked[1] ≈ Vector(u1)
        @test u_unstacked[2] ≈ Vector(u2)
    end
    
    @testset "Spacecraft dynamics wrapper" begin
        sc = default_research_spacecraft(attitude_enabled=true)
        chief = VirtualChief(altitude=500e3)
        
        game_dyn = SpacecraftGameDynamics([sc], chief)
        
        x_ranges = state_partition(game_dyn)
        u_ranges = control_partition(game_dyn)
        
        # Create wrapper for player 1
        dyn_fn = spacecraft_dynamics_wrapper(1, game_dyn, x_ranges, u_ranges)
        
        # Test evaluation
        xᵢ = randn(12)
        uᵢ = randn(9)
        x_joint = xᵢ  # Single player, so joint = individual
        u_joint = uᵢ
        t = 0.0
        
        ẋᵢ = dyn_fn(xᵢ, uᵢ, x_joint, u_joint, t)
        
        @test length(ẋᵢ) == 12
        
        # Should match direct spacecraft dynamics
        p = (spacecraft=sc, n=game_dyn.n)
        ẋᵢ_direct = spacecraft_dynamics(xᵢ, uᵢ, p, t)
        
        @test ẋᵢ ≈ ẋᵢ_direct
    end
    
    @testset "Create separable dynamics" begin
        sc1 = default_research_spacecraft(attitude_enabled=true)
        sc2 = default_research_spacecraft(attitude_enabled=true)
        
        chief = VirtualChief(altitude=500e3)
        game_dyn = SpacecraftGameDynamics([sc1, sc2], chief)
        
        dyn_fns, n_x, n_u, x_ranges, u_ranges = create_separable_dynamics(game_dyn)
        
        # Check returned dimensions
        @test length(n_x) == 2
        @test n_x[1] == 12
        @test n_x[2] == 12
        
        @test length(n_u) == 2
        @test n_u[1] == 9
        @test n_u[2] == 9
        
        # Check dynamics functions are callable
        @test length(dyn_fns) == 2
        
        x1 = randn(12)
        u1 = randn(9)
        x_joint = [x1; randn(12)]
        u_joint = [u1; randn(9)]
        
        ẋ1 = dyn_fns[1](x1, u1, x_joint, u_joint, 0.0)
        @test length(ẋ1) == 12
    end
    
    @testset "Create game parameters" begin
        sc1 = default_research_spacecraft()
        sc2 = default_research_spacecraft()
        
        chief = VirtualChief(altitude=500e3)
        game_dyn = SpacecraftGameDynamics([sc1, sc2], chief)
        
        params = create_game_parameters(game_dyn)
        
        @test haskey(params, :spacecraft)
        @test haskey(params, :reference)
        @test haskey(params, :n)
        @test haskey(params, :x_ranges)
        @test haskey(params, :u_ranges)
        @test haskey(params, :coupled)
        
        @test length(params.spacecraft) == 2
        @test params.n > 0
        @test params.coupled == false
    end
    
    @testset "Multi-player dynamics evaluation" begin
        # Two-player scenario with proper translational-only setup
        
        # Pursuer: translational-only
        pursuer_thrusters = ActuatorConfiguration(
            thrusters=default_research_thrusters(max_thrust=2.0)
        )
        pursuer = Spacecraft(
            mass=80.0,
            inertia=Diagonal([10.0, 12.0, 8.0]),
            actuators=pursuer_thrusters,
            attitude_enabled=false,
            orbital_dynamics=:linear_hcw
        )
        
        # Evader: translational-only
        evader_thrusters = ActuatorConfiguration(
            thrusters=default_research_thrusters(max_thrust=1.5)
        )
        evader = Spacecraft(
            mass=100.0,
            inertia=Diagonal([10.0, 12.0, 8.0]),
            actuators=evader_thrusters,
            attitude_enabled=false,
            orbital_dynamics=:linear_hcw
        )
        
        chief = VirtualChief(altitude=500e3)
        game_dyn = SpacecraftGameDynamics([pursuer, evader], chief)
        
        dyn_fns, n_x, n_u, x_ranges, u_ranges = create_separable_dynamics(game_dyn)
        
        # Joint state and control (both 6D state, 6D control)
        x_p = @SVector randn(6)
        x_e = @SVector randn(6)
        x_joint = [x_p; x_e]
        
        u_p = @SVector randn(6)
        u_e = @SVector randn(6)
        u_joint = [u_p; u_e]
        
        t = 0.0
        
        # Evaluate each player's dynamics
        ẋ_p = dyn_fns[1](x_p, u_p, x_joint, u_joint, t)
        ẋ_e = dyn_fns[2](x_e, u_e, x_joint, u_joint, t)
        
        @test length(ẋ_p) == 6
        @test length(ẋ_e) == 6
        
        # Joint derivative
        ẋ_joint = [ẋ_p; ẋ_e]
        @test length(ẋ_joint) == 12
    end
end