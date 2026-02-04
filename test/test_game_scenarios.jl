@testset "Game Scenarios" begin
    
    @testset "Pursuit-evasion setup" begin
        scenario = create_pursuit_evasion_scenario(
            altitude=500e3,
            pursuer_thrust=2.0,
            evader_thrust=1.5,
            separation=1000.0,
            time_horizon=600.0
        )
        
        # Check structure
        @test isa(scenario.reference, VirtualChief)
        @test length(scenario.deputies) == 2
        @test length(scenario.initial_states) == 2
        @test scenario.time_horizon == 600.0
        
        # Check spacecraft are different
        pursuer = scenario.deputies[1]
        evader = scenario.deputies[2]
        
        @test pursuer.mass == evader.mass
        # Thrust capabilities should differ (via actuator max_thrust)
        
        # Check initial conditions
        state_p = scenario.initial_states[1]
        state_e = scenario.initial_states[2]
        
        # Pursuer at origin
        @test norm(state_p.r) < 1e-10
        
        # Evader ahead
        @test state_e.r[2] ≈ 1000.0  # Along-track separation
        
        # Both have zero velocity initially
        @test norm(state_p.v) < 1e-10
        @test norm(state_e.v) < 1e-10
    end
    
    @testset "Convert scenario to game dynamics" begin
        scenario = create_pursuit_evasion_scenario()
        
        game_dyn = SpacecraftGameDynamics(
            scenario.deputies,
            scenario.reference
        )
        
        @test total_state_dimension(game_dyn) == 12  # 6 + 6 (both translational-only)
        @test total_control_dimension(game_dyn) == 12  # 6 + 6
    end
    
    @testset "Scenario initial state vector" begin
        scenario = create_pursuit_evasion_scenario()
        
        # Convert initial states to vector
        x0_p = to_vector(scenario.initial_states[1])
        x0_e = to_vector(scenario.initial_states[2])
        
        # For translational-only, should be 6D each
        # But to_vector returns 12D always, so extract first 6
        @test length(x0_p) == 12
        @test length(x0_e) == 12
        
        # Stack for joint initial condition
        x0_joint = [x0_p[1:6]; x0_e[1:6]]
        @test length(x0_joint) == 12
    end
    
    @testset "Reference orbit properties for scenario" begin
        scenario = create_pursuit_evasion_scenario(altitude=400e3)
        
        n = mean_motion(scenario.reference)
        T = 2π / n
        
        # Period should be reasonable for LEO
        @test 5000 < T < 6000  # ~90 minutes
        
        # Get reference state at t=0
        r_ref, v_ref = state_at_time(scenario.reference, 0.0)
        
        @test norm(r_ref) ≈ R_EARTH + 400e3 atol=1e3
    end
    
    @testset "Multi-spacecraft scenario" begin
        # Three spacecraft: 1 evader, 2 pursuers
        evader = default_research_spacecraft(
            mass=100.0,
            thruster_force=1.5,
            attitude_enabled=false
        )
        
        pursuer1 = default_research_spacecraft(
            mass=80.0,
            thruster_force=2.0,
            attitude_enabled=false
        )
        
        pursuer2 = default_research_spacecraft(
            mass=80.0,
            thruster_force=2.0,
            attitude_enabled=false
        )
        
        deputies = [evader, pursuer1, pursuer2]
        
        # Initial states
        state_e = SpacecraftState(
            @SVector([0.0, 1000.0, 0.0]),
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        state_p1 = SpacecraftState(
            @SVector([-500.0, 0.0, 0.0]),
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        state_p2 = SpacecraftState(
            @SVector([500.0, 0.0, 0.0]),
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        initial_states = [state_e, state_p1, state_p2]
        
        chief = VirtualChief(altitude=500e3)
        
        scenario = GameScenarioSetup(chief, deputies, initial_states, 600.0)
        
        @test length(scenario.deputies) == 3
        @test length(scenario.initial_states) == 3
        
        # Create game dynamics
        game_dyn = SpacecraftGameDynamics(scenario.deputies, scenario.reference)
        
        @test total_state_dimension(game_dyn) == 18  # 3 × 6
        @test total_control_dimension(game_dyn) == 18  # 3 × 6
    end
    
    @testset "Lady-Bandit-Guard problem structure" begin
        # Lady-Bandit-Guard: 3 players
        # Lady: evading bandit, protected by guard
        # Bandit: pursuing lady
        # Guard: intercepting bandit
        
        lady = default_research_spacecraft(
            mass=100.0,
            thruster_force=1.0,
            attitude_enabled=false
        )
        
        bandit = default_research_spacecraft(
            mass=120.0,
            thruster_force=1.5,
            attitude_enabled=false
        )
        
        guard = default_research_spacecraft(
            mass=100.0,
            thruster_force=2.0,
            attitude_enabled=false
        )
        
        deputies = [lady, bandit, guard]
        
        # Initial configuration: Lady ahead, bandit pursuing, guard behind bandit
        state_lady = SpacecraftState(
            @SVector([0.0, 1500.0, 0.0]),
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        state_bandit = SpacecraftState(
            @SVector([0.0, 500.0, 0.0]),
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        state_guard = SpacecraftState(
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        initial_states = [state_lady, state_bandit, state_guard]
        
        chief = VirtualChief(altitude=500e3)
        scenario = GameScenarioSetup(chief, deputies, initial_states, 600.0)
        
        @test length(scenario.deputies) == 3
        
        # Verify relative positions
        @test state_lady.r[2] > state_bandit.r[2]  # Lady ahead of bandit
        @test state_bandit.r[2] > state_guard.r[2]  # Bandit ahead of guard
    end
    
    @testset "Sun-blocking problem structure" begin
        # Sun-blocking: Inspector trying to maintain line-of-sight blocked by target
        # Assumes sun direction is known (e.g., +x direction in HCW)
        
        target = default_research_spacecraft(
            mass=150.0,
            thruster_force=1.0,
            attitude_enabled=true  # Needs to orient for blocking
        )
        
        inspector = default_research_spacecraft(
            mass=80.0,
            thruster_force=1.5,
            attitude_enabled=false  # Just needs to maintain position
        )
        
        deputies = [target, inspector]
        
        # Target between sun (+x) and inspector
        state_target = SpacecraftState(
            @SVector([500.0, 0.0, 0.0]),  # Radial position
            @SVector(zeros(3)),
            @SVector([0.1, 0.0, 0.0]),    # Slight rotation
            @SVector(zeros(3))
        )
        
        state_inspector = SpacecraftState(
            @SVector([1000.0, 100.0, 0.0]),  # Behind target from sun's perspective
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        initial_states = [state_target, state_inspector]
        
        chief = VirtualChief(altitude=500e3)
        scenario = GameScenarioSetup(chief, deputies, initial_states, 300.0)
        
        @test length(scenario.deputies) == 2
        @test state_dimension(target) == 12  # Attitude-enabled
        @test state_dimension(inspector) == 6  # Translational-only
    end
end