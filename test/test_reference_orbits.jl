@testset "Reference Orbits" begin
    
    @testset "CircularOrbit construction" begin
        # From altitude
        altitude = 500e3  # 500 km
        orbit = CircularOrbit(altitude=altitude)
        
        @test orbit.a ≈ R_EARTH + altitude
        @test orbit.i ≈ 0.0
        @test orbit.Ω ≈ 0.0
        @test orbit.M0 ≈ 0.0
        @test orbit.epoch ≈ 0.0
        @test orbit.μ ≈ μ_EARTH
        
        # With custom parameters
        orbit_inclined = CircularOrbit(
            altitude=600e3,
            inclination=deg2rad(51.6),  # ISS-like
            raan=deg2rad(30.0),
            mean_anomaly_0=deg2rad(45.0),
            epoch=1000.0
        )
        
        @test orbit_inclined.i ≈ deg2rad(51.6)
        @test orbit_inclined.Ω ≈ deg2rad(30.0)
        @test orbit_inclined.M0 ≈ deg2rad(45.0)
        @test orbit_inclined.epoch ≈ 1000.0
    end
    
    @testset "Mean motion calculation" begin
        altitude = 400e3
        orbit = CircularOrbit(altitude=altitude)
        
        n = mean_motion(orbit)
        
        # Check against manual calculation
        a = R_EARTH + altitude
        n_expected = sqrt(μ_EARTH / a^3)
        @test n ≈ n_expected
        
        # Typical LEO: period ~90 minutes
        T = orbital_period(orbit)
        @test 5000 < T < 6000  # 83-100 minutes
    end
    
    @testset "Orbital period" begin
        # LEO
        orbit_leo = CircularOrbit(altitude=400e3)
        T_leo = orbital_period(orbit_leo)
        @test 5400 < T_leo < 5600  # ~90 minutes
        
        # Higher altitude → longer period
        orbit_high = CircularOrbit(altitude=1000e3)
        T_high = orbital_period(orbit_high)
        @test T_high > T_leo
        
        # Relationship: T = 2π/n
        n = mean_motion(orbit_leo)
        @test T_leo ≈ 2π/n
    end
    
    @testset "Mean anomaly propagation" begin
        orbit = CircularOrbit(
            altitude=500e3,
            mean_anomaly_0=deg2rad(30.0),
            epoch=0.0
        )
        
        n = mean_motion(orbit)
        
        # At epoch
        M0 = mean_anomaly_at_time(orbit, 0.0)
        @test M0 ≈ deg2rad(30.0)
        
        # After one period
        T = orbital_period(orbit)
        M_after_period = mean_anomaly_at_time(orbit, T)
        @test M_after_period ≈ deg2rad(30.0) atol=1e-6  # Wrapped back
        
        # After half period
        M_half = mean_anomaly_at_time(orbit, T/2)
        expected_M_half = mod2pi(deg2rad(30.0) + π)
        @test M_half ≈ expected_M_half atol=1e-6
    end
    
    @testset "ECI state at time (equatorial orbit)" begin
        # Equatorial circular orbit
        altitude = 500e3
        orbit = CircularOrbit(
            altitude=altitude,
            inclination=0.0,
            mean_anomaly_0=0.0
        )
        
        # At epoch (M = 0, on x-axis)
        r_eci, v_eci = state_at_time(orbit, 0.0)
        
        a = orbit.a
        n = mean_motion(orbit)
        
        # Position should be on x-axis
        @test r_eci[1] ≈ a atol=1e-3
        @test abs(r_eci[2]) < 1e-3
        @test abs(r_eci[3]) < 1e-3
        
        # Velocity should be in y-direction
        v_mag = n * a
        @test abs(v_eci[1]) < 1e-3
        @test v_eci[2] ≈ v_mag atol=1e-3
        @test abs(v_eci[3]) < 1e-3
        
        # After 90° (quarter orbit)
        T = orbital_period(orbit)
        r_90, v_90 = state_at_time(orbit, T/4)
        
        # Should be on y-axis
        @test abs(r_90[1]) < 1e-3
        @test r_90[2] ≈ a atol=1e-3
        @test abs(r_90[3]) < 1e-3
        
        # Velocity in -x direction
        @test v_90[1] ≈ -v_mag atol=1e-3
        @test abs(v_90[2]) < 1e-3
    end
    
    @testset "ECI state at time (inclined orbit)" begin
        # 45° inclined orbit
        altitude = 600e3
        orbit = CircularOrbit(
            altitude=altitude,
            inclination=π/4,
            mean_anomaly_0=0.0
        )
        
        r_eci, v_eci = state_at_time(orbit, 0.0)
        
        a = orbit.a
        n = mean_motion(orbit)
        
        # Position magnitude should be semi-major axis
        @test norm(r_eci) ≈ a atol=1e-3
        
        # Velocity magnitude should be circular velocity
        v_mag = n * a
        @test norm(v_eci) ≈ v_mag atol=1e-3
        
        # Should be perpendicular
        @test abs(dot(r_eci, v_eci)) < 1e-3
    end
    
    @testset "Orbit properties preservation" begin
        orbit = CircularOrbit(
            altitude=500e3,
            inclination=deg2rad(51.6)
        )
        
        a = orbit.a
        n = mean_motion(orbit)
        T = orbital_period(orbit)
        
        # Sample at multiple times
        times = [0.0, T/4, T/2, 3*T/4, T]
        
        for t in times
            r, v = state_at_time(orbit, t)
            
            # Radius constant (circular)
            @test norm(r) ≈ a atol=1e-3
            
            # Speed constant
            @test norm(v) ≈ n*a atol=1e-3
            
            # Perpendicular
            @test abs(dot(r, v)) < 1e-2
            
            # Energy constant (for circular orbit)
            E = 0.5*norm(v)^2 - μ_EARTH/norm(r)
            E_expected = -μ_EARTH/(2*a)
            @test E ≈ E_expected rtol=1e-6
        end
    end
    
    @testset "VirtualChief construction" begin
        chief = VirtualChief(altitude=500e3)
        
        @test isa(chief.orbit, CircularOrbit)
        @test chief.orbit.a ≈ R_EARTH + 500e3
        
        # Mean motion
        n = mean_motion(chief)
        @test n > 0
        
        # State at time
        r, v = state_at_time(chief, 0.0)
        @test length(r) == 3
        @test length(v) == 3
    end
    
    @testset "GameScenarioSetup construction" begin
        # Create components
        chief = VirtualChief(altitude=500e3)
        
        pursuer = default_research_spacecraft(thruster_force=2.0)
        evader = default_research_spacecraft(thruster_force=1.5)
        
        deputies = [pursuer, evader]
        
        # Initial states
        state1 = SpacecraftState(
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        state2 = SpacecraftState(
            @SVector([0.0, 1000.0, 0.0]),
            @SVector(zeros(3)),
            @SVector(zeros(3)),
            @SVector(zeros(3))
        )
        
        initial_states = [state1, state2]
        
        # Create scenario
        scenario = GameScenarioSetup(chief, deputies, initial_states, 600.0)
        
        @test length(scenario.deputies) == 2
        @test length(scenario.initial_states) == 2
        @test scenario.time_horizon ≈ 600.0
    end
    
    @testset "Pursuit-evasion scenario creation" begin
        scenario = create_pursuit_evasion_scenario(
            altitude=400e3,
            pursuer_thrust=3.0,
            evader_thrust=2.0,
            separation=1500.0,
            time_horizon=300.0
        )
        
        @test length(scenario.deputies) == 2
        @test length(scenario.initial_states) == 2
        @test scenario.time_horizon ≈ 300.0
        
        # Check pursuer and evader
        pursuer = scenario.deputies[1]
        evader = scenario.deputies[2]
        
        @test control_dimension(pursuer) > 0
        @test control_dimension(evader) > 0
        
        # Check initial separation
        state_p = scenario.initial_states[1]
        state_e = scenario.initial_states[2]
        
        separation = norm(state_e.r - state_p.r)
        @test separation ≈ 1500.0
    end
    
    @testset "RealChief construction" begin
        orbit = CircularOrbit(altitude=500e3)
        spacecraft = default_research_spacecraft()
        
        chief = RealChief(spacecraft, orbit)
        
        @test isa(chief.spacecraft, Spacecraft)
        @test isa(chief.initial_orbit, CircularOrbit)
    end
end