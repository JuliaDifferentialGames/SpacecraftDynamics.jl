@testset "Spacecraft Configuration and Dynamics" begin
    
    @testset "SpacecraftState construction" begin
        # From components
        r = @SVector [100.0, 200.0, -50.0]
        v = @SVector [0.1, -0.2, 0.05]
        σ = @SVector [0.1, 0.05, -0.08]
        ω = @SVector [0.01, -0.02, 0.015]
        
        state = SpacecraftState(r, v, σ, ω)
        
        @test state.r ≈ r
        @test state.v ≈ v
        @test state.σ ≈ σ
        @test state.ω ≈ ω
        
        # From vector
        x = SVector{12}(r..., v..., σ..., ω...)
        state_from_vec = SpacecraftState(x)
        
        @test state_from_vec.r ≈ r
        @test state_from_vec.v ≈ v
        @test state_from_vec.σ ≈ σ
        @test state_from_vec.ω ≈ ω
        
        # Round-trip conversion
        x_recovered = to_vector(state)
        @test x_recovered ≈ x
    end
    
    @testset "Spacecraft construction" begin
        mass = 100.0
        inertia = Diagonal([10.0, 12.0, 8.0])
        actuators = default_research_actuators()
        
        sc = Spacecraft(
            mass=mass,
            inertia=inertia,
            actuators=actuators,
            attitude_enabled=true,
            orbital_dynamics=:linear_hcw
        )
        
        @test sc.mass == mass
        @test sc.inertia == inertia
        @test sc.attitude_enabled == true
        @test sc.orbital_dynamics == :linear_hcw
        
        # State and control dimensions
        @test state_dimension(sc) == 12
        @test control_dimension(sc) == 9  # 6 thrusters + 3 wheels
    end
    
    @testset "Default research spacecraft" begin
        sc = default_research_spacecraft(
            mass=80.0,
            thruster_force=2.0,
            attitude_enabled=true
        )
        
        @test sc.mass == 80.0
        @test state_dimension(sc) == 12
        @test control_dimension(sc) == 9
        
        # Without attitude
        sc_trans_only = default_research_spacecraft(
            mass=100.0,
            attitude_enabled=false
        )
        
        @test state_dimension(sc_trans_only) == 6
    end
    
    @testset "Translational dynamics" begin
        sc = default_research_spacecraft(attitude_enabled=false)
        
        # State
        r = @SVector [100.0, 200.0, -50.0]
        v = @SVector [0.1, -0.2, 0.05]
        
        # Force in HCW frame
        F_hcw = @SVector [1.0, 0.5, -0.3]
        
        # Mean motion
        n = 0.001  # rad/s
        
        ṙ, v̇ = translational_dynamics(sc, r, v, F_hcw, n)
        
        # Position derivative should equal velocity
        @test ṙ ≈ v
        
        # Check HCW acceleration structure
        # ax = 3n²x + 2nvy + Fx/m
        expected_ax = 3*n^2*r[1] + 2*n*v[2] + F_hcw[1]/sc.mass
        @test v̇[1] ≈ expected_ax atol=1e-10
        
        # ay = -2nvx + Fy/m
        expected_ay = -2*n*v[1] + F_hcw[2]/sc.mass
        @test v̇[2] ≈ expected_ay atol=1e-10
        
        # az = -n²z + Fz/m
        expected_az = -n^2*r[3] + F_hcw[3]/sc.mass
        @test v̇[3] ≈ expected_az atol=1e-10
    end
    
    @testset "Rotational dynamics" begin
        sc = default_research_spacecraft(attitude_enabled=true)
        
        # State
        σ = @SVector [0.1, 0.05, -0.08]
        ω = @SVector [0.01, -0.02, 0.015]
        
        # Torque
        τ_body = @SVector [0.05, -0.03, 0.02]
        
        σ̇, ω̇ = rotational_dynamics(sc, σ, ω, τ_body)
        
        # Check σ̇ from kinematics
        σ̇_expected = mrp_kinematics(σ, ω)
        @test σ̇ ≈ σ̇_expected
        
        # Check ω̇ from Euler's equation
        ω̇_expected = attitude_dynamics(ω, τ_body, sc.inertia)
        @test ω̇ ≈ ω̇_expected
    end
    
    @testset "Full spacecraft dynamics (attitude-enabled)" begin
        sc = default_research_spacecraft(
            mass=100.0,
            thruster_force=1.0,
            wheel_torque=0.1,
            attitude_enabled=true
        )
        
        # Initial state
        x0 = SVector{12}(
            100.0, 200.0, -50.0,      # r
            0.1, -0.2, 0.05,          # v
            0.1, 0.05, -0.08,         # σ
            0.01, -0.02, 0.015        # ω
        )
        
        # Control: fire +x thruster and x-axis wheel
        u = SVector{9}(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Thrusters
            0.5, 0.0, 0.0                   # Wheels
        )
        
        # Parameters
        n = 0.001  # rad/s
        p = (spacecraft=sc, n=n)
        
        # Evaluate dynamics
        ẋ = spacecraft_dynamics(x0, u, p, 0.0)
        
        @test length(ẋ) == 12
        
        # Position derivative should equal velocity
        @test ẋ[1:3] ≈ x0[4:6]
        
        # Test in-place version
        ẋ_inplace = zeros(12)
        spacecraft_dynamics!(ẋ_inplace, x0, u, p, 0.0)
        
        @test ẋ_inplace ≈ ẋ
    end
    
    @testset "Translational-only spacecraft dynamics" begin
        sc = default_research_spacecraft(
            mass=100.0,
            thruster_force=1.0,
            attitude_enabled=false
        )
        
        # Modify to remove wheels (required for translational-only)
        actuators_trans_only = ActuatorConfiguration(
            thrusters=sc.actuators.thrusters
        )
        
        sc_trans = Spacecraft(
            mass=sc.mass,
            inertia=sc.inertia,
            actuators=actuators_trans_only,
            attitude_enabled=false,
            orbital_dynamics=:linear_hcw
        )
        
        # 6D state
        x0 = SVector{6}(
            100.0, 200.0, -50.0,  # r
            0.1, -0.2, 0.05       # v
        )
        
        # Control: 6 thrusters
        u = SVector{6}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        n = 0.001
        p = (spacecraft=sc_trans, n=n)
        
        ẋ = spacecraft_dynamics(x0, u, p, 0.0)
        
        @test length(ẋ) == 6
        @test ẋ[1:3] ≈ x0[4:6]
    end
    
    @testset "Body-frame thrust transformation" begin
        sc = default_research_spacecraft(attitude_enabled=true)
        
        # 90° rotation about z-axis
        σ = @SVector [0.0, 0.0, tan(π/8)]
        
        # Fire +x thruster in body frame
        u = SVector{9}(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Thrusters
            0.0, 0.0, 0.0                   # Wheels
        )
        
        # Compute wrench in body frame
        F_body, τ_body = compute_wrench(sc.actuators, u)
        @test F_body ≈ @SVector([1.0, 0.0, 0.0])  # +x in body
        
        # Transform to HCW frame
        F_hcw = body_to_hcw(F_body, σ)
        
        # After 90° z-rotation, body +x points in HCW +y
        @test F_hcw ≈ @SVector([0.0, 1.0, 0.0]) atol=1e-10
    end
    
    @testset "Zero control dynamics" begin
        sc = default_research_spacecraft(attitude_enabled=true)
        
        # State with position but zero velocity/attitude
        x0 = SVector{12}(
            1000.0, 0.0, 0.0,      # r (radial position)
            0.0, 0.0, 0.0,         # v
            0.0, 0.0, 0.0,         # σ
            0.0, 0.0, 0.0          # ω
        )
        
        u = @SVector zeros(9)
        n = 0.001
        p = (spacecraft=sc, n=n)
        
        ẋ = spacecraft_dynamics(x0, u, p, 0.0)
        
        # Velocity derivative should show HCW dynamics
        # ax = 3n²x (radial acceleration for zero velocity)
        @test ẋ[4] ≈ 3*n^2*x0[1] atol=1e-10
        
        # No other accelerations
        @test abs(ẋ[5]) < 1e-10
        @test abs(ẋ[6]) < 1e-10
        
        # No attitude rate change
        @test norm(ẋ[7:12]) < 1e-10
    end
    
    @testset "Inertia tensor validation" begin
        # Should accept positive definite matrix
        J_valid = @SMatrix [
            10.0  1.0  0.5;
            1.0  12.0  0.3;
            0.5  0.3   8.0
        ]
        
        actuators = default_research_actuators()
        
        sc_valid = Spacecraft(
            mass=100.0,
            inertia=J_valid,
            actuators=actuators
        )
        
        @test sc_valid.inertia ≈ J_valid
        
        # Should reject non-positive definite
        J_invalid = @SMatrix [
            10.0  15.0  0.0;
            15.0  12.0  0.0;
            0.0   0.0   8.0
        ]
        
        @test_throws AssertionError Spacecraft(
            mass=100.0,
            inertia=J_invalid,
            actuators=actuators
        )
    end
    
    @testset "Mass validation" begin
        actuators = default_research_actuators()
        inertia = Diagonal([10.0, 12.0, 8.0])
        
        # Should reject negative mass
        @test_throws AssertionError Spacecraft(
            mass=-100.0,
            inertia=inertia,
            actuators=actuators
        )
        
        # Should reject zero mass
        @test_throws AssertionError Spacecraft(
            mass=0.0,
            inertia=inertia,
            actuators=actuators
        )
    end
    
    @testset "Orbital dynamics mode validation" begin
        actuators = default_research_actuators()
        inertia = Diagonal([10.0, 12.0, 8.0])
        
        # Valid modes
        sc_linear = Spacecraft(
            mass=100.0,
            inertia=inertia,
            actuators=actuators,
            orbital_dynamics=:linear_hcw
        )
        @test sc_linear.orbital_dynamics == :linear_hcw
        
        sc_nonlinear = Spacecraft(
            mass=100.0,
            inertia=inertia,
            actuators=actuators,
            orbital_dynamics=:nonlinear_hcw
        )
        @test sc_nonlinear.orbital_dynamics == :nonlinear_hcw
        
        # Invalid mode
        @test_throws AssertionError Spacecraft(
            mass=100.0,
            inertia=inertia,
            actuators=actuators,
            orbital_dynamics=:invalid_mode
        )
    end
end