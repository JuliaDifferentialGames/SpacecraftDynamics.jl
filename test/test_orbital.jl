@testset "Orbital Dynamics" begin
    
    @testset "Mean motion calculation" begin
        # 400 km altitude
        altitude = 400e3
        a = R_EARTH + altitude
        n = hcw_mean_motion(a)
        
        # Check units and magnitude
        @test n > 0
        @test n ≈ sqrt(μ_EARTH / a^3)
        
        # Typical LEO: n ≈ 0.001 rad/s (period ~90 min)
        period = 2π / n
        @test 5000 < period < 6000  # 83-100 minutes
        
        # From position vector
        r_chief = @SVector [a, 0.0, 0.0]
        n_from_r = hcw_mean_motion(r_chief)
        @test n_from_r ≈ n
    end
    
    @testset "HCW state matrix" begin
        n = 0.001  # rad/s
        A = hcw_state_matrix(n)
        
        # Check dimensions
        @test size(A) == (6, 6)
        
        # Check structure (S&J Eq 13.95-13.97)
        # Upper-right should be identity
        @test A[1:3, 4:6] ≈ I(3)
        
        # Check specific elements
        @test A[4, 1] ≈ 3*n^2  # x-acceleration coefficient
        @test A[4, 5] ≈ 2*n    # Coriolis term
        @test A[5, 4] ≈ -2*n   # Coriolis term
        @test A[6, 3] ≈ -n^2   # z-acceleration coefficient
    end
    
    @testset "HCW control matrix" begin
        m = 100.0  # kg
        B = hcw_control_matrix(m)
        
        # Check dimensions
        @test size(B) == (6, 3)
        
        # Upper half should be zero
        @test B[1:3, :] ≈ zeros(3, 3)
        
        # Lower half should be (1/m)I
        @test B[4:6, :] ≈ (1/m) * I(3)
    end
    
    @testset "Linear HCW dynamics" begin
        altitude = 500e3
        mass = 100.0
        
        dyn = LinearHCWDynamics(altitude=altitude, mass=mass)
        
        # Check fields
        @test dyn.m == mass
        @test dyn.n > 0
        
        # Test evaluation: ẋ = Ax + Bu
        x = @SVector [100.0, 200.0, -50.0, 0.1, -0.2, 0.05]
        u = @SVector [1.0, 0.5, -0.3]
        
        ẋ = dyn(x, u, 0.0)
        
        # Check dimensions
        @test length(ẋ) == 6
        
        # Position derivatives should equal velocities
        @test ẋ[1:3] ≈ x[4:6]
        
        # Verify against manual calculation
        ẋ_manual = dyn.A * x + dyn.B * u
        @test ẋ ≈ ẋ_manual
        
        # Zero control, zero velocity → drifting dynamics
        x_drift = @SVector [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        u_zero = @SVector zeros(3)
        ẋ_drift = dyn(x_drift, u_zero, 0.0)
        
        # Radial position creates acceleration (3n²x term)
        @test ẋ_drift[4] ≈ 3 * dyn.n^2 * x_drift[1]
    end
    
    @testset "Nonlinear HCW dynamics" begin
        mass = 100.0
        dyn = NonlinearHCWDynamics(mass=mass)
        
        # Check fields
        @test dyn.m == mass
        @test dyn.μ ≈ μ_EARTH
        
        # Test evaluation with full state (12D: deputy rel + chief abs)
        altitude = 500e3
        a = R_EARTH + altitude
        n = sqrt(μ_EARTH / a^3)
        
        # Chief state in ECI
        r_chief = @SVector [a, 0.0, 0.0]
        v_chief = @SVector [0.0, n*a, 0.0]
        
        # Deputy relative state in HCW
        r_rel = @SVector [100.0, 200.0, -50.0]
        v_rel = @SVector [0.1, -0.2, 0.05]
        
        # Combined state
        x = SVector{12}(r_rel..., v_rel..., r_chief..., v_chief...)
        u = @SVector [1.0, 0.5, -0.3]
        
        ẋ = dyn(x, u, 0.0)
        
        # Check dimensions
        @test length(ẋ) == 12
        
        # Relative position derivatives should equal relative velocities
        @test ẋ[1:3] ≈ v_rel
        
        # Chief position derivative should equal chief velocity
        @test ẋ[7:9] ≈ v_chief
        
        # Chief acceleration should be two-body gravity
        a_chief_expected = -(μ_EARTH / a^2) * (r_chief / a)
        @test ẋ[10:12] ≈ a_chief_expected atol=1e-6
    end
    
    @testset "Two-body acceleration" begin
        # At Earth's surface
        r_surface = @SVector [R_EARTH, 0.0, 0.0]
        a_surface = two_body_acceleration(r_surface, μ_EARTH)
        
        # Should point toward Earth center (negative r direction)
        @test a_surface[1] < 0
        @test abs(a_surface[2]) < 1e-10
        @test abs(a_surface[3]) < 1e-10
        
        # Magnitude should be g₀ ≈ 9.81 m/s²
        @test norm(a_surface) ≈ 9.81 atol=0.1
        
        # At LEO altitude
        altitude = 400e3
        r_leo = @SVector [R_EARTH + altitude, 0.0, 0.0]
        a_leo = two_body_acceleration(r_leo, μ_EARTH)
        
        # Should be weaker than surface gravity
        @test norm(a_leo) < norm(a_surface)
        
        # Should satisfy a = -μ/r²
        expected_mag = μ_EARTH / norm(r_leo)^2
        @test norm(a_leo) ≈ expected_mag atol=1e-6
    end
    
    @testset "Linear vs Nonlinear HCW comparison" begin
        # For small separations, linear and nonlinear should be similar
        altitude = 500e3
        mass = 100.0
        
        lin_dyn = LinearHCWDynamics(altitude=altitude, mass=mass)
        nonlin_dyn = NonlinearHCWDynamics(mass=mass)
        
        # Setup state
        a = R_EARTH + altitude
        n = sqrt(μ_EARTH / a^3)
        r_chief = @SVector [a, 0.0, 0.0]
        v_chief = @SVector [0.0, n*a, 0.0]
        
        # Small relative state
        r_rel = @SVector [10.0, 20.0, -5.0]  # Small separation
        v_rel = @SVector [0.01, -0.02, 0.005]
        
        u = @SVector [0.1, 0.05, -0.03]
        
        # Linear dynamics
        x_lin = SVector{6}(r_rel..., v_rel...)
        ẋ_lin = lin_dyn(x_lin, u, 0.0)
        
        # Nonlinear dynamics
        x_nonlin = SVector{12}(r_rel..., v_rel..., r_chief..., v_chief...)
        ẋ_nonlin = nonlin_dyn(x_nonlin, u, 0.0)
        
        # Compare relative state derivatives (first 6 elements)
        # Should be close for small separations
        @test ẋ_nonlin[1:6] ≈ ẋ_lin atol=1e-3
    end
end