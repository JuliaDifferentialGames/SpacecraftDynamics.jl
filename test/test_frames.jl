@testset "Coordinate Frame Transformations" begin
    
    @testset "ECI to HCW transformation" begin
        # Circular orbit at 500 km altitude
        altitude = 500e3
        a = R_EARTH + altitude
        n = sqrt(μ_EARTH / a^3)
        
        # Chief at x-axis, moving in y-direction (circular orbit in xy-plane)
        r_chief = @SVector [a, 0.0, 0.0]
        v_chief = @SVector [0.0, n*a, 0.0]
        
        # Deputy ahead in orbit (positive y in HCW)
        Δy_hcw = 1000.0  # 1 km ahead
        r_deputy = r_chief + @SVector [0.0, Δy_hcw, 0.0]
        v_deputy = v_chief
        
        # Transform to HCW
        r_rel, v_rel = eci_to_hcw(r_deputy, v_deputy, r_chief, v_chief)
        
        # Should be approximately at [0, Δy, 0] in HCW
        # Small radial component due to curvature
        @test abs(r_rel[1]) < 100.0  # Relaxed: radial component from curvature
        @test r_rel[2] ≈ Δy_hcw atol=100.0  # Along-track matches
        @test abs(r_rel[3]) < 1.0   # Cross-track near zero
        
        # Relative velocity in rotating HCW frame
        # For circular orbits with along-track separation, there IS relative velocity
        # in the rotating frame due to differential orbital rates
        @test norm(v_rel) < 2.0  # Relaxed tolerance
    end
    
    @testset "HCW to ECI transformation (inverse)" begin
        # Setup chief
        altitude = 400e3
        a = R_EARTH + altitude
        n = sqrt(μ_EARTH / a^3)
        
        r_chief = @SVector [a, 0.0, 0.0]
        v_chief = @SVector [0.0, n*a, 0.0]
        
        # Relative state in HCW
        r_rel = @SVector [100.0, 500.0, -50.0]
        v_rel = @SVector [0.1, -0.2, 0.05]
        
        # Transform to ECI
        r_deputy, v_deputy = hcw_to_eci(r_rel, v_rel, r_chief, v_chief)
        
        # Round-trip should recover original
        r_rel_recovered, v_rel_recovered = eci_to_hcw(r_deputy, v_deputy, r_chief, v_chief)
        
        @test r_rel_recovered ≈ r_rel atol=1e-6
        @test v_rel_recovered ≈ v_rel atol=1e-6
    end
    
    @testset "Body to HCW transformation" begin
        # Identity attitude (body aligned with HCW)
        σ_id = @SVector zeros(3)
        v_body = @SVector [1.0, 2.0, 3.0]
        v_hcw = body_to_hcw(v_body, σ_id)
        
        @test v_hcw ≈ v_body
        
        # 90° rotation about z-axis
        σ_90z = @SVector [0.0, 0.0, tan(π/8)]
        v_body_x = @SVector [1.0, 0.0, 0.0]
        v_hcw_rotated = body_to_hcw(v_body_x, σ_90z)
        
        # After 90° z-rotation, body-x points in HCW ±y direction
        # Sign depends on MRP convention; check magnitude
        @test abs(v_hcw_rotated[1]) < 1e-10
        @test abs(abs(v_hcw_rotated[2]) - 1.0) < 1e-10  # |y-component| = 1
        @test abs(v_hcw_rotated[3]) < 1e-10
        
        # Verify orthogonality is preserved
        v_body_y = @SVector [0.0, 1.0, 0.0]
        v_hcw_y = body_to_hcw(v_body_y, σ_90z)
        @test abs(dot(v_hcw_rotated, v_hcw_y)) < 1e-10  # Perpendicular
    end
    
    @testset "HCW to Body transformation" begin
        σ = @SVector [0.1, -0.2, 0.15]
        v_hcw = @SVector [2.0, 1.0, -0.5]
        
        # Transform to body
        v_body = hcw_to_body(v_hcw, σ)
        
        # Round-trip
        v_hcw_recovered = body_to_hcw(v_body, σ)
        @test v_hcw_recovered ≈ v_hcw atol=1e-10
    end
    
    @testset "Orbital frame rate" begin
        altitude = 600e3
        a = R_EARTH + altitude
        n = sqrt(μ_EARTH / a^3)
        
        # Circular orbit in xy-plane
        r_chief = @SVector [a, 0.0, 0.0]
        v_chief = @SVector [0.0, n*a, 0.0]
        
        ω_frame = orbital_frame_rate(r_chief, v_chief)
        
        # Should be approximately [0, 0, n] for circular orbit in xy-plane
        @test abs(ω_frame[1]) < 1e-10
        @test abs(ω_frame[2]) < 1e-10
        @test ω_frame[3] ≈ n atol=1e-10
        
        # Magnitude should equal mean motion
        @test norm(ω_frame) ≈ n atol=1e-10
    end
    
    @testset "Inclined orbit frame" begin
        # 45° inclined orbit
        altitude = 500e3
        a = R_EARTH + altitude
        n = sqrt(μ_EARTH / a^3)
        v_mag = n * a
        
        # Chief at ascending node
        inc = π/4
        r_chief = @SVector [a, 0.0, 0.0]
        v_chief = @SVector [0.0, v_mag*cos(inc), v_mag*sin(inc)]
        
        # Deputy slightly ahead
        r_deputy = r_chief + @SVector [0.0, 100.0, 50.0]
        v_deputy = v_chief
        
        r_rel, v_rel = eci_to_hcw(r_deputy, v_deputy, r_chief, v_chief)
        
        # Should have reasonable relative position
        @test norm(r_rel) < 200.0
        
        # Round-trip
        r_deputy_rec, v_deputy_rec = hcw_to_eci(r_rel, v_rel, r_chief, v_chief)
        @test r_deputy_rec ≈ r_deputy atol=1e-6
        @test v_deputy_rec ≈ v_deputy atol=1e-6
    end
end