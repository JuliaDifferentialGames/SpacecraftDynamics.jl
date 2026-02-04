@testset "Attitude Kinematics and Dynamics" begin
    
    @testset "Skew-symmetric matrix" begin
        v = @SVector [1.0, 2.0, 3.0]
        v_skew = skew(v)
        
        # Should be skew-symmetric
        @test v_skew' ≈ -v_skew
        
        # Should satisfy [v×]w = v × w
        w = @SVector [4.0, 5.0, 6.0]
        @test v_skew * w ≈ cross(v, w)
        
        # Trace should be zero
        @test tr(v_skew) ≈ 0.0 atol=1e-14
    end
    
    @testset "MRP kinematics" begin
        # Small rotation case
        σ = @SVector [0.1, 0.05, -0.08]
        ω = @SVector [0.01, -0.02, 0.015]
        
        σ̇ = mrp_kinematics(σ, ω)
        
        # Check dimension
        @test length(σ̇) == 3
        
        # Check σ̇ is proportional to ω for small σ
        # When σ ≈ 0, G(σ) ≈ I/4, so σ̇ ≈ ω/4
        @test norm(σ̇ - ω/4) / norm(ω) < 0.1
        
        # Zero angular velocity should give zero derivative
        σ̇_zero = mrp_kinematics(σ, @SVector(zeros(3)))
        @test norm(σ̇_zero) < 1e-14
    end
    
    @testset "MRP shadow switching" begin
        # Small MRP - no switching
        σ_small = @SVector [0.3, 0.2, 0.1]
        σ_new, switched = mrp_shadow_switch(σ_small)
        @test !switched
        @test σ_new ≈ σ_small
        
        # Large MRP - should switch
        σ_large = @SVector [1.5, 0.5, 0.3]
        σ_shadow, switched = mrp_shadow_switch(σ_large)
        @test switched
        @test mrp_norm(σ_shadow) < 1.0
        
        # Shadow relationship: σ_shadow = -σ / (σᵀσ)
        σ_norm_sq = dot(σ_large, σ_large)
        @test σ_shadow ≈ -σ_large / σ_norm_sq
        
        # Exactly at threshold
        σ_boundary = @SVector([1.0, 0.0, 0.0])
        _, switched_boundary = mrp_shadow_switch(σ_boundary, threshold=1.0)
        @test !switched_boundary  # |σ| = 1.0, not > 1.0
    end

    @testset "MRP shadow switching integration" begin
        using DifferentialEquations
        
        # Test case: constant angular velocity causes MRP to grow
        # Without switching, |σ| → ∞
        # With switching, |σ| stays < 1
        
        function mrp_dynamics!(dσ, σ, p, t)
            ω = p.ω  # Constant angular velocity
            dσ .= mrp_kinematics(SVector{3}(σ), ω)
        end
        
        # Constant spin about z-axis
        ω_spin = @SVector [0.0, 0.0, 0.5]  # rad/s
        p = (ω=ω_spin,)
        
        # Initial condition
        σ0 = [0.1, 0.0, 0.0]
        tspan = (0.0, 20.0)  # Long enough to require switching
        
        # Solve WITHOUT callback
        prob_no_cb = ODEProblem(mrp_dynamics!, σ0, tspan, p)
        sol_no_cb = solve(prob_no_cb, Tsit5())
        
        # Final |σ| without switching (will be large)
        σ_final_no_cb = sol_no_cb.u[end]
        @test norm(σ_final_no_cb) > 1.0  # Grows beyond unit circle
        
        # Solve WITH callback
        cb = create_mrp_switching_callback(attitude_indices=1:3)
        prob_with_cb = ODEProblem(mrp_dynamics!, σ0, tspan, p)
        sol_with_cb = solve(prob_with_cb, Tsit5(), callback=cb)
        
        # Check |σ| stays bounded
        for u in sol_with_cb.u
            @test norm(u) ≤ 1.1  # Allow small overshoot before switching
        end
        
        # Both solutions should represent same physical rotation at end
        R_no_cb = mrp_to_dcm(SVector{3}(σ_final_no_cb))
        R_with_cb = mrp_to_dcm(SVector{3}(sol_with_cb.u[end]))
        
        # DCMs should be similar (may differ due to different MRP representations)
        # Check rotation angle is consistent
        trace_diff = abs(tr(R_no_cb) - tr(R_with_cb))
        @test trace_diff < 0.1  # Same rotation angle
    end

    @testset "Manual shadow switching" begin
        # Test in-place switching
        σ = [1.5, 0.5, 0.3]
        σ_norm_sq_before = dot(σ, σ)
        
        switched = mrp_shadow_switch!(σ)
        
        @test switched == true
        @test norm(σ) < 1.0
        
        # Verify shadow relationship
        # σ_new = -σ_old / |σ_old|²
        σ_expected_norm_sq = 1 / σ_norm_sq_before
        @test dot(σ, σ) ≈ σ_expected_norm_sq atol=1e-10
        
        # No switching when already small
        σ_small = [0.3, 0.2, 0.1]
        σ_copy = copy(σ_small)
        switched = mrp_shadow_switch!(σ_small)
        
        @test switched == false
        @test σ_small == σ_copy  # Unchanged
    end

    @testset "Continuous vs discrete callback comparison" begin
        using DifferentialEquations
        
        function mrp_dynamics!(dσ, σ, p, t)
            ω = @SVector [0.0, 0.0, 1.0]  # 1 rad/s about z
            dσ .= mrp_kinematics(SVector{3}(σ), ω)
        end
        
        σ0 = [0.5, 0.0, 0.0]
        tspan = (0.0, 10.0)
        
        # Discrete callback
        cb_discrete = create_mrp_switching_callback(attitude_indices=1:3)
        prob1 = ODEProblem(mrp_dynamics!, copy(σ0), tspan)
        sol_discrete = solve(prob1, Tsit5(), callback=cb_discrete)
        
        # Continuous callback  
        cb_continuous = create_mrp_continuous_switching_callback(attitude_indices=1:3)
        prob2 = ODEProblem(mrp_dynamics!, copy(σ0), tspan)
        sol_continuous = solve(prob2, Tsit5(), callback=cb_continuous)
        
        # Both should keep |σ| ≤ 1
        for u in sol_discrete.u
            @test norm(u) ≤ 1.1
        end
        
        for u in sol_continuous.u
            @test norm(u) ≤ 1.0001  # Tighter bound for continuous
        end
        
        # Final rotations should be equivalent
        R_discrete = mrp_to_dcm(SVector{3}(sol_discrete.u[end]))
        R_continuous = mrp_to_dcm(SVector{3}(sol_continuous.u[end]))
        
        @test R_discrete ≈ R_continuous atol=1e-3
    end
    
    @testset "MRP to DCM conversion" begin
        # Identity rotation (σ = 0)
        σ_id = @SVector zeros(3)
        R_id = mrp_to_dcm(σ_id)
        @test R_id ≈ I(3)
        
        # 90° rotation about z-axis
        # Principal rotation: Φ = π/2, e = [0, 0, 1]
        # MRP: σ = e * tan(Φ/4) = [0, 0, tan(π/8)]
        Φ = π/2
        σ_90z = @SVector [0.0, 0.0, tan(Φ/4)]
        R_90z = mrp_to_dcm(σ_90z)
        
        # Expected DCM for 90° rotation about z
        R_expected = @SMatrix [
            0.0  1.0  0.0;  
            -1.0  0.0  0.0;  
            0.0  0.0  1.0
        ]
        @test R_90z ≈ R_expected atol=1e-10
        
        # DCM properties
        @test R_90z' * R_90z ≈ I(3) atol=1e-10  # Orthogonal
        @test det(R_90z) ≈ 1.0 atol=1e-10       # Proper rotation
        
        # Random MRP
        σ_rand = @SVector [0.3, -0.2, 0.15]
        R_rand = mrp_to_dcm(σ_rand)
        @test R_rand' * R_rand ≈ I(3) atol=1e-10
        @test det(R_rand) ≈ 1.0 atol=1e-10
    end
    
    @testset "DCM to MRP conversion" begin
        # Identity
        R_id = SMatrix{3,3,Float64}(I(3))
        σ_id = dcm_to_mrp(R_id)
        @test norm(σ_id) < 1e-10
        
        # 90° rotation about x-axis
        R_90x = @SMatrix [
            1.0  0.0  0.0;
            0.0  0.0 -1.0;
            0.0  1.0  0.0
        ]
        σ_90x = dcm_to_mrp(R_90x)
        
        # Should give σ = [tan(π/8), 0, 0]
        @test σ_90x[1] ≈ tan(π/8) atol=1e-10
        @test abs(σ_90x[2]) < 1e-10
        @test abs(σ_90x[3]) < 1e-10
        
        # Round-trip conversion
        σ_original = @SVector [0.2, -0.3, 0.1]
        R = mrp_to_dcm(σ_original)
        σ_recovered = dcm_to_mrp(R)
        @test σ_recovered ≈ σ_original atol=1e-10
    end
    
    @testset "Attitude dynamics (Euler's equation)" begin
        # Diagonal inertia (principal axes)
        J = Diagonal([10.0, 12.0, 8.0])
        
        # Zero torque, zero angular velocity → zero acceleration
        ω_zero = @SVector zeros(3)
        τ_zero = @SVector zeros(3)
        ω̇ = attitude_dynamics(ω_zero, τ_zero, J)
        @test norm(ω̇) < 1e-14
        
        # Pure torque about x-axis
        ω = @SVector zeros(3)
        τ_x = @SVector [1.0, 0.0, 0.0]
        ω̇_x = attitude_dynamics(ω, τ_x, J)
        
        # Should give ω̇ = [τ/Jₓ, 0, 0]
        @test ω̇_x[1] ≈ 1.0 / J.diag[1]
        @test abs(ω̇_x[2]) < 1e-14
        @test abs(ω̇_x[3]) < 1e-14
        
        # Torque-free motion with angular velocity (gyroscopic coupling)
        # For principal axes: ω̇ₓ = (Jᵧ - Jᵧ)ωᵧωᵧ / Jₓ
        ω_spin = @SVector [0.1, 0.2, 0.05]
        τ_free = @SVector zeros(3)
        ω̇_free = attitude_dynamics(ω_spin, τ_free, J)
        
        # Verify Euler's equation manually
        Jx, Jy, Jz = J.diag
        ω̇_expected = @SVector [
            ((Jy - Jz) * ω_spin[2] * ω_spin[3]) / Jx,
            ((Jz - Jx) * ω_spin[3] * ω_spin[1]) / Jy,
            ((Jx - Jy) * ω_spin[1] * ω_spin[2]) / Jz
        ]
        @test ω̇_free ≈ ω̇_expected atol=1e-12
        
        # Full inertia matrix (non-diagonal)
        J_full = @SMatrix [
            10.0  1.0  0.5;
            1.0  12.0  0.3;
            0.5  0.3  8.0
        ]
        
        ω̇_full = attitude_dynamics(ω_spin, τ_x, J_full)
        @test length(ω̇_full) == 3
        
        # Verify it satisfies Jω̇ = τ - ω × Jω
        Jω = J_full * ω_spin
        lhs = J_full * ω̇_full
        rhs = τ_x - cross(ω_spin, Jω)
        @test lhs ≈ rhs atol=1e-10
    end
    
    @testset "MRP norm" begin
        σ = @SVector [0.3, 0.4, 0.0]
        @test mrp_norm(σ) ≈ 0.5
        
        σ_unit = @SVector([1.0, 0.0, 0.0])
        @test mrp_norm(σ_unit) ≈ 1.0
    end
    
    @testset "MRP singularity handling" begin
        # Near 180° rotation (approaches MRP singularity)
        Φ_near_pi = π - 0.01
        σ_large = @SVector [0.0, 0.0, tan(Φ_near_pi/4)]
        
        # Should still convert to valid DCM
        R_large = mrp_to_dcm(σ_large)
        @test R_large' * R_large ≈ I(3) atol=1e-8
        @test det(R_large) ≈ 1.0 atol=1e-8
        
        # Convert back (may use shadow set internally)
        σ_recovered = dcm_to_mrp(R_large)
        R_recovered = mrp_to_dcm(σ_recovered)
        
        # DCMs should match (even if MRP representations differ)
        @test R_recovered ≈ R_large atol=1e-5
    end
end