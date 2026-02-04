@testset "Actuators" begin
    
    @testset "Thruster configuration" begin
        # Simple 6-thruster config
        positions = [
            @SVector(zeros(3)) for _ in 1:6
        ]
        
        directions = [
            @SVector([1.0, 0.0, 0.0]),
            @SVector([-1.0, 0.0, 0.0]),
            @SVector([0.0, 1.0, 0.0]),
            @SVector([0.0, -1.0, 0.0]),
            @SVector([0.0, 0.0, 1.0]),
            @SVector([0.0, 0.0, -1.0])
        ]
        
        max_thrust = 1.0
        
        config = ThrusterConfiguration(positions, directions, max_thrust)
        
        # Check fields
        @test length(config.positions) == 6
        @test length(config.directions) == 6
        @test config.continuous == true
        
        # Check direction normalization
        for d in config.directions
            @test norm(d) ≈ 1.0
        end
    end
    
    @testset "Default research thrusters" begin
        config = default_research_thrusters(max_thrust=2.0)
        
        # Should have 6 thrusters
        @test length(config.max_thrust) == 6
        
        # All at center of mass
        for pos in config.positions
            @test norm(pos) < 1e-10
        end
        
        # Check max thrust
        for thrust in config.max_thrust
            @test thrust ≈ 2.0
        end
    end
    
    @testset "Thruster wrench computation" begin
        config = default_research_thrusters(max_thrust=1.0)
        
        # Fire +x thruster at full thrust
        u = @SVector [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        F, τ = compute_thruster_wrench(config, u)
        
        @test F ≈ @SVector([1.0, 0.0, 0.0])
        @test norm(τ) < 1e-10  # No torque (at CoM)
        
        # Fire +x and -x thrusters equally → no net force
        u_cancel = @SVector [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        F_cancel, τ_cancel = compute_thruster_wrench(config, u_cancel)
        
        @test norm(F_cancel) < 1e-10
        @test norm(τ_cancel) < 1e-10
        
        # Fire all thrusters at 50%
        u_half = @SVector fill(0.5, 6)
        F_half, τ_half = compute_thruster_wrench(config, u_half)
        
        # Opposing thrusters cancel
        @test norm(F_half) < 1e-10
        @test norm(τ_half) < 1e-10
    end
    
    @testset "Thruster with offset (torque generation)" begin
        # Single thruster offset from CoM
        positions = [@SVector([0.5, 0.0, 0.0])]  # 0.5m in +x
        directions = [@SVector([0.0, 1.0, 0.0])]  # Fires in +y
        max_thrust = 1.0
        
        config = ThrusterConfiguration(positions, directions, max_thrust)
        
        u = @SVector [1.0]  # Full thrust
        F, τ = compute_thruster_wrench(config, u)
        
        # Force in +y direction
        @test F ≈ @SVector([0.0, 1.0, 0.0])
        
        # Torque: r × F = [0.5, 0, 0] × [0, 1, 0] = [0, 0, 0.5]
        @test τ ≈ @SVector([0.0, 0.0, 0.5]) atol=1e-10
    end
    
    @testset "Reaction wheel configuration" begin
        axes = [
            @SVector([1.0, 0.0, 0.0]),
            @SVector([0.0, 1.0, 0.0]),
            @SVector([0.0, 0.0, 1.0])
        ]
        
        max_torque = 0.1
        max_momentum = 10.0
        
        config = ReactionWheelConfiguration(axes, max_torque, max_momentum)
        
        # Check fields
        @test length(config.axes) == 3
        @test all(config.max_torque .≈ 0.1)
        @test all(config.max_momentum .≈ 10.0)
        
        # Check axis normalization
        for axis in config.axes
            @test norm(axis) ≈ 1.0
        end
    end
    
    @testset "Default research wheels" begin
        config = default_research_wheels(max_torque=0.2, max_momentum=15.0)
        
        # Should have 3 wheels
        @test length(config.max_torque) == 3
        
        # Check parameters
        for torque in config.max_torque
            @test torque ≈ 0.2
        end
        
        for momentum in config.max_momentum
            @test momentum ≈ 15.0
        end
    end
    
    @testset "Wheel torque computation" begin
        config = default_research_wheels(max_torque=0.1)
        
        # Command torque about x-axis
        u = @SVector [1.0, 0.0, 0.0]  # Full torque on x-wheel
        τ = compute_wheel_torque(config, u)
        
        @test τ ≈ @SVector([0.1, 0.0, 0.0])
        
        # Command torque about all axes
        u_all = @SVector [0.5, -0.3, 0.8]
        τ_all = compute_wheel_torque(config, u_all)
        
        @test τ_all ≈ 0.1 * u_all
        
        # Clamping test (control > 1)
        u_large = @SVector [2.0, 0.0, 0.0]
        τ_clamped = compute_wheel_torque(config, u_large)
        
        # Should clamp to max torque
        @test τ_clamped[1] ≈ 0.1
    end
    
    @testset "Combined actuator configuration" begin
        thrusters = default_research_thrusters(max_thrust=1.0)
        wheels = default_research_wheels(max_torque=0.1)
        
        config = ActuatorConfiguration(thrusters=thrusters, wheels=wheels)
        
        # Check control dimension
        n_u = control_dimension(config)
        @test n_u == 6 + 3  # 6 thrusters + 3 wheels
        
        # Compute wrench
        u = @SVector [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Thrusters
            0.5, -0.3, 0.0                  # Wheels
        ]
        
        F, τ = compute_wrench(config, u)
        
        # Force from thrusters only
        @test F ≈ @SVector([1.0, 0.0, 0.0])
        
        # Torque from wheels (thrusters at CoM contribute no torque)
        @test τ ≈ @SVector([0.05, -0.03, 0.0])
    end
    
    @testset "Thrusters-only configuration" begin
        thrusters = default_research_thrusters(max_thrust=2.0)
        config = ActuatorConfiguration(thrusters=thrusters)
        
        @test control_dimension(config) == 6
        @test !isnothing(config.thrusters)
        @test isnothing(config.wheels)
    end
    
    @testset "Wheels-only configuration" begin
        wheels = default_research_wheels(max_torque=0.2)
        config = ActuatorConfiguration(wheels=wheels)
        
        @test control_dimension(config) == 3
        @test isnothing(config.thrusters)
        @test !isnothing(config.wheels)
    end
    
    @testset "Default research actuators" begin
        config = default_research_actuators(
            thruster_force=1.5,
            wheel_torque=0.15
        )
        
        @test control_dimension(config) == 9
        @test !isnothing(config.thrusters)
        @test !isnothing(config.wheels)
        
        # Check parameters propagated
        @test config.thrusters.max_thrust[1] ≈ 1.5
        @test config.wheels.max_torque[1] ≈ 0.15
    end
end