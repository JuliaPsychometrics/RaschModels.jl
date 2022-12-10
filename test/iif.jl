@testset "Item Information Function" begin
    @test RaschModels._iif(0.0, 0.0, 0) == RaschModels._iif(0.0, 0.0, 1)
    @test RaschModels._iif(0.0, 0.0, 1) == 0.25

    for theta in randn(10)
        @test RaschModels._iif(theta, 0.0, 1) == RaschModels._irf(theta, 0.0, 1) * RaschModels._irf(theta, 0.0, 0)
    end
end
