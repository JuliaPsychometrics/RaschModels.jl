@testset "Item Response Function" begin
    @test RaschModels._irf(0.0, 0.0, 0) == RaschModels._irf(0.0, 0.0, 1)
    @test RaschModels._irf(0.0, 0.0, 1) == 0.5
end
