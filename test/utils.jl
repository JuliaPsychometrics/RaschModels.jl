@testset "Utility functions" begin
    @testset "matrix_to_long" begin
        m = rand(0:1, 2, 2)
        long = RaschModels.matrix_to_long(m)
        @test length(long) == 3
        @test eltype(long[1]) == Int
        @test long[1] == vec(m)
        @test long[2] == [1, 1, 2, 2]
        @test long[3] == [1, 2, 1, 2]

        m = [1 missing; 0 1]
        long_dropped_missing = RaschModels.matrix_to_long(m)
        @test length(long_dropped_missing) == 3
        @test eltype(long_dropped_missing[1]) == Int
        @test long_dropped_missing[1] == [1, 0, 1]
        @test long_dropped_missing[2] == [1, 1, 2]
        @test long_dropped_missing[3] == [1, 2, 2]

        long_nodropped_missing = RaschModels.matrix_to_long(m, dropmissing=false)
        @test length(long_nodropped_missing) == 3
        @test eltype(long_nodropped_missing[1]) == Union{Missing,Int}
        @test isequal(long_nodropped_missing[1], [1, 0, missing, 1])
        @test long_nodropped_missing[2] == [1, 1, 2, 2]
        @test long_nodropped_missing[3] == [1, 2, 1, 2]
    end

    @testset "betanames" begin
        @test RaschModels.betanames(1) == [Symbol("beta[1]")]
        @test RaschModels.betanames(2) == [Symbol("beta[1]"), Symbol("beta[2]")]
    end

    @testset "taunames" begin
        @test RaschModels.taunames(1) == [Symbol("tau[1]")]
        @test RaschModels.taunames(2) == [Symbol("tau[1]"), Symbol("tau[2]")]
        @test RaschModels.taunames(2, item=1) == [Symbol("tau[1][1]"), Symbol("tau[1][2]")]
    end
end
