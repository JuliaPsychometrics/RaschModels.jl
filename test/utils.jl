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

    @testset "gettotals" begin
        s = [0, 4, 3, 2, 2, 0]
        totals = RaschModels.gettotals(s, minimum(s), maximum(s))
        @test length(totals) == length(0:4)
        @test eltype(totals) == Int
        @test totals == [2, 0, 2, 1, 1]

        totals_restricted = RaschModels.gettotals(s, 1, maximum(s))
        @test length(totals_restricted) == length(1:4)
        @test eltype(totals_restricted) == Int
        @test totals_restricted == [0, 2, 1, 1]
    end

    @testset "normalize_sumzero!" begin
        values = [0.0, -0.809, -0.494, -0.415, -0.651, -1.049, -0.412, -0.731, -0.613, -0.613] 
        vcov = [
            0.0 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000;
            0.0 0.167 0.088 0.088 0.088 0.086 0.088 0.088 0.088 0.088;
            0.0 0.088 0.167 0.088 0.088 0.086 0.088 0.088 0.088 0.088;
            0.0 0.088 0.088 0.167 0.088 0.086 0.088 0.088 0.088 0.088;
            0.0 0.088 0.088 0.088 0.166 0.086 0.088 0.088 0.088 0.088;
            0.0 0.088 0.088 0.088 0.088 0.137 0.096 0.096 0.096 0.096;
            0.0 0.088 0.088 0.088 0.088 0.096 0.136 0.096 0.096 0.096;
            0.0 0.088 0.088 0.088 0.088 0.096 0.096 0.135 0.096 0.096;
            0.0 0.088 0.088 0.088 0.088 0.096 0.096 0.096 0.135 0.096;
            0.0 0.088 0.088 0.088 0.088 0.096 0.096 0.096 0.096 0.135
        ]
        values_sumzero = [0.579, -0.23,  0.085, 0.164, -0.072, -0.47, 0.167, -0.153, -0.034, -0.034]
        vcov_sumzero = [
             0.078 -0.009 -0.009 -0.009 -0.009 -0.009 -0.009 -0.009 -0.009 -0.009;
            -0.009  0.071 -0.008 -0.008 -0.008 -0.008 -0.008 -0.008 -0.008 -0.008;
            -0.009 -0.008  0.071 -0.008 -0.008 -0.008 -0.008 -0.008 -0.008 -0.008;
            -0.009 -0.008 -0.008  0.071 -0.008 -0.008 -0.008 -0.008 -0.008 -0.008;
            -0.009 -0.008 -0.008 -0.008  0.070 -0.008 -0.008 -0.008 -0.008 -0.008;
            -0.009 -0.008 -0.008 -0.008 -0.008  0.041  0.000  0.000  0.000  0.000;
            -0.009 -0.008 -0.008 -0.008 -0.008  0.000  0.040  0.000  0.000  0.000;
            -0.009 -0.008 -0.008 -0.008 -0.008  0.000  0.000  0.039  0.000  0.000;
            -0.009 -0.008 -0.008 -0.008 -0.008  0.000  0.000  0.000  0.039  0.000;
            -0.009 -0.008 -0.008 -0.008 -0.008  0.000  0.000  0.000  0.000  0.039
        ]
        RaschModels.normalize_sumzero!(values, vcov)
        @test values ≈ values_sumzero atol=0.01
        @test vcov ≈ vcov_sumzero atol=0.01
    end
end
