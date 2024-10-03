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

        long_nodropped_missing = RaschModels.matrix_to_long(m, dropmissing = false)
        @test length(long_nodropped_missing) == 3
        @test eltype(long_nodropped_missing[1]) == Union{Missing,Int}
        @test isequal(long_nodropped_missing[1], [1, 0, missing, 1])
        @test long_nodropped_missing[2] == [1, 1, 2, 2]
        @test long_nodropped_missing[3] == [1, 2, 1, 2]
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

    @testset "getrowsums" begin
        m = [
            0 0 1
            0 1 0
            1 1 0
            1 1 1
            0 0 0
        ]
        P, I = size(m)
        rs = RaschModels.getrowsums(m)
        @test length(rs) == P
        @test minimum(rs) == 0
        @test maximum(rs) == I
        @test eltype(rs) == Int
        @test rs == [1, 1, 2, 3, 0]
    end

    @testset "getcolsums" begin
        m = [
            0 0 0 1
            0 0 1 0
            0 1 1 0
            0 1 1 1
            0 0 0 0
        ]
        P, I = size(m)
        cs = RaschModels.getcolsums(m)
        @test length(cs) == I
        @test minimum(cs) == 0
        @test maximum(cs) == 3
        @test eltype(cs) == Int
        @test cs == [0, 2, 3, 2]
    end
end
