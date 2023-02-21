# The implemented algorithms to compute person parameter estimates
# are tested against reported values from Rost (2004).
@testset "Person parameter estimation" begin
    @testset "Rasch Model" begin
        betas = [-1.17, -0.69, 0.04, 0.7, 1.12]
        @testset "Maximum Likelihood Estimation" begin
            pp_result = RaschModels._fit_personpars(RaschModel, betas, PersonParameterMLE())
            @test length(pp_result.values) == length(betas) + 1
            @test pp_result.values[1] === NaN
            @test pp_result.values[end] === NaN
            @test pp_result.values[2:(end-1)] ≈ [-1.59, -0.47, 0.48, 1.59] atol = 0.01
            @test pp_result.se[1] === NaN
            @test pp_result.se[end] === NaN
            @test pp_result.se[2:(end-1)] ≈ [1.18, 0.99, 0.99, 1.17] atol = 0.01
        end
        @testset "Warm's Weighted Likelihood Estimation" begin
            pp_result = RaschModels._fit_personpars(RaschModel, betas, PersonParameterWLE())
            @test length(pp_result.values) == length(betas) + 1
            @test pp_result.values ≈ [-2.77, -1.33, -0.41, 0.42, 1.33, 2.75] atol = 0.01
            @test pp_result.se ≈ [1.71, 1.11, 0.98, 0.98, 1.11, 1.71] atol = 0.01
        end
   end
end