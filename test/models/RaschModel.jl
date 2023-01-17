@testset "RaschModel" begin
    X = rand(0:1, 10, 3)
    model_mcmc = fit(RaschModel, X, MH(), 100)
    model_mle = fit(RaschModel, X, MLE())

    @testset "irf" begin
        @test RaschModels._irf(0.0, 0.0, 0) == RaschModels._irf(0.0, 0.0, 1)
        @test RaschModels._irf(0.0, 0.0, 1) == 0.5
        @test RaschModels._irf(0.0, -99.9, 1) > 0.99
        @test RaschModels._irf(0.0, 99.9, 1) < 0.001
    end

    @testset "iif" begin
        @test RaschModels._iif(0.0, 0.0, 0) == RaschModels._iif(0.0, 0.0, 1)
        @test RaschModels._iif(0.0, 0.0, 1) == 0.25
        @test RaschModels._iif(0.0, 99.9, 1) < 0.001
        @test RaschModels._iif(0.0, -99.9, 1) < 0.001
    end

    @testset "expected_score" begin
        @test expected_score(model_mcmc, 0.0, 1) == irf(model_mcmc, 0.0, 1)
        @test expected_score(model_mle, 0.0, 1) == irf(model_mle, 0.0, 1)
        @test expected_score(model_mcmc, 0.0) == sum(irf(model_mcmc, 0.0, i) for i in 1:size(X, 2))
        @test expected_score(model_mle, 0.0) == sum(irf(model_mle, 0.0, i) for i in 1:size(X, 2))
    end

    @testset "information" begin
        @test information(model_mcmc, 0.0, 1) == iif(model_mcmc, 0.0, 1)
        @test information(model_mle, 0.0, 1) == iif(model_mle, 0.0, 1)
        @test information(model_mcmc, 0.0) == sum(iif(model_mcmc, 0.0, i) for i in 1:size(X, 2))
        @test information(model_mle, 0.0) == sum(iif(model_mle, 0.0, i) for i in 1:size(X, 2))
    end
end