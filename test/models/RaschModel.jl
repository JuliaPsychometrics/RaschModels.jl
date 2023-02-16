@testset "RaschModel" begin
    X = rand(0:1, 10, 3)
    model_mcmc = fit(RaschModel, X, MH(), 100)
    model_mle = fit(RaschModel, X, MLE())

    @testset "Model construction" begin
        @test estimation_type(model_mcmc) == SamplingEstimate
        @test model_mcmc.parnames_beta ==
              [Symbol("beta[1]"), Symbol("beta[2]"), Symbol("beta[3]")]

        @test estimation_type(model_mle) == PointEstimate
        @test model_mle.parnames_beta ==
              [Symbol("beta[1]"), Symbol("beta[2]"), Symbol("beta[3]")]
    end

    @testset "irf" begin
        @test RaschModels._irf(RaschModel, 0.0, 0.0, 0) ==
              RaschModels._irf(RaschModel, 0.0, 0.0, 1)
        @test RaschModels._irf(RaschModel, 0.0, 0.0, 1) == 0.5
        @test RaschModels._irf(RaschModel, 0.0, -99.9, 1) > 0.99
        @test RaschModels._irf(RaschModel, 0.0, 99.9, 1) < 0.001

        @test all(irf(model_mcmc, -99.9, 1, 1) .< 0.001)
        @test all(irf(model_mcmc, -99.9, 1, 0) .> 0.99)

        @test irf(model_mle, -99.9, 1, 1) < 0.001
        @test irf(model_mle, -99.9, 1, 0) > 0.99
    end

    @testset "iif" begin
        @test RaschModels._iif(RaschModel, 0.0, 0.0) == 0.25
        @test RaschModels._iif(RaschModel, 0.0, 99.9) < 0.001
        @test RaschModels._iif(RaschModel, 0.0, -99.9) < 0.001
        @test RaschModels._iif(RaschModel, 0.0, 0.0) ==
              RaschModels._iif(RaschModel, 0.0, 0.0; scoring_function = identity)

        @test all(iif(model_mcmc, -99.9, 1, 1) .< 0.001)
        @test all(iif(model_mcmc, -99.9, 1, 0) .< 0.001)

        @test iif(model_mle, -99.9, 1, 1) < 0.001
        @test iif(model_mle, -99.9, 1, 0) < 0.001
    end

    @testset "expected_score" begin
        @test expected_score(model_mcmc, 0.0, 1) == irf(model_mcmc, 0.0, 1)
        @test expected_score(model_mcmc, 0.0) ==
              sum(irf(model_mcmc, 0.0, i) for i in 1:size(X, 2))
        @test expected_score(model_mcmc, 0.0) ==
              expected_score(model_mcmc, 0.0; scoring_function = identity)
        @test expected_score(model_mcmc, 0.0, scoring_function = x -> 0) == zeros(100)
        @test expected_score(model_mcmc, 0.0, 1, scoring_function = x -> 2x) ==
              irf(model_mcmc, 0.0, 1) * 2
        @test expected_score(model_mcmc, 0.0, 1:2, scoring_function = x -> 2x) ==
              irf(model_mcmc, 0.0, 1) .* 2 .+ irf(model_mcmc, 0.0, 2) .* 2

        @test expected_score(model_mle, 0.0, 1) == irf(model_mle, 0.0, 1)
        @test expected_score(model_mle, 0.0) ==
              sum(irf(model_mle, 0.0, i) for i in 1:size(X, 2))
        @test expected_score(model_mle, 0.0) ==
              expected_score(model_mle, 0.0; scoring_function = identity)
        @test expected_score(model_mle, 0.0, scoring_function = x -> 0) == 0
        @test expected_score(model_mle, 0.0, 1; scoring_function = x -> 2x) ==
              irf(model_mle, 0.0, 1) * 2
        @test expected_score(model_mle, 0.0, 1:2, scoring_function = x -> 2x) ==
              irf(model_mle, 0.0, 1) * 2 + irf(model_mle, 0.0, 2) * 2
    end

    @testset "information" begin
        @test information(model_mcmc, 0.0, 1) == iif(model_mcmc, 0.0, 1)
        @test information(model_mcmc, 0.0) ==
              sum(iif(model_mcmc, 0.0, i) for i in 1:size(X, 2))
        @test information(model_mcmc, 0.0) ==
              information(model_mcmc, 0.0, scoring_function = identity)
        @test information(model_mcmc, 0.0, scoring_function = x -> 0) == zeros(100)

        info_1 =
            RaschModels._iif.(
                RaschModel,
                0.0,
                RaschModels.getitempars(model_mcmc, 1),
                scoring_function = x -> 2x,
            )
        info_2 =
            RaschModels._iif.(
                RaschModel,
                0.0,
                RaschModels.getitempars(model_mcmc, 2),
                scoring_function = x -> 2x,
            )

        @test information(model_mcmc, 0.0, 1, scoring_function = x -> 2x) == info_1
        @test information(model_mcmc, 0.0, 1:2, scoring_function = x -> 2x) ==
              info_1 .+ info_2

        @test information(model_mle, 0.0, 1) == iif(model_mle, 0.0, 1)
        @test information(model_mle, 0.0) ==
              sum(iif(model_mle, 0.0, i) for i in 1:size(X, 2))
        @test information(model_mle, 0.0) ==
              information(model_mle, 0.0, scoring_function = identity)
        @test information(model_mle, 0.0, scoring_function = x -> 0) == 0

        info_1 = RaschModels._iif(
            RaschModel,
            0.0,
            RaschModels.getitempars(model_mle, 1),
            scoring_function = x -> 2x,
        )
        info_2 = RaschModels._iif(
            RaschModel,
            0.0,
            RaschModels.getitempars(model_mle, 2),
            scoring_function = x -> 2x,
        )

        @test information(model_mle, 0.0, 1, scoring_function = x -> 2x) == info_1
        @test information(model_mle, 0.0, 1:2, scoring_function = x -> 2x) ==
              info_1 + info_2
    end
end
