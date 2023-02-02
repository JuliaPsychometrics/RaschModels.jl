@testset "PolytomousRaschModel" begin
    n_iter = 100
    n_categories = 4
    n_items = 2

    X = rand(1:n_categories, 10, n_items)
    rsm_mcmc = fit(RatingScaleModel, X, MH(), n_iter)
    rsm_mle = fit(RatingScaleModel, X, MLE())

    PRM = RaschModels.PolytomousRaschModel

    # scoring function that maps k=1,...,K -> 0,...,1
    partial_credit(k, K=n_categories) = (k - 1) / (K - 1)

    @testset "irf" begin
        @test length(RaschModels._irf(PRM, zeros(2), zeros(1))) == 2
        @test length(RaschModels._irf(PRM, zeros(3), zeros(2))) == 3

        @test RaschModels._irf(PRM, zeros(2), zeros(1)) == [0.5, 0.5]
        @test RaschModels._irf(PRM, zeros(2), [1e9]) ≈ [0, 1]
        @test RaschModels._irf(PRM, zeros(2), [-1e9]) ≈ [1, 0]

        @test RaschModels._irf(PRM, zeros(3), zeros(2)) == fill(1 / 3, 3)
        @test RaschModels._irf(PRM, zeros(3), [0, 1e9]) ≈ [0, 0, 1]
        @test RaschModels._irf(PRM, zeros(3), [-1e9, 0]) ≈ [1, 0, 0]

        @test length(irf(rsm_mcmc, 0.0, 1, 1)) == n_iter
        @test size(irf(rsm_mcmc, 0.0, 1)) == (n_iter, n_categories)

        @test irf(rsm_mle, 0.0, 1, 1) isa Float64
        @test length(irf(rsm_mle, 0.0, 1)) == n_categories

        # implementation
        @test all(irf(rsm_mcmc, -1e9, 1, 1) .≈ 1)
        @test all(irf(rsm_mcmc, -1e9, 1, n_categories) .≈ 0)
        @test all(irf(rsm_mcmc, 1e9, 1, 1) .≈ 0)
        @test all(irf(rsm_mcmc, 1e9, 1, n_categories) .≈ 1)

        @test irf(rsm_mle, -1e9, 1, 1) ≈ 1
        @test irf(rsm_mle, -1e9, 1, n_categories) ≈ 0
        @test irf(rsm_mle, 1e9, 1, 1) ≈ 0
        @test irf(rsm_mle, 1e9, 1, n_categories) ≈ 1
    end

    @testset "iif" begin
        @test length(RaschModels._iif(PRM, [0.5, 0.5], 0)) == 1
        @test length(RaschModels._iif(PRM, fill(1 / 3, 3), 0)) == 1

        @test RaschModels._icif(PRM, 0, 10) == 0
        @test RaschModels._icif(PRM, 0.5, 10) == 5

        @test length(iif(rsm_mcmc, 0.0, 1, 1)) == n_iter
        @test size(iif(rsm_mcmc, 0.0, 1)) == (n_iter, n_categories)

        @test iif(rsm_mle, 0.0, 1, 1) isa Float64
        @test length(iif(rsm_mle, 0.0, 1)) == n_categories

        # implementation
    end

    @testset "expected_score" begin
        @test all(expected_score(rsm_mcmc, -1e9, 1) .≈ 1)
        @test all(expected_score(rsm_mcmc, 1e9, 1) .≈ n_categories)

        @test all(expected_score(rsm_mcmc, -1e9, 1:2) .≈ 2)
        @test all(expected_score(rsm_mcmc, 1e9, 1:2) .≈ 2 * n_categories)

        @test all(expected_score(rsm_mcmc, -1e9) .≈ n_items)
        @test all(expected_score(rsm_mcmc, 1e9) .≈ n_categories * n_items)

        @test expected_score(rsm_mcmc, 0.0) == expected_score(rsm_mcmc, 0.0, scoring_function=identity)
        @test expected_score(rsm_mcmc, 0.0, 1) == expected_score(rsm_mcmc, 0.0, 1, scoring_function=identity)
        @test expected_score(rsm_mcmc, 0.0, scoring_function=x -> 0) == zeros(n_iter)

        @test all(expected_score(rsm_mcmc, -1e9, scoring_function=partial_credit) .≈ 0)
        @test all(expected_score(rsm_mcmc, 1e9, scoring_function=partial_credit) .≈ n_items)

        @test expected_score(rsm_mle, -1e9, 1) ≈ 1
        @test expected_score(rsm_mle, 1e9, 1) ≈ n_categories

        @test expected_score(rsm_mle, -1e9, 1:2) ≈ 2
        @test expected_score(rsm_mle, 1e9, 1:2) ≈ 2 * n_categories

        @test expected_score(rsm_mle, -1e9) ≈ n_items
        @test expected_score(rsm_mle, 1e9) ≈ n_categories * n_items

        @test expected_score(rsm_mle, 0.0) == expected_score(rsm_mle, 0.0, scoring_function=identity)
        @test expected_score(rsm_mle, 0.0, 1) == expected_score(rsm_mle, 0.0, 1, scoring_function=identity)
        @test expected_score(rsm_mle, 0.0, scoring_function=x -> 0) == 0

        @test expected_score(rsm_mle, -1e9, scoring_function=partial_credit) ≈ 0
        @test expected_score(rsm_mle, 1e9, scoring_function=partial_credit) ≈ n_items
    end

    @testset "information" begin
        @test all(information(rsm_mcmc, -1e9, 1) .≈ 0)
        @test all(information(rsm_mcmc, 1e9, 1) .≈ 0)

        @test all(information(rsm_mcmc, -1e9, 1:2) .≈ 0)
        @test all(information(rsm_mcmc, 1e9, 1:2) .≈ 0)

        @test all(information(rsm_mcmc, -1e9) .≈ 0)
        @test all(information(rsm_mcmc, 1e9) .≈ 0)

        @test information(rsm_mcmc, 0.0) == information(rsm_mcmc, 0.0, scoring_function=identity)
        @test information(rsm_mcmc, 0.0, 1) == information(rsm_mcmc, 0.0, 1, scoring_function=identity)
        @test information(rsm_mcmc, 0.0, scoring_function=x -> 0) == zeros(n_iter)

        @test all(information(rsm_mcmc, -1e9, scoring_function=partial_credit) .≈ 0.0)
        @test all(information(rsm_mcmc, 1e9, scoring_function=partial_credit) .≈ 0.0)

        @test information(rsm_mle, -1e9, 1) ≈ 0
        @test information(rsm_mle, 1e9, 1) ≈ 0

        @test information(rsm_mle, -1e9, 1:2) ≈ 0
        @test information(rsm_mle, 1e9, 1:2) ≈ 0

        @test information(rsm_mle, -1e9) ≈ 0
        @test information(rsm_mle, 1e9) ≈ 0

        @test information(rsm_mle, 0.0) == information(rsm_mle, 0.0, scoring_function=identity)
        @test information(rsm_mle, 0.0, 1) == information(rsm_mle, 0.0, 1, scoring_function=identity)
        @test information(rsm_mle, 0.0, scoring_function=x -> 0) == 0.0

        @test information(rsm_mle, -1e9, scoring_function=partial_credit) ≈ 0.0
        @test information(rsm_mle, 1e9, scoring_function=partial_credit) ≈ 0.0
    end
end
