@testset "RatingScaleModel" begin
    data = rand(1:3, 100, 2)

    rsm_mle = fit(RatingScaleModel, data, MLE())
    @test estimation_type(rsm_mle) == PointEstimate
    @test rsm_mle.parnames_beta == [Symbol("beta[1]"), Symbol("beta[2]")]
    @test rsm_mle.parnames_tau == [Symbol("tau[1]"), Symbol("tau[2]")]

    rsm_mcmc = fit(RatingScaleModel, data, MH(), 100)
    @test estimation_type(rsm_mcmc) == SamplingEstimate
    @test rsm_mcmc.parnames_beta == [Symbol("beta[1]"), Symbol("beta[2]")]
    @test rsm_mcmc.parnames_tau == [Symbol("tau[1]"), Symbol("tau[2]")]
end
