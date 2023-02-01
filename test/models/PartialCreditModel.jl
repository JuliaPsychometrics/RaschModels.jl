@testset "PartialCreditModel" begin
    data = rand(1:3, 100, 2)

    pcm_mle = fit(PartialCreditModel, data, MLE())
    @test estimation_type(pcm_mle) == PointEstimate
    @test pcm_mle.parnames_beta == [Symbol("beta[1]"), Symbol("beta[2]")]
    @test pcm_mle.parnames_tau == [
        [Symbol("tau[1][1]"), Symbol("tau[1][2]")],
        [Symbol("tau[2][1]"), Symbol("tau[2][2]")]
    ]
end
