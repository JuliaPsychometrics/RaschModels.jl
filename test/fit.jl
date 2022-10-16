@testset "Model fitting" begin
    @testset "MCMC Sampling" begin
        P = 10
        I = 2
        data = rand(0:1, P, I)
        m = fit(RaschModel, data, MH())
        parnames = names(m.pars, :parameters)

        @test m.pars isa Chains

        # item parameter names
        for i in 1:I
            @test Symbol("beta[$i]") in parnames
        end

        @test :mu_beta in parnames
        @test :sigma_beta in parnames

        # person parameter names
        for p in 1:P
            @test Symbol("theta[$p]") in parnames
        end
    end
end
