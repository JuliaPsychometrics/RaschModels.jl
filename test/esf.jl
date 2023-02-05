# The implemented algorithms to compute elementary symmetric functions and 
# their derivatives are tested against reported values from Baker & Harwell (1996). 
# They start the computation by using product-normalized epsilon values (ϵᵢ = exp(-βᵢ)) 
# estimated for 5 items of the LSAT dataset (Bock & Lieberman, 1970) under the 
# Rasch Model using FORTRAN code (subroutine GAMMA) by Fischer & Formann (1972). 
@testset "Elementary symmetric functions" begin
    @testset "for dichotomous data" begin
        ϵ = [3.5118, .6219, .2905, .8450, 1.8648]
        γ0 = [1.0, 7.1340, 16.9493, 16.7781, 7.0529, 0.9997]
        γ1 = [
            1.0 1.0 1.0 1.0 1.0;
            3.6222 6.5121 6.8435 6.2890 5.2692;
            4.2288 12.8994 14.9612 11.6351 7.1233;
            1.9273 8.7560 12.4319 6.9465 3.4946;
            0.2847 1.6076 3.4414 1.1831 0.5361;
            0.0 0.0 0.0 0.0 0.0  
        ]
        γ2_diags = [
            1.0 2.1840 3.3989 5.5851 0.9997;
            0.0 6.5121 1.1240 4.2622 0.9997;
            0.0 0.1807 14.9612 2.4283 0.9997;
            0.0 0.5255 1.4725 6.9465 0.9997;
            0.0 1.1597 2.6971 5.3337 0.5361
        ]
        @testset "Summation Algorithm" begin
            esf_sum = RaschModels.esf(ϵ, SummationAlgorithm())
            @test esf_sum.γ0 ≈ γ0 atol=.001
            @test esf_sum.γ1 ≈ γ1 atol=.001 
            @test size(esf_sum.γ2) == (6, 5, 5)
            @test all([esf_sum.γ2[:, i, i] == esf_sum.γ1[:, i] for i in 1:5])
            @test all([isapprox(diag(esf_sum.γ2[:, :, i]), γ2_diags[i, :]; atol=.001) for i in 1:5])
        end
    end
end

