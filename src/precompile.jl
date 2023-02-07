using SnoopPrecompile

SnoopPrecompile.verbose[] = true

@precompile_setup begin
    data_dich = rand(0:1, 50, 3)
    data_poly = rand(1:4, 50, 3)

    @precompile_all_calls begin
        rasch_cml = fit(RaschModel, data_dich, CML())
        irf(rasch_cml, 0.0, 1)
        iif(rasch_cml, 0.0, 1)
        expected_score(rasch_cml, 0.0)
        information(rasch_cml, 0.0)

        # rasch_mcmc = fit(RaschModel, data_dich, NUTS(200, 0.65), 200)
        # irf(rasch_mcmc, 0.0, 1)
        # iif(rasch_mcmc, 0.0, 1)
        # expected_score(rasch_mcmc, 0.0)
        # information(rasch_mcmc, 0.0)
    end
end
