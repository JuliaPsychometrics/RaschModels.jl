using RaschModels

@time begin
    data = rand(0:1, 50, 3)
    rasch_cml = fit(RaschModel, data, CML())

    irf(rasch_cml, 0.0, 1)
    iif(rasch_cml, 0.0, 1)
    expected_score(rasch_cml, 0.0)
    information(rasch_cml, 0.0)
end
