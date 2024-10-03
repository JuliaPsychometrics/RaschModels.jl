const CUD = ContinuousUnivariateDistribution

"""
    $(TYPEDEF)

Prior distributions for Rasch models.

## Fields
$(FIELDS)

## Details

Item difficulties are parameterised as

```math
\\beta = \\mu_\\beta + \\beta_{\\mathrm{norm}} \\cdot \\sigma_\\beta
```
"""
@kwdef struct Prior{T<:CUD,U<:CUD,V<:CUD,W<:CUD,X<:CUD}
    mu_beta::T = Normal()
    sigma_beta::U = InverseGamma(3, 2)
    beta_norm::V = Normal()
    theta::W = Normal()
    tau::X = Normal()
end
