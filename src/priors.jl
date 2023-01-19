"""
    Prior

Prior distributions for Rasch models.

## Fields
- `theta`: The prior distribution for the person ability distribution
- `mu_beta`: The prior distribution for the intercept of the item difficulty distribution
- `sigma_beta`: The prior distribution for the standard deviation of the item difficulty distribution
- `beta_norm`: The prior distribution for the standardized item difficulty distribution (see details)
- `tau`: The prior distribution for threshold parameters in the rating scale model

## Details

Item difficulties are parameterised as

```math
\\beta = \\mu_\\beta + \\beta_{\\mathrm{norm}} \\cdot \\sigma_\\beta
```
"""
Base.@kwdef struct Prior
    mu_beta::ContinuousUnivariateDistribution = Normal()
    sigma_beta::ContinuousUnivariateDistribution = InverseGamma(3, 2)
    beta_norm::ContinuousUnivariateDistribution = Normal()
    theta::ContinuousUnivariateDistribution = Normal()
    tau::ContinuousUnivariateDistribution = Normal()
end
