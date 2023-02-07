function turing_model(T::Type{<:AbstractRaschModel}; priors::Prior = Prior())
    return _turing_model(T; priors)
end
