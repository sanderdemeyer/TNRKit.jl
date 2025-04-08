mutable struct HOTRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap{E,S,2,2}; finalize=finalize!) where {E,S}
        return new(T, finalize)
    end
end

mutable struct HOTRG_single_impurity <: TNRScheme
    T::TensorMap
    S::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap, S::TensorMap; finalize=finalize!)
        return new(T, S, finalize)
    end
end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    # join vertically
    @tensor MMdag[-1 -2; -3 -4] := scheme.T[-1 5; 1 2] * scheme.T[-2 3; 5 4] *
                                   conj(scheme.T[-3 6; 1 2]) * conj(scheme.T[-4 3; 6 4])

    # get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc=trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc=trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the left, U on the right
    @tensor scheme.T[-1 -2; -3 -4] := scheme.T[1 5; -3 3] * conj(U[1 2; -1]) * U[3 4; -4] *
                                      scheme.T[2 -2; 5 4]

    # join horizontally
    @tensor MMdag[-1 -2; -3 -4] := scheme.T[1 -1; 2 5] * scheme.T[5 -2; 4 3] *
                                   conj(scheme.T[1 -3; 2 6]) * conj(scheme.T[6 -4; 4 3])

    # get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc=trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc=trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the bottom, U on top
    @tensor scheme.T[-1 -2; -3 -4] := scheme.T[-1 1; 3 5] * scheme.T[5 2; 4 -4] *
                                      conj(U[1 2; -2]) *
                                      U[3 4; -3]
    return scheme
end

function step!(scheme::HOTRG_single_impurity, trunc::TensorKit.TruncationScheme)
    @tensor MMdag[-1 -2; -3 -4] := scheme.T[-1 5; 2 1] * scheme.T[-2 3; 4 5] *
                                   adjoint(scheme.T)[4 6; -4 3] *
                                   adjoint(scheme.T)[2 1; -3 6]

    # Get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc=trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc=trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the left, U on the right
    @tensor scheme.T[-1 -2; -3 -4] := adjoint(U)[-1; 1 2] * scheme.T[1 5; 4 -4] *
                                      scheme.T[2 -2; 3 5] * U[4 3; -3]

    @tensor scheme.S[-1 -2; -3 -4] := 1/2* adjoint(U)[-1; 1 2] * scheme.S[1 5; 4 -4] *
                                      scheme.T[2 -2; 3 5] * U[4 3; -3] +
                                      1/2 * adjoint(U)[-1; 1 2] * scheme.T[1 5; 4 -4] *
                                      scheme.S[2 -2; 3 5] * U[4 3; -3]

    return scheme
end

function Base.show(io::IO, scheme::HOTRG)
    println(io, "HOTRG - Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end

function Base.show(io::IO, scheme::HOTRG_single_impurity)
    println(io, "HOTRG - Higher Order TRG with impurity")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * S: $(summary(scheme.S))")
    return nothing
end
