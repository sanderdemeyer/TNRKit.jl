mutable struct HOTRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap; finalize=finalize!)
        return new(T, finalize)
    end
end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    @plansor MMdag[-1 -2; -3 -4] := scheme.T[-1 5; 1 2] * scheme.T[-2 3; 5 4] *
                                    conj(scheme.T[-3 6; 1 2]) * conj(scheme.T[-4 3; 6 4])

    # Get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc=trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc=trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the left, U on the right
    @plansor scheme.T[-1 -2; -3 -4] := scheme.T[1 5; -3 3] * conj(U[1 2; -1]) * U[3 4; -4] *
                                       scheme.T[2 -2; 5 4]
    return scheme
end

# example convcrit function
hotrg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::HOTRG)
    println(io, "HOTRG - Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
