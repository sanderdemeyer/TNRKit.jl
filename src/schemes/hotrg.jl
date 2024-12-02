mutable struct HOTRG <: TRGScheme
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap)
        new(T, finalize!)
    end
end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    # Contract along the horizontal direction
    @tensor M[-1 -2 -3; -4 -5 -6] := scheme.T[-1 1; -5 -6] * scheme.T[-2 -3; -4 1]

    # Get unitaries
    U, _, _, εₗ = tsvd(M, (1, 2), (3, 4, 5, 6); trunc=trunc)
    UR, _, _, εᵣ = tsvd(M, (4, 5), (1, 2, 3, 6); trunc=trunc)

    if εₗ > εᵣ
        U = UR
    end

    # adjoint(U) on the left, U on the right
    @tensor scheme.T[-1 -2; -3 -4] := adjoint(U)[-1; 1 2] * M[1 2 -2; 3 4 -4] * U[4 3; -3]

    return scheme
end

function finalize!(scheme::HOTRG)
    n = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= n

    # turn the tensor by 90 degrees
    scheme.T = permute(scheme.T, ((2, 3), (4, 1)))

    return n
end