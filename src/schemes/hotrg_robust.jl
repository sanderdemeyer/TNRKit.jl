mutable struct HOTRG_robust <: TRGScheme
    T::TensorMap

    finalize!::Function
    function HOTRG_robust(T::TensorMap; finalize=finalize!)
        new(T, finalize)
    end
end

function step!(scheme::HOTRG_robust, trunc::TensorKit.TruncationScheme)
    # Contract along the horizontal direction
    @tensor M[-1 -2 -3; -4 -5 -6] := scheme.T[-1 1; -5 -6] * scheme.T[-2 -3; -4 1]

    # Get projectors
    _, R1 = leftorth(M, (1,2,3,6), (4,5))
    R2, _ = rightorth(M, (1,2),(3,4,5,6))

    @tensor temp[-1; -2] := R1[-1; 1 2] * R2[2 1; -2]
    U, S, V, _ = tsvd(temp, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S, -0.5)
    @tensor Proj_1[-1 -2; -3] := R2[-1 -2; 1] * adjoint(V)[1; 2] * inv_s[2; -3]
    @tensor Proj_2[-1; -2 -3] := inv_s[-1; 1] * adjoint(U)[1; 2] * R1[2; -2 -3]


    # adjoint(U) on the left, U on the right
    @tensor scheme.T[-1 -2; -3 -4] := Proj_2[-1; 1 2] * M[1 2 -2; 3 4 -4] * Proj_1[3 4; -3]

    return scheme
end


function finalize!(scheme::HOTRG_robust)
    n = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= n

    # turn the tensor by 90 degrees
    scheme.T = permute(scheme.T, ((2, 3), (4, 1)))

    return n
end

# example convcrit function
hotrg_robust_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))