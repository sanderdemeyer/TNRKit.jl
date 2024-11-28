mutable struct BTRG <: TRGScheme
    # data
    T::TensorMap
    S1::TensorMap
    S2::TensorMap
    k::Float64

    finalize!::Function
    function BTRG(T::TensorMap, k::Number; finalize=finalize!)
        # Construct S1 and S2 as identity matrices.
        new(T, id(space(T, 1)), id(space(T, 1)), k, finalize)
    end
end

function pseudopow(t::TensorMap, a::Real; tol=eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (c, b) in blocks(t′)
        @inbounds for I in LinearAlgebra.diagind(b)
            b[I] = b[I] < tol ? b[I] : b[I]^a
        end
    end
    return t′
end


function step!(scheme::BTRG, trunc::TensorKit.TruncationScheme)
    U, S, V, _ϵ = tsvd(scheme.T, (1, 4), (2, 3); trunc=trunc)

    # make the legs normal again
    U = permute(U, (1,), (2, 3))
    V = permute(V, (1, 2), (3,))

    S_a = pseudopow(S, (1 - scheme.k) / 2)
    S_b = pseudopow(S, scheme.k)

    @plansor begin
        A[-1; -2 -3] := U[-1; -2 1] * S_a[1; -3]
        B[-1 -2; -3] := S_a[-1; 1] * V[1 -2; -3]
        E[-1; -2] := S_b[-1; -2]
    end

    U, S, V, _ϵ = tsvd(scheme.T, (1, 2), (3, 4); trunc=trunc)

    # spaces are already correct
    S_a = pseudopow(S, (1 - scheme.k) / 2)
    S_b = pseudopow(S, scheme.k)

    @plansor begin
        C[-1 -2; -3] := U[-1 -2; 1] * S_a[1; -3]
        D[-1; -2 -3] := S_a[-1; 1] * V[1; -2 -3]
        F[-1; -2] := S_b[-1; -2]
    end

    S1 = scheme.S1
    S2 = scheme.S2

    @tensor T′[-1 -2; -3 -4] := B[-1 1; 8] * S1[2; 1] * D[-2; 3 2] * S2[3; 4] * A[4; 5 -3] * S1[5; 6] * C[7 6; -4] * S2[8; 7]

    # Turn and mirror
    T′ = permute(T′, (2, 1), (4, 3))

    scheme.T = T′
    scheme.S1 = E
    scheme.S2 = F
    return scheme
end

function finalize!(scheme::BTRG)
    # Turn and mirror
    scheme.T = permute(scheme.T, (2, 1), (4, 3))
    # calculate norm
    n = norm(@tensor scheme.T[1 2; 3 4] * scheme.S1[4; 2] * scheme.S2[3; 1])
    scheme.T /= n
    return n
end

# example convcrit function
btrg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))