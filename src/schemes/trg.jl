mutable struct TRG <: TRGScheme
    # data
    T::TensorMap

    # run! parameters
    crit::stopcrit
    finalize!::Function

    function TRG(T::TensorMap; stop=maxiter(100), f=finalize!)
        return new(T, stop, f)
    end
end

function step!(scheme::TRG, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(scheme.T, (1, 4), (2, 3); trunc=trunc)

    # make the legs normal again
    U = permute(U, (1,), (2, 3))
    V = permute(V, (1, 2), (3,))

    @plansor begin
        A[-1; -2 -3] := U[-1; -2 1] * sqrt(S)[1; -3]
        B[-1 -2; -3] := sqrt(S)[-1; 1] * V[1 -2; -3]
    end

    U, S, V, _ = tsvd(scheme.T, (1, 2), (3, 4); trunc=trunc)

    # spaces are already correct
    @plansor begin
        C[-1 -2; -3] := U[-1 -2; 1] * sqrt(S)[1; -3]
        D[-1; -2 -3] := sqrt(S)[-1; 1] * V[1; -2 -3]
    end

    @tensor T′[-1 -2; -3 -4] := B[-1 1; 4] * D[-2; 2 1] * A[2; 3 -3] * C[4 3; -4]
    scheme.T = T′
    return scheme
end

function finalize!(scheme::TRG)
    n = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= n
    return n
end

# example convcrit function
trg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))