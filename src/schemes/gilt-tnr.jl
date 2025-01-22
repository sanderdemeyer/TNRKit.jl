mutable struct GILTTNR <: TRGScheme
    T::TensorMap

    ε::Float64
    function GILTTNR(T::TensorMap; ε=5e-8)
        return new(copy(T), ε)
    end
end

function step!(scheme::GILTTNR, trunc::TensorKit.TruncationScheme)
    # step 1: GILT
    giltscheme = GILT(scheme.T; ε=scheme.ε)
    for i in 1:20 # for now just do 20 steps
        @show i
        _step!(giltscheme, truncbelow(scheme.ε))
    end
    
    @info "passed gilt"
    U, S, V, _ = tsvd(giltscheme.T1, ((1, 2), (3, 4)); trunc=trunc)

    @plansor begin
        A[-1 -2; -3] := U[-1 -2; 1] * sqrt(S)[1; -3]
        B[-1; -2 -3] := sqrt(S)[-1; 1] * V[1; -2 -3]
    end

    U, S, V, _ = tsvd(giltscheme.T2, ((1, 4), (2, 3)); trunc=trunc)

    # Flip legs to their original domain (to mitigate space mismatch at the end)
    U = permute(U, ((1,), (2, 3)))
    V = permute(V, ((1, 2), (3,)))

    @plansor begin
        C[-1; -2 -3] := U[-1; -2 1] * sqrt(S)[1; -3]
        D[-1 -2; -3] := sqrt(S)[-1; 1] * V[1 -2; -3]
    end

    @tensor scheme.T[-1 -2; -3 -4] := D[-1 1; 4] * B[-2; 2 1] * C[2; 3 -3] * A[4 3; -4]
    return scheme
end

gilttnr_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::GILTTNR)
    println(io, "Gilt-TRN - GILT + TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * ε: $(scheme.ε)")
    return nothing
end
