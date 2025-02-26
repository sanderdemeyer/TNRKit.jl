mutable struct GILTTNR <: TNRScheme
    T::TensorMap

    ε::Float64
    stopgilt::stopcrit
    finalize!::Function
    function GILTTNR(T::TensorMap; ε=5e-8, stopgilt=giltcrit() & maxiter(50),
                     finalize=finalize!)
        return new(copy(T), ε, stopgilt, finalize)
    end
end

function step!(scheme::GILTTNR, trunc::TensorKit.TruncationScheme)
    # step 1: GILT
    giltscheme = GILT(scheme.T; ε=scheme.ε)

    gilt_steps = 1
    inner_counter = 1
    crit = false
    done_legs = Dict(direction => false for direction in (:N, :E, :S, :W))

    @infov 3 "Starting GILT\n$(giltscheme)\n"
    t = @elapsed while !crit
        for direction in (:S, :N, :E, :W)
            done = apply_gilt!(giltscheme, direction, trunc)
            done_legs[direction] = done
            @infov 4 "GILT step $gilt_steps.$inner_counter, legs: N: $(done_legs[:N]), E: $(done_legs[:E]), S: $(done_legs[:S]), W: $(done_legs[:W])"
            inner_counter += 1
        end
        inner_counter = 1
        gilt_steps += 1 # a gilt step is a full sweep
        crit = scheme.stopgilt(gilt_steps, done_legs)
    end

    @infov 3 "GILT finished\n $(stopping_info(scheme.stopgilt, gilt_steps, done_legs))\n Elapsed time: $(t)s\n Iterations: $gilt_steps"

    # step 2: TRG
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

    @tensor scheme.T[-1 -2; -3 -4] := D[-1 1; 4] * B[-2; 3 1] * C[3; 2 -3] * A[4 2; -4]
    return scheme
end

function Base.show(io::IO, scheme::GILTTNR)
    println(io, "Gilt-TRN - GILT + TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * ε: $(scheme.ε)")
    return nothing
end

# stopping criterion for GILT
struct giltcrit <: stopcrit end

(crit::giltcrit)(steps::Int, done_legs) = all(values(done_legs))

function stopping_info(crit::giltcrit, steps::Int, data)
    return "Gilt criterion reached: all legs converged"
end
