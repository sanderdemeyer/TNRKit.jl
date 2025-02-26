# This implementation is entirely based on the original code: https://github.com/Gilt-TNR/Gilt-TNR
# With just the paper (https://doi.org/10.1103/PhysRevB.97.045111) as a reference, it is almost impossible to get the details right.
mutable struct GILT
    T1::TensorMap
    T2::TensorMap

    ε::Float64
    function GILT(T::TensorMap; ε=5e-8)
        return new(copy(T), copy(T), ε)
    end
end

function _step!(scheme::GILT, trunc::TensorKit.TruncationScheme)
    # Environment: bottom leg broken
    S², U = environment_spectrum(scheme, Val{:S})

    @plansor t[-1] := U[1 1; -1]

    epsid = scheme.ε^2 * id(domain(S²))

    @tensor t′[-1] := t[1] * (S² * inv(epsid + S²))[1; -1]

    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]
    R′ = transpose(R′)
    U, S, V, _ = tsvd(R′; trunc=trunc)
    sqrtS = sqrt(S)

    @tensor scheme.T1[-1 -2; -3 -4] := sqrtS[1; -3] * U[2; 1] * scheme.T1[-1 -2; 2 -4]
    @tensor scheme.T2[-1 -2; -3 -4] := sqrtS[-1; 1] * V[1; 2] * scheme.T2[2 -2; -3 -4]

    nS = maximum(abs.((S - id(domain(S))).data))

    # Environment: top leg broken
    S², U = environment_spectrum(scheme, Val{:N})
    @plansor t[-1] := U[1 1; -1]

    epsid = scheme.ε^2 * id(domain(S²))

    @tensor t′[-1] := t[1] * (S² * inv(epsid + S²))[1; -1]

    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]

    U, S, V, _ = tsvd(R′; trunc=trunc)
    sqrtS = sqrt(S)

    @tensor scheme.T2[-1 -2; -3 -4] := scheme.T2[-1 -2; 1 -4] * U[1; 2] * sqrtS[2; -3]
    @tensor scheme.T1[-1 -2; -3 -4] := sqrtS[-1; 1] * V[1; 2] * scheme.T1[2 -2; -3 -4]

    nN = maximum(abs.((S - id(domain(S))).data))

    # Environment: right leg broken
    S², U = environment_spectrum(scheme, Val{:E})

    @plansor t[-1] := U[1 1; -1]

    epsid = scheme.ε^2 * id(domain(S²))

    @tensor t′[-1] := t[1] * (S² * inv(epsid + S²))[1; -1]

    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]
    R′ = transpose(R′)
    U, S, V, _ = tsvd(R′; trunc=trunc)
    sqrtS = sqrt(S)

    @tensor scheme.T1[-1 -2; -3 -4] := scheme.T1[-1 1; -3 -4] * V[2; 1] * sqrtS[-2; 2]
    @tensor scheme.T2[-1 -2; -3 -4] := sqrtS[2; -4] * U[1; 2] * scheme.T2[-1 -2; -3 1]

    nE = maximum(abs.((S - id(domain(S))).data))

    # Environment: left leg broken
    S², U = environment_spectrum(scheme, Val{:W})

    @plansor t[-1] := U[1 1; -1]

    epsid = scheme.ε^2 * id(domain(S²))

    @tensor t′[-1] := t[1] * (S² * inv(epsid + S²))[1; -1]

    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]
    U, S, V, _ = tsvd(R′; trunc=trunc)
    sqrtS = sqrt(S)

    @tensor scheme.T1[-1 -2; -3 -4] := scheme.T1[-1 -2; -3 1] * U[1; 2] * sqrtS[2; -4] # space mismatch
    @tensor scheme.T2[-1 -2; -3 -4] := sqrtS[-2; 1] * V[1; 2] * scheme.T2[-1 2; -3 -4]

    nW = maximum(maximum(abs.((S - id(domain(S))).data)))

    return scheme, (nS, nN, nE, nW)
end

function apply_gilt!(scheme::GILT, direction::Symbol, trunc::TensorKit.TruncationScheme)
    S², U = environment_spectrum(scheme, Val{direction})
    S² /= sum(S².data)

    R′ = optimize_R′(U, S², scheme.ε, trunc)
    if direction in (:E, :S)
        R′ = transpose(R′)
    end

    R′1, s, R′2 = tsvd(R′; trunc=trunc) # consider changing this to truncbelow(scheme.ε*1e-3) like in the original code

    apply_R′!(scheme, R′1, R′2, direction)

    done = maximum(maximum(abs.((s).data))) - 1 < 1e-2 # TODO: make this user configurable

    return done
end

function optimize_R′(U, S, ε, trunc::TensorKit.TruncationScheme)
    @plansor t[-1] := U[1 1; -1]
    epsid = ε^2 * id(domain(S))
    @tensor t′[-1] := t[1] * (S * inv(epsid + S))[1; -1]
    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]

    u, s, v, _ = tsvd(R′; trunc=trunc)
    done_recursing = maximum(maximum(abs.((s).data))) - 1 < 1e-2 # TODO: make this user configurable
    if !done_recursing
        sqrts = sqrt(s)
        @tensor begin
            us[-1; -2] := u[-1; 1] * sqrts[1; -2]
            sv[-1; -2] := sqrts[-1; 1] * v[1; -2]
            Uuvs[-1 -2; -3] := U[1, 2; -3] * us[1; -1] * sv[-2; 2]
            UuvsS[-1 -2; -3] := Uuvs[-1 -2; 1] * S[1; -3]
        end
        Uinner, Sinner, _, _ = tsvd(UuvsS)
        Sinner /= sum(Sinner.data)
        Rpinner = optimize_R′(Uinner, Sinner, ε, trunc)
        @tensor R′[-1; -2] := us[-1; 1] * Rpinner[1; 2] * sv[2; -2]
    end
    return R′
end

function environment_corners(scheme::GILT)
    @tensor NW[-1 -2; -3 -4] := adjoint(scheme.T2)[-1 2; 1 -4] * scheme.T2[1 -2; -3 2]
    @tensor NE[-1 -2; -3 -4] := scheme.T1[-1 -2; 1 2] * adjoint(scheme.T1)[1 2; -3 -4]
    @tensor SE[-1 -2; -3 -4] := scheme.T2[-1 2; 1 -4] * adjoint(scheme.T2)[1 -2; -3 2]
    @tensor SW[-1 -2; -3 -4] := adjoint(scheme.T1)[-1 -2; 1 2] * scheme.T1[1 2; -3 -4]
    return NW, NE, SE, SW
end

function environment_spectrum(scheme::GILT, location::Type{Val{:N}})
    NW, NE, SE, SW = environment_corners(scheme)
    @tensor E[-1 -2; -3 -4] := NW[-3 2; -1 1] * NE[-2 6; -4 5] * SE[4 5; 3 6] * SW[3 1; 4 2]
    return eigh(E)
end
function environment_spectrum(scheme::GILT, location::Type{Val{:E}})
    NW, NE, SE, SW = environment_corners(scheme)
    @tensor E[-1 -2; -3 -4] := NW[1 4; 2 3] * NE[2 -1; 1 -3] * SE[6 -4; 5 -2] * SW[5 3; 6 4]
    return eigh(E)
end
function environment_spectrum(scheme::GILT, location::Type{Val{:S}})
    NW, NE, SE, SW = environment_corners(scheme)
    @tensor E[-1 -2; -3 -4] := NW[3 6; 4 5] * NE[4 2; 3 1] * SE[-1 1; -3 2] * SW[-4 5; -2 6]
    return eigh(E)
end
function environment_spectrum(scheme::GILT, location::Type{Val{:W}})
    NW, NE, SE, SW = environment_corners(scheme)
    @tensor E[-1 -2; -3 -4] := NW[5 -2; 6 -4] * NE[6 4; 5 3] * SE[2 3; 1 4] * SW[1 -3; 2 -1]
    return eigh(E)
end

function apply_R′!(scheme::GILT, R′1::TensorMap, R′2::TensorMap, direction::Symbol)
    if direction == :N
        @tensor contractcheck=true scheme.T2[-1 -2; -3 -4] := scheme.T2[-1 -2; 1 -4] * R′1[1; -3]
        @tensor contractcheck=true scheme.T1[-1 -2; -3 -4] := R′2[-1; 1] * scheme.T1[1 -2; -3 -4]
    elseif direction == :E #requires some transposing
        @tensor contractcheck=true scheme.T1[-1 -2; -3 -4] := scheme.T1[-1 1; -3 -4] * R′2[-2; 1]
        @tensor contractcheck=true scheme.T2[-1 -2; -3 -4] := R′1[1; -4] * scheme.T2[-1 -2; -3 1]    
    elseif direction == :S #requires some transposing
        @tensor contractcheck=true scheme.T1[-1 -2; -3 -4] := R′1[1; -3] * scheme.T1[-1 -2; 1 -4]
        @tensor contractcheck=true scheme.T2[-1 -2; -3 -4] := R′2[-1; 1] * scheme.T2[1 -2; -3 -4]
    elseif direction == :W
        @tensor contractcheck=true scheme.T1[-1 -2; -3 -4] := scheme.T1[-1 -2; -3 1] * R′1[1; -4]
        @tensor contractcheck=true scheme.T2[-1 -2; -3 -4] := R′2[-2; 1] * scheme.T2[-1 1; -3 -4]
    else
        throw(ArgumentError("Invalid direction: $direction"))
    end
    return scheme
end

function Base.show(io::IO, scheme::GILT)
    println(io, "GILT - Graph Independent Local Truncation")
    println(io, "  * T1: $(summary(scheme.T1))")
    println(io, "  * T2: $(summary(scheme.T2))")
    println(io, "  * ε: $(scheme.ε)")
    return nothing
end
