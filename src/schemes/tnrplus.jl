mutable struct TNRplus <: TNRScheme
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function TNRplus(TA::TensorMap, TB::TensorMap)
        return new(TA, TB, finalize!)
    end
    function TNRplus(T::TensorMap, finalize!)
        return new(T, copy(T), finalize!)
    end
end

function Ψ_A(scheme::LoopTNR)
    psi = AbstractTensorMap[permute(scheme.TA, ((2,), (1, 3, 4))),
                            permute(scheme.TB, ((1,), (3, 4, 2))),
                            permute(scheme.TA, ((3,), (4, 2, 1))),
                            permute(scheme.TB, ((4,), (2, 1, 3)))]
    return psi
end

#Utility functions for QR decomp

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @tensor temp[-1; -2 -3 -4] := L[-1; 1] * T[1; -2 -3 -4]
    _, Rt = leftorth(temp, ((1, 2, 3), (4,)))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @tensor temp[-1; -2 -3 -4] := T[-1; -2 -3 1] * R[1; -4]
    Lt, _ = rightorth(temp, ((1,), (2, 3, 4)))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @tensor temp[-1; -2 -3] := L[-1; 1] * T[1; -2 -3]
    _, Rt = leftorth(temp, ((1, 2), (3,)))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @tensor temp[-1; -2 -3] := T[-1; -2 1] * R[1; -3]
    Lt, _ = rightorth(temp, ((1,), (2, 3)))
    return Lt
end

#Maximum function that works for any TensorMap
function maximumer(T::TensorMap)
    maxi = []
    for (_, d) in blocks(T)
        push!(maxi, maximum(abs.(d)))
    end
    return maximum(maxi)
end

#Functions to find the left and right projectors

function find_L(pos::Int, psi::Array, entanglement_criterion::stopcrit)
    L = id(space(psi[pos])[1])
    crit = true
    steps = 0
    error = [Inf]
    n = length(psi)
    while crit
        new_L = copy(L)
        for i in (pos - 1):(pos + n - 2)
            new_L = QR_L(new_L, psi[i % n + 1])
        end
        new_L = new_L / maximumer(new_L)

        if space(new_L) == space(L)
            push!(error, abs(norm(new_L - L)))
        end

        L = new_L
        steps += 1
        crit = entanglement_criterion(steps, error)
    end

    return L
end

function find_R(pos::Int, psi::Array, entanglement_criterion::stopcrit)
    n = length(psi)
    if numin(psi[mod(pos - 2, n) + 1]) == 2
        R = id(space(psi[mod(pos - 2, n) + 1])[3]')
    else
        R = id(space(psi[mod(pos - 2, n) + 1])[4]')
    end
    crit = true
    steps = 0
    error = [Inf]
    while crit
        new_R = copy(R)

        for i in (pos - 2):-1:(pos - n - 1)
            new_R = QR_R(new_R, psi[mod(i, n) + 1])
        end
        new_R = new_R / maximumer(new_R)

        if space(new_R) == space(R)
            push!(error, abs(norm(new_R - R)))
        end
        R = new_R
        steps += 1
        crit = entanglement_criterion(steps, error)
    end

    return R
end

function P_decomp(R::TensorMap, L::TensorMap, trunc::TensorKit.TruncationScheme)
    @tensor temp[-1; -2] := L[-1; 1] * R[1; -2]
    U, S, V, _ = tsvd(temp, ((1,), (2,)); trunc=trunc)
    re_sq = pseudopow(S, -0.5)

    @tensor PR[-1; -2] := R[-1; 1] * adjoint(V)[1; 2] * re_sq[2; -2]
    @tensor PL[-1; -2] := re_sq[-1; 1] * adjoint(U)[1; 2] * L[2; -2]

    return PR, PL
end

function find_projectors(psi::Array, entanglement_criterion::stopcrit,
                         trunc::TensorKit.TruncationScheme)
    PR_list = []
    PL_list = []
    n = length(psi)
    for i in 1:n
        L = find_L(i, psi, entanglement_criterion)

        R = find_R(i, psi, entanglement_criterion)

        pr, pl = P_decomp(R, L, trunc)

        push!(PR_list, pr)
        push!(PL_list, pl)
    end
    return PR_list, PL_list
end

#Functions to construct Ψ_B

function one_loop_projector(phi::Array, pos::Int, trunc::TensorKit.TruncationScheme)
    L = id(space(phi[1])[1])
    R = id(space(phi[end])[4]')
    for i in 1:pos
        L = QR_L(L, phi[i])
    end
    for i in length(phi):-1:(pos + 1)
        R = QR_R(R, phi[i])
    end
    PR, PL = P_decomp(R, L, trunc)
    return PR, PL
end

function Ψ_B(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    ΨB = []

    for i in 1:4
        s1, s2 = SVD12(ΨA[i], trunc)
        push!(ΨB, s1)
        push!(ΨB, s2)
    end




    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(100) & convcrit(1e-12, ΨB_function)
    PR_list, PL_list = find_projectors(ΨB, criterion, trunc)

    ΨB_disentangled = []
    for i in 1:8
        @tensor B1[-1; -2 -3] := PL_list[i][-1; 1] * ΨB[i][1; -2 2] *
                                 PR_list[mod(i, 8) + 1][2; -3]
        push!(ΨB_disentangled, B1)
    end
    return ΨB_disentangled
end


