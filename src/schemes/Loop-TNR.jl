mutable struct Loop_TNR <: TRGScheme
    # data
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function Loop_TNR(TA::TensorMap, TB::TensorMap; finalize=finalize!)
        return new(TA, TB, finalize)
    end
end

# Base.ndims(::AbstractTensorMap{S,N1,N2}) where {S,N1,N2} = N1 + N2


function psiA_old(scheme::Loop_TNR)
    psi = [permute(scheme.TA, (4, 2), (3, 1)), permute(scheme.TB, (3, 1), (2, 4)), permute(scheme.TA, (2, 4), (1, 3)), permute(scheme.TB, (1, 3), (4, 2))]
    return psi
end



function psiA_new(scheme::Loop_TNR)
    psi = AbstractTensorMap[permute(scheme.TA, (2,), (3, 4, 1)), permute(scheme.TB, (1,), (2, 3, 4)), permute(scheme.TA, (4,), (1, 2, 3)), permute(scheme.TB, (3,), (4, 1, 2))]
    return psi
end

#Entanglement filtering step 

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,2,2}) where {E,S}
    @tensor temp[-1 -2; -3 -4] := L[-2; 1] * T[-1 1; -3 -4]
    _, Rt = leftorth(temp, (1, 2, 4), (3,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,2,2}) where {E,S}
    @tensor temp[-1 -2; -3 -4] := T[-1 -2; 1 -4] * R[1; -3]
    Lt, _ = rightorth(temp, (2,), (1, 3, 4))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @tensor temp[-1; -2 -3 -4] := L[-1; 1] * T[1; -2 -3 -4]
    _, Rt = leftorth(temp, (1, 3, 4), (2,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @tensor temp[-1; -2 -3 -4] := T[-1; 1 -3 -4] * R[1; -2]
    Lt, _ = rightorth(temp, (1,), (2, 3, 4))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @tensor temp[-1; -2 -3] := L[-1; 1] * T[1; -2 -3]
    _, Rt = leftorth(temp, (1, 3), (2,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @tensor temp[-1; -2 -3] := T[-1; 1 -3] * R[1; -2]
    Lt, _ = rightorth(temp, (1,), (2, 3))
    return Lt
end



function maximumer(T::TensorMap)
    maxi = []
    for (_, d) in blocks(T)
        push!(maxi, maximum(abs.(d)))
    end
    return maximum(maxi)
end

function find_L(pos::Int, psi::Array, maxsteps::Int, minerror::Float64)
    # L = id(space(psi[pos])[2])
    L = id(space(psi[pos])[1])
    crit = true
    steps = 0
    error = Inf
    n = length(psi)
    while crit

        new_L = copy(L)
        for i = pos-1:pos+n-2
            new_L = QR_L(new_L, psi[i%n+1])
        end
        new_L = new_L / maximumer(new_L)

        if space(new_L) == space(L)
            error = abs(norm(new_L - L))

        end

        L = new_L
        steps += 1
        crit = steps < maxsteps && error > minerror
    end

    return L
end

function find_R(pos::Int, psi::Array, maxsteps::Int, minerror::Float64)
    # R = id(space(psi[mod(pos-2,4)+1])[3]')
    R = id(space(psi[mod(pos - 2, 4)+1])[2]')
    crit = true
    steps = 0
    error = Inf
    n = length(psi)
    while crit
        new_R = copy(R)

        for i = pos-2:-1:pos-n-1

            new_R = QR_R(new_R, psi[mod(i, n)+1])
        end
        new_R = new_R / maximumer(new_R)

        if space(new_R) == space(R)
            error = abs(norm(new_R - R))
        end
        R = new_R
        steps += 1
        crit = steps < maxsteps && error > minerror
    end

    return R
end


function P_decomp(R::TensorMap, L::TensorMap)
    @tensor temp[-1; -2] := L[-1; 1] * R[1; -2]
    U, S, V, _ = tsvd(temp, (1,), (2,))
    re_sq = pseudopow(S, -0.5)

    @tensor PR[-1; -2] := R[-1, 1] * adjoint(V)[1; 2] * re_sq[2, -2]
    @tensor PL[-1; -2] := re_sq[-1, 1] * adjoint(U)[1; 2] * L[2, -2]

    return PR, PL
end

function find_projectors(psi::Array, maxsteps::Int, minerror::Float64)
    PR_list = []
    PL_list = []
    n = length(psi)
    for i = 1:n

        L = find_L(i, psi, maxsteps, minerror)

        R = find_R(i, psi, maxsteps, minerror)

        pr, pl = P_decomp(R, L)

        push!(PR_list, pr)
        push!(PL_list, pl)
    end
    return PR_list, PL_list
end



function entanglement_filtering!(scheme::Loop_TNR, maxsteps::Int, minerror::Float64)
    psi = psiA_new(scheme)
    PR_list, PL_list = find_projectors(psi, maxsteps, minerror)


    TA = copy(scheme.TA)
    TB = copy(scheme.TB)

    @tensor scheme.TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PR_list[4][1; -1] * PL_list[1][-2; 2] * PR_list[2][3; -3] * PL_list[3][-4; 4] 
    @tensor scheme.TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PL_list[2][-1; 1] * PR_list[3][2; -2]  * PL_list[4][-3; 3]  * PR_list[1][4; -4]

    return scheme
end



#Constructing initial PsiB

function one_loop_projector(phi::Array, pos::Int)
    L = id(space(phi[1])[1])
    R = id(space(phi[end])[2]')
    for i = 1:pos
        L = QR_L(L, phi[i])
    end
    for i = length(phi):-1:pos+1
        R = QR_R(R, phi[i])
    end
    PR, PL = P_decomp(R, L)
    return PR, PL
end

function SVD12(T::AbstractTensorMap{E,S,1,3}, d_cut::Int) where {E,S}
    U, s, V, _ = tsvd(T, (1, 4), (2, 3), trunc = truncdim(d_cut))
    @tensor S1[-1; -2 -3] := U[-1 -3; 1] * sqrt(s)[1; -2]
    @tensor S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
    return S1, S2
end


function psiB(scheme::Loop_TNR, d_cut::Int)
    psiA = psiA_new(scheme)
    psi = psiA_new(scheme)
    psiB_new = []

    for i = 1:4
        s1, s2 = SVD12(psiA[i], d_cut)
        phi = copy(psi)
        popat!(phi, i)
        insert!(phi, i, s1)
        insert!(phi, i + 1, s2)

        pr, pl = one_loop_projector(phi, i)

        @tensor B1[-1; -2 -3] := s1[-1; 1 -3] * pr[1; -2]
        @tensor B2[-1; -2 -3] := pl[-1; 1] * s2[1; -2 -3]
        push!(psiB_new, B1)
        push!(psiB_new, B2)
    end
    return psiB_new
end



#cost functions


function const_C(psiA)
    @tensor tmp[-1 -2; -3 -4] := psiA[1][-1; -3 1 2] * adjoint(psiA[1])[-4 1 2; -2]
    for i = 2:4
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiA[i][1; -3 3 4] * adjoint(psiA[i])[-4 3 4; 2]
    end
    return @tensor tmp[1 2; 1 2]
end

function tN(pos, psiB)
    @tensor tmp[-1 -2; -3 -4] := psiB[mod(pos, 8)+1][-1; -3 1]*adjoint(psiB[mod(pos, 8)+1])[-4 1; -2]
    for i = pos+1:pos+6
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiB[mod(i, 8)+1][1; -3 3] * adjoint(psiB[mod(i, 8)+1])[-4 3; 2]
    end
    return tmp
end

function tW(pos, psiA, psiB)
    next_a = mod(ceil(Int, pos/2), 4) + 1
    next_b = mod(2*ceil(Int, pos/2) + 1, 8)
    
    @tensor tmp[-1 -2; -3 -4] := psiA[next_a][-1; -3 1 2] * adjoint(psiB[next_b])[3 2;-2] * adjoint(psiB[next_b + 1])[-4 1; 3]
    for i = next_a: next_a + 1
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiA[mod(i,4) + 1][1; -3 3 4] * adjoint(psiB[2*(mod(i,4) + 1) - 1])[5 4; 2]* adjoint(psiB[2*(mod(i,4) + 1)])[-4 3; 5]
    end

    if pos % 2 == 0
        @tensor W[-1;-2 -3] := tmp[1 -1; 2 3] * psiA[ceil(Int, pos/2)][2; 1 -3 4] *adjoint(psiB[pos - 1])[-2 4; 3]
    else
        @tensor W[-1; -2 -3] := tmp[1 3; 2 -2] * psiA[ceil(Int, pos/2)][2; 1 4 -3] *adjoint(psiB[pos + 1])[3 4; -1]
    end

    return W
end

function cost_func(pos, psiA, psiB)
    C = const_C(psiA)
    N = tN(pos, psiB)
    TNT = norm(@tensor N[1 2; 3 4]* psiB[pos][3; 1 5] * adjoint(psiB[pos])[2 5; 4])
    W = tW(pos, psiA, psiB)
    WdT = norm(@tensor W[3; 1 2] * adjoint(psiB[pos])[3 2; 1])
    dWT = norm(@tensor adjoint(W)[1 2; 3]*psiB[pos][1; 3 2])
    
    return C + TNT - WdT - dWT
end

#optimization


function opt_T(N, W)

    function apply_f(x::TensorMap, ::Val{false})
        @tensor b[-1; -2 -3] := N[1 -1; 2 -2]*x[2;1 -3]
        return b
    end
    function apply_f(b::TensorMap, ::Val{true})
        @tensor x[-1; -2 -3] := adjoint(N)[-1 1; -2 2]*b[2; 1 -3]
        return x
    end
    
    new_T, info = lssolve((apply_f), W, LSMR(500, 1e-12, 1))


    #new_N = permute(N, (1,3), (2, 4))
    #new_inv_N = permute(pinv(new_N), (4,2), (3,1))
    #@tensor psi_B[pos][-1; -2 -3] := new_inv_N[-1 1; -2 2] * W[2; 1 -3]
    return new_T
end

function loop_opt!(scheme::Loop_TNR, maxsteps_opt::Int, minerror_opt::Float64, d_cut::Int)
    psi_A = psiA_new(scheme)
    psi_B = psiB(scheme, d_cut)
    
    cost = Inf
    sweep = 0
    while abs(cost) > minerror_opt && sweep < maxsteps_opt
        for i = 1:8
            N = tN(i, psi_B)
            W = tW(i, psi_A, psi_B)
            new_N = permute(N, (1,3), (2, 4))
            new_inv_N = permute(pinv(new_N), (4,2), (3,1))
            #new_T = opt_T(N, W)
            @tensor psi_B[i][-1; -2 -3] := new_inv_N[-1 1; -2 2] * W[2; 1 -3]
        end
        sweep += 1
        @show sweep
        cost = cost_func(1, psi_A, psi_B)
        @show cost
    end
    @tensor scheme.TA[-1 -2; -3 -4] := psi_B[5][4;-1 1] * psi_B[8][-2; 2 1] * psi_B[1][2; -3 3] * psi_B[4][-4; 4 3]
    @tensor scheme.TB[-1 -2; -3 -4] := psi_B[2][-1; 1 4] * psi_B[3][1; -2 2] * psi_B[6][-3; 3 2] * psi_B[7][3; -4 4]
    return scheme
end

function step!(scheme::Loop_TNR, d_cut::Int, maxsteps::Int, minerror::Float64, maxsteps_opt::Int, minerror_opt::Float64)
    entanglement_filtering!(scheme, maxsteps, minerror)
    #entanglement_filtering!(scheme, maxsteps, minerror)
    loop_opt!(scheme, maxsteps_opt, minerror_opt, d_cut)
    return scheme
end

function finalize!(scheme::Loop_TNR)
    
    n = norm(@tensor opt=true scheme.TA[1 2; 3 4]*scheme.TB[3 5; 1 6]*scheme.TB[7 4; 8 2]*scheme.TA[8 6; 7 5])

    scheme.TA /= n^(1/4)
    scheme.TB /= n^(1/4)
    return n^(1/4)
end




