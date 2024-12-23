mutable struct Loop_TNR <: TRGScheme
    # data
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function Loop_TNR(TA::TensorMap, TB::TensorMap; finalize=finalize!)
        return new(TA, TB, finalize)
    end
end

function make_psi(scheme::Loop_TNR)
    psi = [permute(scheme.TA, (4,2), (3,1)), permute(scheme.TB, (3,1),(2,4)), permute(scheme.TA, (2,4),(1,3)), permute(scheme.TB, (1,3),(4,2))]
    return psi
end


#Entanglement filtering step 

function QR_L(L::TensorMap, T::TensorMap)
    @tensor temp[-1 -2; -3 -4] := L[-2; 1]*T[-1 1; -3 -4]
    _, Rt = leftorth(temp, (1,2,4),(3,))
    return Rt
end

function QR_R(R::TensorMap, T::TensorMap)
    @tensor temp[-1 -2; -3 -4] := T[-1 -2; 1 -4]*R[1; -3]
    Lt, _ = rightorth(temp, (2,),(1,3,4))
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
    L = id(space(psi[pos])[2])
    
    crit = true
    steps = 0
    error = Inf

    while crit 
        
        new_L = copy(L)
        for i = pos-1:pos+2
            new_L = QR_L(new_L, psi[i%4 + 1])
        end
        new_L = new_L/maximumer(new_L)

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
    R = id(space(psi[mod(pos-2,4)+1])[3]')
    crit = true
    steps = 0
    error = Inf

    while crit 
        new_R = copy(R)
        
        for i = pos-2:-1:pos-5
            
            new_R = QR_R(new_R, psi[mod(i,4) + 1])
        end
        new_R = new_R/maximumer(new_R)

        if space(new_R) == space(R)
            error = abs(norm(new_R - R))
        end
        R = new_R
        steps += 1
        crit = steps < maxsteps && error > minerror
    end

    return R
end


function P_decomp(R::TensorMap, L::TensorMap, trunc::TensorKit.TruncationScheme)
    @tensor temp[-1; -2] := L[-1; 1]*R[1; -2]
    U, S, V, _ = tsvd(temp, (1,), (2,); trunc = trunc)
    re_sq = pseudopow(S, -0.5)
    
    @tensor PR[-1;-2] := R[-1, 1]*adjoint(V)[1;2]*re_sq[2, -2]
    @tensor PL[-1;-2] := re_sq[-1, 1]*adjoint(U)[1;2]*L[2, -2]

    return PR, PL
end

function find_projectors(psi::Array, maxsteps::Int, minerror::Float64, trunc::TensorKit.TruncationScheme)
    PR_list = []
    PL_list = []
    for i = 1:4
        
        L = find_L(i, psi, maxsteps, minerror)
        
        R = find_R(i, psi, maxsteps, minerror)
        
        pr, pl = P_decomp(R, L, trunc)
        
        push!(PR_list, pr)
        push!(PL_list, pl)
    end
    return PR_list, PL_list
end

function entanglement_filtering!(scheme::Loop_TNR, maxsteps::Int, minerror::Float64, trunc::TensorKit.TruncationScheme)
    psi = make_psi(scheme)
    

    PR_list, PL_list = find_projectors(psi, maxsteps, minerror, trunc)
    
    @tensor psi1[-1 -2; -3 -4] := psi[1][1 2; 3 4]*PL_list[3][-1;1]*PL_list[1][-2;2]*PR_list[2][3; -3]*PR_list[4][4; -4]
    TA = permute(psi1, (4,2),(3,1))
    @tensor psi2[-1 -2; -3 -4] := psi[2][1 2; 3 4]*PL_list[4][-1; 1]*PL_list[2][-2; 2]*PR_list[3][3;-3]*PR_list[1][4;-4]
    
    TB = permute(psi2, (2,3),(1,4))
    U1 = isometry(flip(space(TA)[1]),space(TA)[1])
    Udg1 = adjoint(U1)
    U2 = isometry(flip(space(TB)[2]),space(TB)[2])
    Udg2 = adjoint(U2)

    @tensor scheme.TA[-1 -2; -3 -4] := TA[1 -2; -3 4]*U1[-1;1]*Udg2[4; -4]
    @tensor scheme.TB[-1 -2; -3 -4] := TB[-1 2; 3 -4]*U2[-2; 2]*Udg1[3; -3]
    return scheme
end



function finalize!(scheme::Loop_TNR)
    n1 = norm(@tensor scheme.TA[1 2; 1 2])
    n2 = norm(@tensor scheme.TB[1 2; 1 2])
    scheme.TA /= n1
    scheme.TB /= n2
    return n1, n2
end


