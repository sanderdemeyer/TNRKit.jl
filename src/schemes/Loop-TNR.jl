mutable struct Loop_TNR <: TRGScheme
    # data
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function TRG(TA::TensorMap, TB::TensorMap; finalize=finalize!)
        return new(TA, TB, finalize)
    end
end

function make_psi(scheme::Loop_TNR)
    #TODO: change ordering of incoming and ordering legs and fuse the physical legs if necessary
    psi = [scheme.TA, permute(scheme.TB, (4,1),(2,3)), permute(scheme.TA, (3,4),(1,2)), permute(scheme.TB, (2,3),(4,1))]
    return psi
end


#Entanglement filtering step 

function QR_L(L::TensorMap, T::TensorMap)
    @tensor temp[-1 -2; -3 -4] := L[-2; 1]*T[-1 1; -3 -4]
    _, R = leftorth(temp, (1,2,4),(3,); alg = QR())
    return R
end

function QR_R(R::TensorMap, T::TensorMap)
    @tensor temp[-1 -2; -3 -4] := T[-1 -2; 1 -4]*R[1; -3]
    L, _ = rightorth(temp, (2,),(1,3,4); alg = LQ())
    return L
end

function maximumer(T::TensorMap)
    maxi = []
    for (_, b) in blocks(T)
        push!(maxi, maximum(abs.(b[])))
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
            error = norm(new_L - L)
        end
        L = new_L
        steps += 1
        crit = steps < maxsteps && error > minerror
    end

    return L
end

function find_R(pos::Int, psi::Array, maxsteps::Int, minerror::Float64)
    R = id(space(psi[mod(pos-2,4)+1])[3])
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
            error = norm(new_R - R)
        end
        R = new_R
        steps += 1
        crit = steps < maxsteps && error > minerror
    end

    return R
end





