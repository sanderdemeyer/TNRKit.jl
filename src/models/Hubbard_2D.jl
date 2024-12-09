function R_tensor()
    R = zeros(ComplexF64, 2,2,2,2,2,2,2,2,2,2,2,2)
    for (pi11, pi12, pj11, pj12, pi21, pi22, i11, i12, j11, j12, i21, i22) in Iterators.product([0:1 for _ in 1:12]...)
        R[pi11+1, pi12+1, pj11+1, pj12+1, pi21+1, pi22+1, i11+1, i12+1, j11+1, j12+1, i21+1, i22+1] = 
            i11*i21 + i12*(i21 + pj11 + pi21 + pi11 + j11 + i22) + j11*(i21 + pj11 + pi21 + pi11) + 
            j12*(i21 + pj11 + pi21 + pi11 + i22 + pj12 + pi22 + pi12) + i22*(pj11 + pi21 + pi11) + pi22*(pj11 + pi21 + pi11 + pj12) +
            pi21*pj11 + pj12*(pj11 + pi11) + pi12*pi11
    end
    return R
end

function Hubbard2D(μ::Number,t::Number, ϵ::Number, U::Number)
    δ(x, y) = ==(x, y)
    T = zeros(ComplexF64, 2,2,2,2,2,2,2,2,2,2,2,2)
    R = R_tensor()
    V = Vect[FermionParity](0 => 1, 1 => 1)
    
    for (pi11, pi12, pj11, pj12, pi21, pi22, i11, i12, j11, j12, i21, i22) in Iterators.product([0:1 for _ in 1:12]...)
        r = R[pi11+1, pi12+1, pj11+1, pj12+1, pi21+1, pi22+1, i11+1, i12+1, j11+1, j12+1, i21+1, i22+1]
        T[pi11+1, pi12+1, pj11+1, pj12+1, pi21+1, pi22+1, i21+1, i22+1, i11+1, j11+1, i12+1, j12+1] = 
        ((-1)^r)*((-1)^(i11 + i12))*(sqrt(t*ϵ)^(i11 + i12 + j11 + j12 + pi11 + pi12 +pj11 + pj12))*
        (δ(1, i22 + i12 + pj12)*δ(1, pi22 + pi12 + j12)*δ(1, i21 + i11 +pj11)*δ(1, pi21 + pi11 + j11) - 
        (μ*ϵ + 1)*())
            
    end
    
end
