
using LinearAlgebra: I

function triangle_bad()
    M = zeros((2,2,2,2))
    M[1,1,1,2]=1.0
    M[2,1,1,1]=1.0
    M[2,1,1,2]=1.0
    M[2,2,2,1]=1.0
    M[1,2,2,2]=1.0
    M[1,2,2,1]=1.0
    T = TensorMap(M, ℂ^2⊗ℂ^2←ℂ^2⊗ℂ^2)
    
    return T

end

function triangle_good(β)
    ham = [1. -1;-1 1]
    w1 = exp.(- β * ham)
    w2 = exp.(- β * ham/2)

    m = reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,w2,w2,w1),4,4,4,4)
    
    return TensorMap(m, ℂ^4⊗ℂ^4←ℂ^4⊗ℂ^4)
end

function triangle_bad_2(β)
    
    ham = ComplexF64[1. -1;-1 1]
    w = exp.(- β * ham)
    we = ham .* w

    m = reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),we, w , w ),4,4,4,4) + 
        reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w , we, w ),4,4,4,4) +
        reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w , w , we),4,4,4,4)
    
    return TensorMap(m, ℂ^4⊗ℂ^4←ℂ^4⊗ℂ^4)
end
function triangle_bad_3(β)  
    
    ham = zeros(ComplexF64, 2,2,2)
    ham[1,1,1] = ham[2,2,2] = 4
    w = exp.(- β * ham)

    m = reshape(ein"ijk,kl,nl,ml->ijnm"(w,I(2),I(2),I(2)),2,2,2,2)
    
    return TensorMap(m, ℂ^2⊗ℂ^2←ℂ^2⊗ℂ^2)
end