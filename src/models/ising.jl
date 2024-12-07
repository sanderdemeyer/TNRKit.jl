function classical_ising(β::Number; h=0)
    function σ(i::Int64)
        return 2i-3
    end
        
    T_array = [exp(β * (σ(i)σ(j) + σ(j)σ(k) + σ(k)σ(l) + σ(l)σ(i)) + h/2 * β * (σ(i) + σ(j) + σ(k) + σ(l))) for i in 1:2, j in 1:2, k in 1:2, l in 1:2]
    
    T = TensorMap(T_array, ℝ^2⊗ℝ^2←ℝ^2⊗ℝ^2)
        
    return T
end

function classical_ising_symmetric(β)
    V = Vect[Z2Irrep](0=>1,1=>1)
    Ising = zeros(2,2,2,2)
    c = cosh(β)
    s = sinh(β)
    for i=1:2
        for j=1:2
            for k=1:2
                for l=1:2
                    if (i+j+k+l)==4
                        Ising[i,j,k,l]=2*c*c
                    elseif (i+j+k+l)==6
                        Ising[i,j,k,l]=2*c*s
                    elseif (i+j+k+l)==8
                        Ising[i,j,k,l]=2*s*s
                    end
                end
            end
        end
    end
    return TensorMap(Ising,V⊗V←V⊗V)
end