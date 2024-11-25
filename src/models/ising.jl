function classical_ising(β::Number; h=0)
    function σ(i::Int64)
        return 2i-3
    end
        
    T_array = [exp(β * (σ(i)σ(j) + σ(j)σ(k) + σ(k)σ(l) + σ(l)σ(i)) + h/2 * β * (σ(i) + σ(j) + σ(k) + σ(l))) for i in 1:2, j in 1:2, k in 1:2, l in 1:2]
    
    T = TensorMap(T_array, ℝ^2⊗ℝ^2←ℝ^2⊗ℝ^2)
        
    return T
end