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