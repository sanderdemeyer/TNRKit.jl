mutable struct Boundary_TRG <: TRGScheme
    T::TensorMap
    E1::TensorMap
    E2::TensorMap
    finalize!::Function
    function Boundary_TRG(T::TensorMap, E1::TensorMap, E2::TensorMap; finalize=finalize!)
        new(T, E1, E2, finalize)
    end
end


function step!(scheme::Boundary_TRG, trunc::TensorKit.TruncationScheme)
    # The boundary tensors are all along the horizontal direction
    # Contract along the vertical direction
    @tensor temp_bound_north[-1 -2 -3; -4] := scheme.E1[-1 -2; 1]*scheme.E1[1 -3; -4]
    @tensor temp_bulk[-1 -2 -3; -4 -5 -6] = scheme.T[-1 -2; 1 -6]*scheme.T[1 -3; -4 -5]

    #construct projectors for vertical contraction
    R1, _ = rightorth(temp_bound_north, (2,3), (4,1))
    _, R2 = leftorth(temp_bulk, (1,2,3,4), (5,6))
    R3, _ = rightorth(temp_bulk, (2,3), (4,5,6,1))

    @tensor R1R2[-1; -2] := R2[-1; 1 2] * R1[2 1; -2]
    @tensor R3R2[-1; -2] := R2[-1; 1 2] * R3[2 1; -2]

    U, S, V, _ = tsvd(R1R2, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S, -0.5)

    
    @tensor Proj_1[-1;-2 -3] := R2[1; -2 -3] * adjoint(U)[2; 1] * inv_s[-1; 2]
    @tensor Proj_2[-1 -2;-3] := inv_s[1; -3] * adjoint(V)[2; 1] * R1[-1 -2 ;2]

    U, S, V, _ = tsvd(R3R2, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S, -0.5)
    @tensor Proj_3[-1;-2 -3] := R2[1; -2 -3] * adjoint(U)[2; 1] * inv_s[-1; 2]
    @tensor Proj_4[-1 -2;-3] := inv_s[1; -3] * adjoint(V)[2; 1] * R3[-1 -2 ;2]

    #renormalise the tensors along vertical
    @tensor temp_E1[-1 -2; -3] := temp_bound_north[-1 1 2; -3]*Proj_1[-2; 2 1]
    @tensor bulk_env[-1 -2; -3 -4] := temp_bulk[-1 1 2; -3 3 4]*Proj_3[-2; 2 1]*Proj_2[4 3; -4]
    @tensor bulk_temp[-1 -2; -3 -4] := temp_bulk[-1 1 2; -3 3 4]*Proj_3[-2; 2 1]*Proj_4[4 3; -4]
    
    #contract along the horizontal direction

    @tensor another_north[-1 -2 -3; -4 -5] := temp_E1[-1 1; -5]* bulk_env[-2 -3; -4 1]
    @tensor another_bulk[-1 -2 -3; -4 -5 -6] := bulk_temp[-1 1; -5 -6]*bulk_temp[-2 -3; -4 1]

    #construct projectors for horizontal contraction
    R1, _ = rightorth(another_north, (1,2), (3,4,5))
    _, R2 = leftorth(another_north, (1,2,3), (4,5))
    R3, _ = rightorth(another_bulk, (1,2), (3,4,5,6))
    _, R4 = leftorth(another_bulk, (1,2,3,6), (4,5))

    @tensor R2R1[-1; -2] := R2[-1; 1 2] * R1[2 1; -2]
    @tensor R4R3[-1; -2] := R4[-1; 1 2] * R3[2 1; -2]

    U, S, V, _ = tsvd(R2R1, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S, -0.5)
    @tensor Proj_5[-1 -2; -3] := R1[-1 -2; 1] * adjoint(V)[1; 2] * inv_s[2; -3]
    @tensor Proj_6[-1; -2 -3] := inv_s[-1; 1] * adjoint(U)[1; 2] * R2[2; -2 -3]

    U, S, V, _ = tsvd(R4R3, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S, -0.5)
    @tensor Proj_7[-1 -2; -3] := R3[-1 -2; 1] * adjoint(V)[1; 2] * inv_s[2; -3]
    @tensor Proj_8[-1; -2 -3] := inv_s[-1; 1] * adjoint(U)[1; 2] * R4[2; -2 -3]

    #renormalise the tensors along horizontal
    @tensor scheme.E1[-1 -2; -3] := Proj_6[-1; 2 1]*another_north[1 2 -2; 3 4]*Proj_5[4 3; -3]
    @tensor scheme.T[-1 -2; -3 -4] := Proj_8[-1; 2 1]*another_bulk[1 2 -2; 3 4 -4]*Proj_7[4 3; -3]
    

    return scheme
end
#TODO: The below function is much more complicated than the one for the infinite lattice models.
function finalize!(scheme::Boundary_TRG)
    n = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= n

    # turn the tensor by 90 degrees
    scheme.T = permute(scheme.T, ((2, 3), (4, 1)))

    return n
end

# example convcrit function
hotrg_robust_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

