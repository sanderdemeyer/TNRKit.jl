mutable struct Boundary_TRG <: TRGScheme
    T::TensorMap
    E1::TensorMap
    E2::TensorMap
    finalize!::Function
    function Boundary_TRG(T::TensorMap, E1::TensorMap, E2::TensorMap; finalize=finalize!)
        new(T, E1, E2, finalize)
    end
end

 
function boundary_subroutine(T::TensorMap, E::TensorMap, trunc::TensorKit.TruncationScheme)
    @tensor temp_bound_north[-1 -2 -3; -4] := E[-1 -2; 1]*E[1 -3; -4]
    @tensor temp_bulk[-1 -2 -3; -4 -5 -6] := T[-1 -2; 1 -6]*T[1 -3; -4 -5]

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
    #@tensor Proj_4[-1 -2;-3] := inv_s[1; -3] * adjoint(V)[2; 1] * R3[-1 -2 ;2]

    #renormalise the tensors along vertical
    @tensor temp_E1[-1 -2; -3] := temp_bound_north[-1 1 2; -3]*Proj_1[-2; 2 1]
    @tensor bulk_env[-1 -2; -3 -4] := temp_bulk[-1 1 2; -3 3 4]*Proj_3[-2; 2 1]*Proj_2[4 3; -4]
    #@tensor bulk_temp[-1 -2; -3 -4] := temp_bulk[-1 1 2; -3 3 4]*Proj_3[-2; 2 1]*Proj_4[4 3; -4]
    
    #contract along the horizontal direction

    @tensor another_north[-1 -2 -3; -4 -5] := temp_E1[-1 1; -5]* bulk_env[-2 -3; -4 1]
    #@tensor another_bulk[-1 -2 -3; -4 -5 -6] := bulk_temp[-1 1; -5 -6]*bulk_temp[-2 -3; -4 1]

    #construct projectors for horizontal contraction
    R1, _ = rightorth(another_north, (1,2), (3,4,5))
    _, R2 = leftorth(another_north, (1,2,3), (4,5))
    # R3, _ = rightorth(another_bulk, (1,2), (3,4,5,6))
    # _, R4 = leftorth(another_bulk, (1,2,3,6), (4,5))

    @tensor R2R1[-1; -2] := R2[-1; 1 2] * R1[2 1; -2]
    #@tensor R4R3[-1; -2] := R4[-1; 1 2] * R3[2 1; -2]

    U, S, V, _ = tsvd(R2R1, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S, -0.5)
    @tensor Proj_5[-1 -2; -3] := R1[-1 -2; 1] * adjoint(V)[1; 2] * inv_s[2; -3]
    @tensor Proj_6[-1; -2 -3] := inv_s[-1; 1] * adjoint(U)[1; 2] * R2[2; -2 -3]

    # U, S, V, _ = tsvd(R4R3, (1,), (2,); trunc = trunc)
    # inv_s = pseudopow(S, -0.5)
    # @tensor Proj_7[-1 -2; -3] := R3[-1 -2; 1] * adjoint(V)[1; 2] * inv_s[2; -3]
    # @tensor Proj_8[-1; -2 -3] := inv_s[-1; 1] * adjoint(U)[1; 2] * R4[2; -2 -3]

    #renormalise the tensors along horizontal
    @tensor new_E[-1 -2; -3] := Proj_6[-1; 2 1]*another_north[1 2 -2; 3 4]*Proj_5[4 3; -3]
    #@tensor new_T[-1 -2; -3 -4] := Proj_8[-1; 2 1]*another_bulk[1 2 -2; 3 4 -4]*Proj_7[4 3; -3]

    return new_E
end

function bulk_subroutine(T::TensorMap, trunc::TensorKit.TruncationScheme)
    
    @tensor vertical_temp[-1 -2 -3; -4 -5 -6] := T[-1 -2; 1 -6]*T[1 -3;-4 -5]
    #construct projectors for vertical contraction
    R1, _ = rightorth(vertical_temp, (2,3), (4,5,6,1))
    _, R2 = leftorth(vertical_temp, (1,2,3,4), (5,6))
    @tensor R2R1[-1; -2] := R2[-1; 1 2] * R1[2 1; -2]
    
    U, S, V, _ = tsvd(R2R1, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S, -0.5)
    @tensor Proj_1[-1 -2; -3] := R1[-1 -2; 1] * adjoint(V)[1; 2] * inv_s[2; -3]
    @tensor Proj_2[-1; -2 -3] := inv_s[-1; 1] * adjoint(U)[1; 2] * R2[2; -2 -3]

    @tensor T[-1 -2; -3 -4] := Proj_2[-2; 2 1]*vertical_temp[-1 1 2; -3 3 4]*Proj_1[4 3; -4]

    #contract along the horizontal direction
    @tensor horizontal_temp[-1 -2 -3; -4 -5 -6] := T[-1 1; -5 -6]*T[-2 -3; -4 1]
    R3, _ = rightorth(horizontal_temp, (1,2), (3,4,5,6))
    _, R4 = leftorth(horizontal_temp, (1,2,3,6), (4,5))
    @tensor R4R3[-1; -2] := R4[-1; 1 2] * R3[2 1; -2]

    U, S, V, _ = tsvd(R4R3, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S, -0.5)
    @tensor Proj_3[-1 -2; -3] := R3[-1 -2; 1] * adjoint(V)[1; 2] * inv_s[2; -3]
    @tensor Proj_4[-1; -2 -3] := inv_s[-1; 1] * adjoint(U)[1; 2] * R4[2; -2 -3]

    @tensor T[-1 -2; -3 -4] := Proj_4[-1; 2 1]*horizontal_temp[1 2 -2; 3 4 -4]*Proj_3[4 3; -3]
    
    return T
end


function step!(scheme::Boundary_TRG, trunc::TensorKit.TruncationScheme)
    # The boundary tensors are all along the horizontal direction
    # Contract along the vertical direction
    scheme.E1 = boundary_subroutine(scheme.T, scheme.E1, trunc)
    println("Part 1 done")
    T_temp_2 = permute(scheme.T, (3,4),(1,2))
    E_temp = permute(scheme.E2, (2,3),(1,))

    E_temp = boundary_subroutine(T_temp_2, E_temp, trunc)
    scheme.E2 = permute(E_temp, (3,),(1,2))
    U = isometry(flip(space(scheme.E2)[1]), space(scheme.E2)[1])
    Udg = adjoint(U)
    U_intobulk = isometry(flip(space(scheme.E2)[2]), space(scheme.E2)[2])
    @tensor scheme.E2[-1; -2 -3] := scheme.E2[1; 2 3]*U[-1; 1]*U_intobulk[2; -2]*Udg[3; -3]
    
    scheme.T = bulk_subroutine(scheme.T, trunc)
    
    return scheme
end

function finalize!(scheme::Boundary_TRG)
    Bulk_norm = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= Bulk_norm

    Env_norm = norm(@tensor scheme.E1[1 2; 1]*scheme.E2[3; 3 2])
    scheme.E1 /= sqrt(Env_norm)
    scheme.E2 /= sqrt(Env_norm)
    return Bulk_norm, sqrt(Env_norm)
end

# example convcrit function
hotrg_robust_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

