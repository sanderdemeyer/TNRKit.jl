mutable struct HOTRG <: TRGScheme
    # data
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap; finalize=finalize!)
        return new(T, finalize)
    end
end

# function step!(scheme::TRG, trunc::TensorKit.TruncationScheme)
#     U, S, V, _ = tsvd(scheme.T, (1, 2), (3, 4); trunc=trunc)
    
#     @plansor begin
#         A[-1 -2; -3] := U[-1 -2; 1]*sqrt(S)[1; -3]
#         B[-1; -2 -3] := sqrt(S)[-1; 1]*V[1; -2 -3]
#     end

#     U, S, V, _ = tsvd(scheme.T, (1, 4), (2, 3); trunc=trunc)

#     # Flip legs to their original domain (to mitigate space mismatch at the end)
#     U = permute(U, (1,), (2, 3))
#     V = permute(V, (1, 2), (3,))

#     @plansor begin
#         C[-1; -2 -3] := U[-1; -2 1]*sqrt(S)[1; -3]
#         D[-1 -2; -3] := sqrt(S)[-1; 1]*V[1 -2; -3]
#     end

#     # @plansor complains here, not sure why
#     @tensor scheme.T[-1 -2; -3 -4] := D[-1 1; 4] * B[-2; 2 1] * C[2; 3 -3] * A[4 3; -4]
#     return scheme
# end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    #HOTRG along horizontal
    @tensor M[-1 -2 -3; -4 -5 -6] := scheme.T[-1 1; -5 -6]*scheme.T[-2 -3; -4 1]

    UL, _, _, ϵ1 = tsvd(M, (1,2), (3,4,5,6); trunc = trunc)

    _, _, UR, ϵ2 = tsvd(M, (1,2,3,6),(5,4); trunc = trunc)

    PL = adjoint(UL)
    PR = UL

    if ϵ2 < ϵ1
        PL = UR
        PR = adjoint(UR)
    end

    @tensor scheme.T[-1 -2; -3 -4] := PL[-1; 1 2]*M[1 2 -2; 3 4 -4]*PR[4 3; -3]

    #HOTRG along vertical
    @tensor M[-1 -2 -3; -4 -5 -6] := scheme.T[-1 -2; 1 -6]*scheme.T[1 -3; -4 -5]

    UD, _, _, ϵ1 = tsvd(M, (2,3), (4,5,6,1); trunc = trunc)

    _,_,UU, ϵ2 = tsvd(M, (6,5),(1,2,3,4); trunc = trunc)

    PD = adjoint(UD)
    PU = UD

    if ϵ2 < ϵ1
        PD = UU
        PU = adjoint(UU)
    end
    @tensor scheme.T[-1 -2; -3 -4] := PD[-2; 1 2]*M[-1 1 2; -3 3 4]*PU[4 3; -4]
    return scheme
end
