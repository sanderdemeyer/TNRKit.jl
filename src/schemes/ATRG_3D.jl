mutable struct ATRG_3D <: TRGScheme
    T::TensorMap

    finalize!::Function
    function ATRG_3D(T::TensorMap; finalize=finalize!)
        new(T, finalize)
    end
end

function step!(scheme::ATRG_3D, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(scheme.T, (1, 5, 6), (2, 4, 3); trunc=trunc)
    @tensor A[-1 -2; -3 -4] := U[-1 -3 -4; -2]
    @tensor B[-1 -2; -3 -4] := S[-4; 1] * V[1; -1 -3 -2]
    @tensor C[-1 -2; -3 -4] := U[-1 -3 -4; 1] * S[1; -2]
    @tensor D[-1 -2; -3 -4] := V[-4; -1 -3 -2]

    @tensor M[-1 -2 -3; -4 -5 -6] := B[-2 1; -4 -6] * C[-1 -3; -5 1]

    U1, S1, V1, _ = tsvd(M, (1, 5, 6), (2, 4, 3); trunc=trunc)

    @tensor X[-1 -2; -3 -4] := U1[-1 -3 -4; 1] * sqrt(S1)[1; -2]
    @tensor Y[-1 -2; -3 -4] := sqrt(S1)[-4; 1] * V1[1; -1 -3 -2]

    @tensor AX_tensor[-1 -2 -3; -4 -5 -6] := A[-1 1; -4 -6] * X[-2 -3; -5 1]
    @tensor YD_tensor[-1 -2 -3; -4 -5 -6] := Y[-1 1; -4 -6] * D[-2 -3; -5 1]

    _, R1 = leftorth(YD_tensor, (1, 2, 3, 6), (4, 5))
    R2, _ = rightorth(AX_tensor, (1, 2), (3, 4, 5, 6))

    @tensor temp1[-1; -2] := R1[-1; 1 2] * R2[1 2; -2]
    U2, S2, V2, _ = tsvd(temp1, (1,), (2,); trunc=trunc)
    inv_s = pseudopow(S2, -0.5)
    @tensor Proj_1[-1 -2; -3] := R2[-1 -2; 1] * adjoint(V2)[1; 2] * inv_s[2; -3]
    @tensor Proj_2[-1; -2 -3] := inv_s[-1; 1] * adjoint(U2)[1; 2] * R1[2; -2 -3]

    _, R3 = leftorth(YD_tensor, (3, 4, 5, 6), (1, 2))
    R4, _ = rightorth(AX_tensor, (4, 5), (1, 2, 3, 6))

    @tensor temp2[-1; -2] := R3[-1; 1 2] * R4[1 2; -2]
    U3, S3, V3, _ = tsvd(temp2, (1,), (2,); trunc=trunc)
    inv_s = pseudopow(S3, -0.5)
    @tensor Proj_3[-1 -2; -3] := R4[-1 -2; 1] * adjoint(V3)[1; 2] * inv_s[2; -3]
    @tensor Proj_4[-1; -2 -3] := inv_s[-1; 1] * adjoint(U3)[1; 2] * R3[2; -2 -3]

    @tensor H[-1 -2; -3 -4] := YD_tensor[1 2 -2; 3 4 -4] * Proj_1[3 4; -3] * Proj_3[1 2; -1]
    @tensor G[-1 -2; -3 -4] := AX_tensor[1 2 -2; 3 4 -4] * Proj_2[-1; 1 2] * Proj_4[-3; 3 4]

    @tensor scheme.T[-1 -2 -3; -4 -5 -6] := G[-1 1; -5 -6] * H[-2 -3; -4 1]

    return scheme
end

function finalize!(scheme::ATRG_3D)
    n = norm(@tensor scheme.T[1 2 3; 1 2 3])
    scheme.T /= n

    # turn the tensor by 90 degrees
    scheme.T = permute(scheme.T, ((2, 6, 4), (5, 3, 1)))

    return n
end

ATRG_3D_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(1-steps))