Ising_βc = log(1.0 + sqrt(2)) / 2.0
Potts_βc(q) = log(1.0 + sqrt(q))

function classical_ising(β::Number; h=0)
    function σ(i::Int64)
        return 2i - 3
    end

    T_array = [exp(β * (σ(i)σ(j) + σ(j)σ(k) + σ(k)σ(l) + σ(l)σ(i)) + h / 2 * β * (σ(i) + σ(j) + σ(k) + σ(l))) for i in 1:2, j in 1:2, k in 1:2, l in 1:2]

    #T = TensorMap(T_array, ℝ^2⊗ℝ^2←ℝ^2⊗ℝ^2)
    V = Vect[Z2Irrep](0 => 1, 1 => 1)
    T = TensorMap(T_array, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)
    #T = TensorMap(T_array, V⊗V←V⊗V)
    return T
end

function classical_ising_symmetric(β)
    V = Vect[Z2Irrep](0 => 1, 1 => 1)
    Ising = zeros(2, 2, 2, 2)
    c = cosh(β)
    s = sinh(β)
    for i = 1:2
        for j = 1:2
            for k = 1:2
                for l = 1:2
                    if (i + j + k + l) == 4
                        Ising[i, j, k, l] = 2 * c * c
                    elseif (i + j + k + l) == 6
                        Ising[i, j, k, l] = 2 * c * s
                    elseif (i + j + k + l) == 8
                        Ising[i, j, k, l] = 2 * s * s
                    end
                end
            end
        end
    end
    return TensorMap(Ising, V ⊗ V ← V ⊗ V)
end

function classical_Potts(q::Int64, β::Float64)
    V = ℂ^q
    A_potts = TensorMap(zeros, V ⊗ V ← V ⊗ V)

    for i = 1:q
        for j = 1:q
            for k = 1:q
                for l = 1:q
                    E = -(Int(i == j) + Int(j == k) + Int(k == l) + Int(l == i))
                    A_potts[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end
    return A_potts
end

function Plaquette_Potts(q::Int64, β::Float64, J::Float64)
    V = ℝ^q
    A_potts = TensorMap(zeros, V ⊗ V ← V ⊗ V)
    Triple_id = TensorMap(zeros, V ← V ⊗ V)

    for i = 1:q
        for j = 1:q
            for k = 1:q
                Triple_id[i, j, k] = Int(i == j == k)
                for l = 1:q
                    E = -(Int(i == j) + Int(j == k) + Int(k == l) + Int(l == i)) / 4 - J * Int(i == j == k == l)
                    A_potts[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end

    @tensor opt = true T[-1 -2 -3 -4; -5 -6 -7 -8] := A_potts[1 2; 3 4] * Triple_id[1; -1 -8] * Triple_id[2; -2 -3] * Triple_id[3; -4 -5] * Triple_id[4; -6 -7]
    U = isometry(fuse(V, V), V ⊗ V)
    @tensor opt = true T_fused[-1 -2; -3 -4] := T[1 2 3 4; 5 6 7 8] * U[-1; 1 2] * U[-2; 3 4] * U[-3; 5 6] * U[-4; 7 8]
    return T_fused
end
