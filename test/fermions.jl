f_bench = -1.4515448845652446
T = gross_neveu_start(0, 0, 0)

function rotl90_pf(T)
    return permute(T, ((3,1),(4,2)))
end

function symmetrize(T_flipped)
    T_unflipped = flip(T_flipped, (1,2); inv = true)
    T_c4_unflipped = (T_unflipped + rotl90(T_unflipped) + rotl90(rotl90(T_unflipped)) + rotl90(rotl90(rotl90(T_unflipped)))) / 4
    T_c4_flipped = flip(T_c4_unflipped, (1,2); inv = false)
    T_c4v_flipped = (T_c4_flipped + T_c4_flipped') / 2
    T_c4v_unflipped = flip(T_c4v_flipped, (1,2); inv = true)
    return T_c4v_flipped, T_c4v_unflipped
end

# === TRG ===
@testset "TRG - Gross-Neveu Model" begin
    scheme = TRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-3
end

# === BTRG ===
@testset "BTRG - Gross-Neveu Model" begin
    scheme = BTRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-4
end

# === HOTRG ===
@testset "HOTRG - Gross-Neveu Model" begin
    scheme = HOTRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))
    @test free_energy(data, 1.0; scalefactor = 4.0) ≈ f_bench rtol = 1.0e-3
end

# === ATRG ===
@testset "ATRG - Gross-Neveu Model" begin
    scheme = ATRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))
    @test free_energy(data, 1.0; scalefactor = 4.0) ≈ f_bench rtol = 1.0e-2
end

# === LoopTNR ===
@testset "LoopTNR - Gross-Neveu Model" begin
    scheme = LoopTNR(T)
    data = run!(scheme, truncdim(8), maxiter(10))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-3
end

# === c4CTM ===
@testset "c4CTM - Gross-Neveu Model" begin
    # Use the Gross-Neveu model, but symmetrize the tensor such that the conditions of C4vCTM are satisfied
    T_flipped = gross_neveu_start(0, 0, 0)
    T_c4v_flipped, T_c4v_unflipped = symmetrize(T_flipped)

    # Check symmetries
    @test norm(T_c4v_flipped - T_c4v_flipped') < 1e-14
    @test norm(T_c4v_unflipped - rotl90(T_c4v_unflipped)) < 1e-14

    β = 1.0

    # Calculate the free energy using c4CTM
    data_c4CTM = run!(c4CTM(T_c4v_unflipped), truncdim(8), maxiter(10))
    free_energy_c4CTM = -data_c4CTM / β

    schemes = [TRG, BTRG, LoopTNR] # HOTRG and ATRG do not seem to work
    for scheme = schemes
        data = run!(scheme(T_c4v_flipped), truncdim(8), maxiter(10))
        @test free_energy_c4CTM ≈ free_energy(data, β) rtol = 1.0e-10
    end
end
