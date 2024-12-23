using Revise, TensorKit, DataFrames, CSV
includet("../src/TRGKit.jl")
using .TRGKit

# function fermionic_trg_finalize!(scheme::TRGScheme)
#     n = norm(@tensor scheme.T[1 2; 1 2])
#     scheme.T /= n
#     return n
# end

# function fermionic_btrg_finalize!(scheme::TRGScheme)
#     scheme.T = permute(scheme.T, (1, 2), (3, 4))

#     n = norm(@tensor scheme.T[1 2; 3 4] * scheme.S1[3; 1] * scheme.S2[4; 2])
#     scheme.T /= n
#     return n
# end

# function fermionic_hotrg_finalize!(scheme::TRGScheme)
#     n = norm(@tensor scheme.T[1 2; 1 2])
#     scheme.T /= n
#     return n
# end

custom_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(1-steps))
custom_convcrit_hot(steps::Int, data) = abs(log(data[end]) * 2.0^(1-steps))

ms = [0]
χs = [32]

for m in ms
    lnz_trgs = []
    lnz_btrgs = []
    lnz_hotrgs = []
    for χ in χs
        T = Hubbard2D_start(2, 1, 0.0001, 4)
        #trg = TRG(copy(T))
        btrg = BTRG(copy(T))
        #hotrg = HOTRG(copy(T))
        #data_trg = run!(trg, truncdim(χ), convcrit(1e-20, custom_convcrit))
        data_btrg = run!(btrg, truncdim(χ), convcrit(1e-20, custom_convcrit))
        #data_hotrg = run!(hotrg, truncdim(χ), convcrit(1e-20, custom_convcrit_hot))

        # lnz_trg = 0
        # for (i, d) in enumerate(data_trg)
        #     lnz_trg += log(d) * 2.0^(1-i)
        # end
        # @show lnz_trg

        lnz_btrg = 0
        for (i, d) in enumerate(data_btrg)
            lnz_btrg += log(d) * 2.0^(1-i)
        end
        @show lnz_btrg
        # lnz_hotrg = 0
        # for (i, d) in enumerate(data_hotrg)
        #     lnz_hotrg += log(d) * 2.0^(1-i)
        # end
        # @show lnz_hotrg

        #push!(lnz_trgs, lnz_trg)
        #push!(lnz_btrgs, lnz_btrg)
        #push!(lnz_hotrgs, lnz_hotrg)
    end
    #CSV.write("data/fermionic_m$(m).csv", DataFrame(χ=χs, lnz_trg=lnz_trgs, lnz_btrg=lnz_btrgs, lnz_hotrg=lnz_hotrgs))
end