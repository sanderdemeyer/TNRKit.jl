using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit
using LinearAlgebra
BLAS.set_num_threads(20)

# # Test the Plaquette_Potts function


# T = Plaquette_Potts(5, Potts_βc(5), 0)
# @show @tensor T[1 2; 1 2]
# scheme = Loop_TNR(T, T)
# norm(@plansor opt = true scheme.TA[1 2; 3 4] * scheme.TB[3 5; 1 6] *
#                                  scheme.TB[7 4; 8 2] * scheme.TA[8 6; 7 5])


# entanglement_filtering!(scheme, 10, 1e-15, 5)
# norm(@plansor opt = true scheme.TA[1 2; 3 4] * scheme.TB[3 5; 1 6] *
#                                  scheme.TB[7 4; 8 2] * scheme.TA[8 6; 7 5])
# #check isotropy
# U = isometry(space(scheme.TA)[1],space(scheme.TA)[1]')
# @tensor T1[-1 -2; -3 -4] := scheme.TA[1 -2; -3 2] * adjoint(U)[-1; 1] * U[2; -4]
# @tensor T2[-1 -2; -3 -4] := scheme.TB[-1 1; 2 -4] * adjoint(U)[-2; 1] * U[2; -3]



q = 5
target = 0.134 − 0.021im
function cft_data_new(scheme::Loop_TNR)

    @tensor opt=true T[-1 -2; -3 -4] := scheme.TA[-1 1; 2 5]*scheme.TB[2 3; -3 6]*scheme.TB[-2 5; 4 1]*scheme.TA[4 6; -4 3]

    D, V = eig(T)
    diag = []
    for (i,d) in blocks(D)
        push!(diag, d...)
    end
    data = sort!(diag; by=x -> real(x), rev = true)
    return (1 / (2π)) * log.(data[1] ./ data)
end
function collect_sigma(J□,χ;step = 10)
    T = Plaquette_Potts(5,Potts_βc(5),J□)
    scheme = Loop_TNR(T, T)
    scheme.finalize!(scheme)
    sigma_list = []

    for i=1:step
        step!(scheme, χ, 10, 1e-15, 50, 1e-12)
        xσ = cft_data_new(scheme)[2]
        @show (i,xσ);
        distance_to_target = target-xσ
        @show distance_to_target
        push!(sigma_list,xσ)
    end
    return sigma_list
end

collect_sigma(0,16;step=10)


