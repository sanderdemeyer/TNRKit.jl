using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

scheme = Loop_TNR(triangle_bad(), triangle_bad())

# entanglement_filtering!(scheme, 100, 1e-10, truncdim(16))
psi = make_psi(scheme)
PR_list, PL_list = find_projectors(psi, 100, 1e-10, truncdim(16))