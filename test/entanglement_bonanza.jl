using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

Ising_βc = log(1.0+sqrt(2))/2.0
scheme = Loop_TNR(gross_neveu_start(0,0,0), gross_neveu_start(0,0,0))

entanglement_filtering!(scheme, 100, 1e-10, truncdim(16))
# psi = make_psi(scheme)
# PR_list, PL_list = find_projectors(psi, 100, 1e-15, truncdim(16))



# @tensor psi1[-1 -2; -3 -4] := psi[1][1 2; 3 4]*PL_list[3][-1;1]*PL_list[1][-2;2]*PR_list[2][3; -3]*PR_list[4][4; -4]
# TA = permute(psi1, (4,2),(3,1))
# @tensor psi2[-1 -2; -3 -4] := psi[2][1 2; 3 4]*PL_list[4][-1; 1]*PL_list[2][-2; 2]*PR_list[3][3;-3]*PR_list[1][4;-4]
# TB = permute(psi2, (2,3),(1,4))
# U1 = isometry(space(TA)[1]',space(TA)[1])
# Udg1 = adjoint(U)
# U2 = isometry(space(TB)[2]',space(TB)[2])
# Udg2 = adjoint(U2)
# @tensor TAf[-1 -2; -3 -4] := TA[1 -2; -3 4]*U1[-1;1]*Udg2[4; -4]
# @tensor TBf[-1 -2; -3 -4] := TB[-1 2; 3 -4]*U2[-2; 2]*Udg1[3; -3]