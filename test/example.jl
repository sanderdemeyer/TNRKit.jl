using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit
# criterion to determine convergence
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^((1-steps)))
# bound_trg_blk(steps::Int, data) = abs(log(data[end][1]) * 4.0^((1-steps)))
# bound_trg_env(steps::Int, data) = abs(log(data[end][2]) * 2.0^((1-steps)))
# stop when converged or after 50 steps, whichever comes first
stopping_criterion = convcrit(1e-20, trg_f)&maxiter(50)
#stopping_criterion = convcrit(1e-12, bound_trg_blk)&maxiter(40)&convcrit(1e-12, bound_trg_env)

# initialize the TRG scheme
Ising_βc = log(1.0 + sqrt(2)) / 2.0
scheme_trg = TRG(gross_neveu_start(0,0,0))

# run the TRG scheme (and normalize and store the norm in the beginning (finalize_beginning=true))
data_trg = run!(scheme_trg, truncdim(32), stopping_criterion; finalize_beginning=true)
# or: data = run!(scheme, truncdim(16)), this will default to maxiter(100)
lnz_trg = 0
for (i, d) in enumerate(data_trg)
    lnz_trg += log(d) * 2.0^((1-i))        
end
@show lnz_trg

scheme_atrg = ATRG(gross_neveu_start(0,0,0))

data_atrg = run!(scheme_atrg, truncdim(16), stopping_criterion; finalize_beginning=true)
lnz_atrg = 0
for (i, d) in enumerate(data_atrg)
    lnz_atrg += log(d) * 2.0^(1-i)        
end
@show lnz_atrg

# initialize the BTRG scheme
scheme_robust = HOTRG_robust(classical_ising_symmetric(Ising_βc))

# run the BTRG scheme
data_robust = run!(scheme_robust, truncdim(32), stopping_criterion; finalize_beginning=true)
lnz_robust = 0
for (i, d) in enumerate(data_robust)
    lnz_robust += log(d) * 2.0^(-i)        
end
@show lnz_robust


scheme = Loop_TNR(gross_neveu_start(0,0,0), gross_neveu_start(0,0,0)) 
data_tnr = []
@info "Finalizing beginning"
push!(data_tnr, scheme.finalize!(scheme))

stopping_criterion_tnr = convcrit(1e-20, trg_f)&maxiter(9)

steps = 0
crit = true
while crit
    @info "Step $(steps + 1), data_tnr[end]: $(!isempty(data_tnr) ? data_tnr[end] : "empty")"
    step!(scheme, 16, 100, 1e-20, 50, 1e-5)
    push!(data_tnr, scheme.finalize!(scheme))
    steps += 1
    crit = stopping_criterion_tnr(steps, data_tnr)
end
lnz_tnr = 0
for (i,d) in enumerate(data_tnr)
    lnz_tnr += log(d) * 2.0^(-i)
end
@show lnz_tnr

@tensor opt=true transfer_ten[-1 -2; -3 -4] := scheme.TA[-1 1; 2 5]*scheme.TB[2 3; -3 6]*scheme.TB[-2 5; 4 1]*scheme.TA[4 6; -4 3]
#@tensor opt=true transfer_ten[-1 -2; -3 -4] := scheme_atrg.T[-1 1; 2 5]*scheme_atrg.T[2 3; -3 6]*scheme_atrg.T[-2 5; 4 1]*scheme_atrg.T[4 6; -4 3]
D, V = eig(transfer_ten)
diag = []
for (i,d) in blocks(D)
    push!(diag, d...)
end
diag = sort!(real(diag))
using JLD2
file = jldopen("tnr_potts_q=3.jld2", "w")
file["tnr_scheme"] = scheme
close(file)