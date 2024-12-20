using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

# criterion to determine convergence
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^((1-steps)))
# bound_trg_blk(steps::Int, data) = abs(log(data[end][1]) * 4.0^((1-steps)))
# bound_trg_env(steps::Int, data) = abs(log(data[end][2]) * 2.0^((1-steps)))
# stop when converged or after 50 steps, whichever comes first
stopping_criterion = convcrit(1e-20, trg_f)&maxiter(40)
#stopping_criterion = convcrit(1e-12, bound_trg_blk)&maxiter(40)&convcrit(1e-12, bound_trg_env)

# initialize the TRG scheme

scheme_trg = TRG(gross_neveu_start(0,0,0))

# run the TRG scheme (and normalize and store the norm in the beginning (finalize_beginning=true))
data_trg = run!(scheme_trg, truncdim(16), stopping_criterion; finalize_beginning=true)
# or: data = run!(scheme, truncdim(16)), this will default to maxiter(100)
lnz_trg = 0
for (i, d) in enumerate(data_trg)
    lnz_trg += log(d) * 2.0^((1-i))        
end
@show lnz_trg

# initialize the BTRG scheme
scheme_robust = HOTRG_robust(gross_neveu_start(0,0,0))

# run the BTRG scheme
data_robust = run!(scheme_robust, truncdim(16), stopping_criterion; finalize_beginning=true)
lnz_robust = 0
for (i, d) in enumerate(data_robust)
    lnz_robust += log(d) * 2.0^(1-i)        
end
@show lnz_robust
