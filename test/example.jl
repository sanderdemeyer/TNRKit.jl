using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

# criterion to determine convergence
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^((-steps)/2))

# stop when converged or after 50 steps, whichever comes first
stopping_criterion = convcrit(1e-20, trg_f)&maxiter(200)

β = 0
A = zeros(4,4,4,4)
A[1,2,2,2] = 1
A[1,3,2,3] = 1
A[1,2,1,1] = 1
A[3,4,3,2] = 1
A[3,3,3,4] = 1
A[4,4,4,2] = 1
A[4,2,1,3] = 1
A[2,3,4,3] = 1
A[2,3,2,4] = 1
A[2,1,1,3] = 1
A[1,2,4,2] = 1;
A = TensorMap(A,ℝ^4⊗ℝ^4 ← ℝ^4⊗ℝ^4);
# initialize the TRG scheme
scheme_trg = TRG(A)

# run the TRG scheme (and normalize and store the norm in the beginning (finalize_beginning=true))
data_trg = run!(scheme_trg, truncdim(16), stopping_criterion; finalize_beginning=false)
# or: data = run!(scheme, truncdim(16)), this will default to maxiter(100)
lnz_triangle_trg = 0
for (i, d) in enumerate(data_trg)
    lnz_triangle_trg += log(d) * 2.0^((-i)/2)        
end
@show lnz_triangle_trg

# initialize the BTRG scheme
scheme_btrg = BTRG(triangle_good(β), -0.5)

# run the BTRG scheme
data_btrg = run!(scheme_btrg, truncdim(16), stopping_criterion; finalize_beginning=false)
lnz_triangle_btrg = 0
for (i, d) in enumerate(data_btrg)
    lnz_triangle_btrg += log(d) * 2.0^(-i)        
end
@show lnz_triangle_btrg
