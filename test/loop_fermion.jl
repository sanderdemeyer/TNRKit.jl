using Revise, TNRKit, TensorKit

T_initial = gross_neveu_start(0,10,0)
scheme = LoopTNR(T_initial)
scheme.finalize!(scheme)

