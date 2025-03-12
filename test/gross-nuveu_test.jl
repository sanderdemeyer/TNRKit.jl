using Revise, TNRKit, TensorKit

function planar_finalize(scheme::HOTRG)
    n = norm(@planar scheme.T[1 2; 3 4] * τ[3 4; 1 2])
    scheme.T /= n 
    scheme.T = permute(scheme.T, ((2, 4), (1, 3)))
    return n 
end

function planar_finalize_antiperiodic(scheme::HOTRG)
    n = norm(@planar twist(scheme.T,3)[1 2; 3 4] * τ[3 4; 1 2])
    scheme.T /= n 
    scheme.T = permute(scheme.T, ((2, 4), (1, 3)))
    return n 
end


Initial_T = gross_neveu_start(0,0,0)
scheme = TRG(Initial_T)

data = run!(scheme, truncdim(16))
 
lnz = 0
for (i, d) in enumerate(data)
    lnz += log(d) * 2.0^(1 - i)
end
@show lnz

#anti-periodic = 1.4512 (non-planar trg)
#periodic = 1.4512 (non-planar trg)
#non-planar norm = 1.4512 (non-planar trg)

#non-planar norm = 1.3710 (planar trg)(16)
#periodic = 1.3711 (planar trg)(0)
#anti-periodic = 1.3710 (planar trg) (4)

#non-planar norm = 1.4513 (hotrg)(16)
#periodic = 1.4513 (hotrg)(0)
#anti-periodic = 1.4513 (hotrg) (4)