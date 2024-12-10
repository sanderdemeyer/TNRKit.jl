function cft_data(scheme::TRGScheme; v=1, unitcell=1)
    # make the indices
    indices = [[i, -i, i+1, -(i+unitcell)] for i in 1:unitcell]
    indices[end][3] = 1
    T = ncon(fill(scheme.T, unitcell), indices)

    outinds = Tuple(collect(1:unitcell))
    ininds = Tuple(collect(unitcell+1:2unitcell))

    T = permute(T, outinds, ininds)
    D, _ = eig(T)

    data = zeros(ComplexF64, dim(space(D, 1)))
    
    i = 1
    for (_, b) in blocks(D)
        for I in LinearAlgebra.diagind(b)
            data[i] = b[I]
             i += 1
        end
    end

    data = reverse(sort(real.(filter(x -> real(x) > 0, data))))
    return unitcell*(1/(2π*v)) * log.(data[1]./data)
end

function cft_data(scheme::BTRG; v=1, unitcell=1)
    throw(NotImplementedError("BTRG requires extra care with the environment tensors, this method will be implemented later"))
end