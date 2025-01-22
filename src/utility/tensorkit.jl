function pseudopow(t::DiagonalTensorMap, a::Real; tol=eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (c, b) in blocks(t′)
        @inbounds for I in LinearAlgebra.diagind(b)
            b[I] = b[I] < tol ? b[I] : b[I]^a
        end
    end
    return t′
end

function Base.maximum(T::TensorMap)
    maxi = zeros(scalartype(T), length(blocks(T)))
    for (_, d) in blocks(T)
        push!(maxi, maximum(abs.(d)))
    end
    return maximum(maxi)
end