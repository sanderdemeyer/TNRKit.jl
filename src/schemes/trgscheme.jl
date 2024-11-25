abstract type TRGScheme end

function step! end
function finalize! end
function criterion end
function run! end

function run!(scheme::TRGScheme, trscheme::TensorKit.TruncationScheme; finalize_beginning=true)

    data = []
    if finalize_beginning
        @info "Finalizing beginning"
        push!(data, scheme.finalize!(scheme))
    end

    steps = 0    
    crit = true
    
    while crit
        @info "Step $steps, data[end]: $(!isempty(data) ? data[end] : "empty")"
        step!(scheme, trscheme)
        push!(data, scheme.finalize!(scheme))
        steps += 1
        crit = scheme.crit(steps, data)
    end
    return data
end