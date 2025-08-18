"""

"""
function free_energy(data, β; scalefactor = 2.0, initial_size = 2.0)
    lnz = 0.0
    x = 1.0 - log(unitcell) / log(scalefactor)
    for (i, z) in enumerate(data)
        lnz += log(z) * scalefactor^(x - i)
    end
    return -lnz / β
end
