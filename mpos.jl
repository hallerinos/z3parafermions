function RotorHamiltonian(parameters, lattice)
    L = length(lattice)
    os = OpSum()
    Qg, jΔ, jt, jJ, pbc = parameters["Qg"], parameters["jΔ"], parameters["jt"], parameters["jJ"], parameters["pbc"]
    factor1 = -2.0 * Qg
    factor2 = Qg^2
    factor3 = -2.0 * jΔ
    for j in 1:L
        os += ("N", j, "N", j)
        os += (factor1, "N", j)
        os += (factor2, "Id", j)
        os += (factor3, "CosπN", j)
        if j != L
            if jt > 0
                os += (-jt, "X", j, "Xdagger", j + 1)
                os += (-jt, "Xdagger", j, "X", j + 1)
            end
            if jJ > 0
                os += (-jJ, "Y", j, "Ydagger", j + 1)
                os += (-jJ, "Ydagger", j, "Y", j + 1)
            end
        end
    end
    if pbc == true
        if jt > 0
            os += (-jt, "X", L, "Xdagger", 1)
            os += (-jt, "Xdagger", L, "X", 1)
        end
        if jJ > 0
            os += (-jJ, "Y", L, "Ydagger", 1)
            os += (-jJ, "Ydagger", L, "Y", 1)
        end
    end

    return MPO(os, lattice)
end

function Xdagger(lattice, i)
    op = OpSum()
    for j in 1:length(lattice)
        if i == j
            op += "Xdagger", j
        end
    end
    return MPO(op, lattice)
end

function sumXdag(lattice)
    op = OpSum()
    for j in 1:length(lattice)
        op += "Xdagger", j
    end
    return MPO(op, lattice)
end

function parity(lattice, i, j)
    op = OpSum()
    # automated MPO generation by a sum of operator expressions
    vec = [["exp2πN", k] for k=i:j]
    tuple = Tuple(v for v in vcat(vec...))
    op += 1, tuple...
    return MPO(op, lattice)
end

function X(lattice, i)
    op = OpSum()
    for j in 1:length(lattice)
        if i == j
            op += "X", j
        end
    end
    return MPO(op, lattice)
end

function Ntot(lattice)
    op = OpSum()
    for j in 1:length(lattice)
        op += "N", j
    end
    return MPO(op, lattice)
end

function Nsqtot(lattice)
    op = OpSum()
    for j in 1:length(lattice)
        op += "N", j, "N", j
    end
    return MPO(op, lattice)
end

function Y(lattice, i)
    op = OpSum()
    for j in 1:length(lattice)
        if i == j
            op += "X", j
        end
    end
    return MPO(op, lattice)
end

function Ydagger(lattice, i)
    op = OpSum()
    for j in 1:length(lattice)
        if i == j
            op += "Ydagger", j
        end
    end
    return MPO(op, lattice)
end

function sumYdag(lattice)
    op = OpSum()
    for j in 1:length(lattice)
        op += "Ydagger", j
    end
    return MPO(op, lattice)
end