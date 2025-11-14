using SparseArrays, LinearAlgebra, JSON, KrylovKit

function make_matrices(; nmax=8, L=4)
    charges = -2/3*nmax:2/3:2/3*nmax

    d = length(charges)

    N = sparse(diagm(charges))
    cpN = sparse(diagm(cos.(π * charges)))
    X = spzeros(Float64, size(N))
    Y = spzeros(Float64, size(N))
    for i in 1:length(charges)-1
        X[i, i+1] = 1
    end
    Y = sparse(X * X * X)

    # check commutators
    if !(X * N - N * X ≈ 2 / 3 .* X)
        println("[X,N] != 2/3 X")
    end
    if !(Y * N - N * Y ≈ 2 .* Y)
        println("[Y,N] != 2/3 Y")
    end

    Xs = []
    Ys = []
    Ns = []
    cpNs = []
    Ntot = spzeros(Float64, (d^L, d^L))
    for j = 1:L
        # @show j
        idL = spdiagm(ones(d^(j - 1)))
        idR = spdiagm(ones(d^(L - j)))
        push!(Xs, sparse(kron(idL, X, idR)))
        push!(Ys, sparse(kron(idL, Y, idR)))
        Nj = sparse(kron(idL, N, idR))
        push!(Ns, Nj)
        push!(cpNs, sparse(kron(idL, cpN, idR)))

        Ntot += Nj
    end

    return Xs, Ys, Ns, cpNs, Ntot
end

function _make_hamiltonian(; nmax=8, L=4, jt=0.0, jJ=1.0, jΔ=0.0, Ec=1.0, Qg=0.0, pbc=true)

    @show jt, jJ, jΔ, Ec, Qg, pbc

    Xs, Ys, Ns, cpNs, Ntot = make_matrices(nmax=nmax, L=L)
    println("Initialized operator matrix elements.")

    hamiltonian = spzeros(eltype(Xs[1]), size(Xs[1]))

    id = spdiagm(ones(size(Xs[1], 1)))
    for j = 1:L
        # @show j
        hamiltonian += -2 * jΔ * cpNs[j]
        hamiltonian += +Ec * (Ns[j] - Qg * id) * (Ns[j] - Qg * id)
    end

    for j = 1:L-1
        # @show j
        hamiltonian += -jt * (Xs[j+1]' * Xs[j] + Xs[j]' * Xs[j+1])
        hamiltonian += -jJ * (Ys[j+1]' * Ys[j] + Ys[j]' * Ys[j+1])
    end
    if pbc == true
        hamiltonian += -jt * (Xs[1]' * Xs[end] + Xs[end]' * Xs[1])
        hamiltonian += -jJ * (Ys[1]' * Ys[end] + Ys[end]' * Ys[1])
    end

    return Xs, Ys, Ns, cpNs, Ntot, hamiltonian
end
make_hamiltonian(; kwargs...) = _make_hamiltonian(; kwargs...)

function _diagonalize(; sectors=[0, 2/3, 4/3, 2], kwargs...)
    Xs, Ys, Ns, cpNs, Ntot, hamiltonian = make_hamiltonian(; kwargs...)
    println("Initialized Hamiltonian matrix.")

    vals = zeros(length(sectors))
    for (s, sector) in enumerate(sectors)
        cs = (abs.(diag(Ntot) .- sector) .< 1e-6)
        dred = length(cs[cs.==1])

        val, vec = eigsolve(hamiltonian[cs, cs], 1, :SR)
        # @show val
        # val, vec = eigsolve(hamiltonian, 2, :SR)
        vals[s] = minimum(val)

        # @show [v'*hamiltonian[cs, cs]*v for v in vec]
        nsqexps = [sum(vec[1]'*n[cs,:]*n[:,cs]*vec[1] for n in Ns)]
        nexps = [vec[1]'*n[cs,cs]*vec[1] for n in Ns]
        # nsqexps = [sum(vec[1]'*n*n*vec[1] for n in Ns)]
        # nexps = [vec[1]'*n*vec[1] for n in Ns]
        @show sum(nexps), nsqexps, vals[s]
    end
    Δ1 = (vals[2] - vals[1])
    Δ2 = (vals[3] - vals[1])
    Δ3 = (vals[4] - vals[1])
    @show Δ1, Δ2, Δ3
    @show vals

    return vec
end
diagonalize(; kwargs...) = _diagonalize(; kwargs...)

function main()
    # load config file
    fn = (ARGS != String[]) ? ARGS[1] : "cfg/default.json"
    cfg = JSON.parsefile(fn)

    # create directories
    if !ispath(cfg["io"]["output_directory"])
        mkpath(cfg["io"]["output_directory"])
    end
    open("$(cfg["io"]["output_directory"])/config.json", "w") do f
        write(f, JSON.json(cfg))
    end

    diagonalize(;
        nmax=cfg["local_hilbert_space"]["nmax"],
        L=cfg["parameters"]["L"],
        Qg=cfg["parameters"]["Qg"],
        jt=cfg["parameters"]["jt"],
        jJ=cfg["parameters"]["jJ"],
        jΔ=cfg["parameters"]["jΔ"],
        pbc=cfg["parameters"]["pbc"]
    )

    return
end

main();