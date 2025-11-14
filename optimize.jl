using LinearAlgebra, ITensors

function optimize(Hamiltonian, ψInit, parameters; conserve_qns=true, sweeps_io="./sweeps.csv", wf_io="./tmp.h5")
    nrestarts = parameters["max_nrestarts"]
    cutoff = parameters["cutoff"]
    outputlevel = parameters["outputlevel"]
    χs = parameters["bond_dimension"]
    nsweeps = parameters["max_nsweeps_per_restart"]
    ene_tol = 10^(-1.0*parameters["ndigits_econvergence"])
    krylovdim = parameters["eigsolve_krylovdim"]
    maxiter = parameters["eigsolve_maxiter"]
    write_when_maxdim_exceeds = parameters["write_when_maxdim_exceeds"]

    if conserve_qns == true
        BLAS.set_num_threads(1)  # BLAS multithreading
        ITensors.Strided.disable_threads()  # disable strided dense matrix multithreading
        # ITensors.enable_threaded_blocksparse()
        ITensors.disable_threaded_blocksparse()
    end

    L = length(ψInit)
    ψ0 = copy(ψInit)
    Edmrg1 = Eprev = Eprev2 = Eprev3 = E0 = sigma = -999
    dmrg0_runtime = dmrg1_runtime = dmrg2_runtime = 0
    Esweeps = []
    sigmasweeps = []

    (outputlevel >= 0) ? println("BLAS config: $(BLAS.get_config())") : nothing
    (outputlevel >= 0) ? println("Number of threads: $(BLAS.get_num_threads())") : nothing
    (outputlevel >= 0) ? println("Strided threads: $(ITensors.Strided.get_num_threads())") : nothing
    for r = 1:nrestarts  # how many restarts
        χ = χs[minimum([r, length(χs)])]

        if parameters["engine"] == 0
            dmrg0_runtime += (@elapsed E0, ψ0 = dmrg0(Hamiltonian, ψ0; nsweeps=nsweeps, maxdim=χ, cutoff=cutoff, observer=DMRGObserver(; energy_tol=ene_tol*L, energy_type=Float64), outputlevel=outputlevel, eigsolve_krylovdim=krylovdim, eigsolve_maxiter=maxiter))
        elseif parameters["engine"] == 1
            dmrg1_runtime += (@elapsed E0, ψ0 = dmrg1(Hamiltonian, ψ0; nsweeps=nsweeps, maxdim=χ, cutoff=cutoff, observer=DMRGObserver(; energy_tol=ene_tol*L, energy_type=Float64), outputlevel=outputlevel, eigsolve_krylovdim=krylovdim, eigsolve_maxiter=maxiter, write_when_maxdim_exceeds=write_when_maxdim_exceeds))
        elseif parameters["engine"] == 2
            dmrg2_runtime = (@elapsed E0, ψ0 = dmrg(Hamiltonian, ψInit; nsweeps=nsweeps, maxdim=χ, cutoff=cutoff, observer=DMRGObserver(;energy_tol=ene_tol*L, energy_type=Float64), outputlevel=outputlevel, write_when_maxdim_exceeds=write_when_maxdim_exceeds, eigsolve_krylovdim=krylovdim))
        end

        normalize!(ψ0)

        # energy observables to estimate convergence
        Hψ0 = noprime(apply(Hamiltonian, ψ0, maxdim=χ))
        Hsq = inner(ψ0', Hamiltonian, Hψ0)
        E0 = inner(ψ0', Hamiltonian, ψ0)
        sigma = real(Hsq - E0^2)
        (outputlevel >= 0) ? println("DMRG1 restarting step $(r-1). Energy density = $(real(E0)/length(ψ0)) ± $(real(sigma)/length(ψ0)). Runtime = $dmrg1_runtime") : nothing

        push!(Esweeps, real(E0))
        push!(sigmasweeps, real(sigma))

        # save sweep energies
        df = DataFrame()
        df[!, "\\braket{\\hat H}"] = Esweeps
        df[!, "\\braket{\\left(E-\\hat H\\right)^2}"] = sigmasweeps
        CSV.write("$(sweeps_io)", df)

        # save intermediate MPS
        file = h5open(wf_io, "w")
        write(file, "psi", ψ0)
        close(file)

        # check convergence
        if r >= length(χs) && r > 1
            # println(real.([inner(ψ0', Hamiltonian, noprime(normalize(Hψ0))), E0, Eprev, Eprev2, Eprev3]))
            (conv = abs(Eprev - E0)) < ene_tol*L ? break : nothing
            (conv = abs(Eprev2 - E0)) < ene_tol*L ? break : nothing
            (conv = abs(Eprev3 - E0)) < ene_tol*L ? break : nothing
            # abs(-abs(inner(ψ0', Hamiltonian, normalize(Hψ0))) - real(E0)) > 1e-12 ? break : nothing
        end

        # restart with new initial state
        if r < nrestarts && conserve_qns
            ψ0 = normalize(Hψ0)
        end

        Eprev = E0
        if mod(r, 2) == 1
            Eprev2 = E0
        end
        if mod(r, 2) == 0
            Eprev3 = E0
        end
    end
    Edmrg1 = real(E0)

    # @info "Energy DMRG1/DMRG2: $Edmrg1/$E0"
    @info "Total runtime DMRG0/DMRG1/DMRG2: $dmrg0_runtime/$dmrg1_runtime/$dmrg2_runtime"

    (outputlevel >= 0) ? println("Final energy density = $(E0/length(ψ0))") : nothing

    return Esweeps, sigmasweeps, ψ0
end