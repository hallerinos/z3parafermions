using ITensors, ITensorMPS
using HDF5
using LinearAlgebra
using JSON
using DataFrames, CSV
# using MKL
include("dmrg1.jl")
include("projmpo1.jl")
include("diskprojmpo1.jl")
include("z3space_naive.jl")
# include("z3space_nofrac.jl")
# include("z3space.jl")
include("mpos.jl")
include("optimize.jl")
include("dmrg0.jl")
include("projmpo0.jl")

let  # lazy way to avoid variables in global scope
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

        # initialize state
        lattice = siteinds("Rotor", cfg["parameters"]["L"]; conserve_qns=cfg["local_hilbert_space"]["conserve_quantum_numbers"], nmax=cfg["local_hilbert_space"]["nmax"])
        # lattice = siteinds("Rotor", cfg["parameters"]["L"]; conserve_qns=cfg["local_hilbert_space"]["conserve_quantum_numbers"], n_2e=cfg["local_hilbert_space"]["n_2e"])
        state = ["0" for i = 1:cfg["parameters"]["L"]]
        if cfg["local_hilbert_space"]["conserve_quantum_numbers"]
            @info "Start from vacuum"
            ψ0 = MPS(lattice, state)
        else
            @info "Start from random state"
            ψ0 = randomMPS(lattice, maximum(cfg["dmrg"]["bond_dimension"]))
        end
        
        if isfile(cfg["dmrg"]["initial_state"])
            @info "Load initial state: $(cfg["dmrg"]["initial_state"])"
            file = h5open(cfg["dmrg"]["initial_state"], "r")
            ψr = read(file, "psi", MPS)
            close(file)
            
            if length(ψr) == length(ψ0)
                ψ0 = ψr
                lattice = siteinds(ψ0)
            else
                @warn "Initial state incompatible with required MPS length. Proceed with vacuum state."
            end
        end

        H = RotorHamiltonian(cfg["parameters"], lattice)

        ctr = 0

        df_all = DataFrame()

        file_name_wavefunction = "$(cfg["io"]["output_directory"])/$(cfg["io"]["final_str"])_$ctr.h5"
        file_name_sweeps_output = "$(cfg["io"]["output_directory"])/sweeps_$ctr.csv"
        for (s, s2) in zip(cfg["parameters"]["iterate_apply_Xdag"], cfg["parameters"]["iterate_apply_Ydag"])  # this may not be needed...

            [ψ0 = normalize(apply(sumXdag(lattice), ψ0, maxdim=minimum([2 * maximum(cfg["dmrg"]["bond_dimension"]), 1024]))) for i = 1:s]
            [ψ0 = normalize(apply(sumYdag(lattice), ψ0, maxdim=minimum([2 * maximum(cfg["dmrg"]["bond_dimension"]), 1024]))) for i = 1:s2]

            E0s, sigmas, ψ0 = optimize(H, ψ0, cfg["dmrg"], conserve_qns=cfg["local_hilbert_space"]["conserve_quantum_numbers"], sweeps_io=file_name_sweeps_output, wf_io=file_name_wavefunction)

            # compute energy spread
            Hψ0 = apply(H, ψ0, maxdim=minimum([2 * maximum(cfg["dmrg"]["bond_dimension"]), 2048]))
            E0 = real(inner(ψ0, Hψ0))
            E0sq = inner(ψ0', H, Hψ0)
            Espread = real(E0sq - E0^2)

            df = DataFrame()
            df[!, "\\braket{\\hat H}"] = [E0]
            df[!, "\\braket{\\left(E-\\hat H\\right)^2}"] = [Espread]
            @info "Energy density / spread: $(round(E0/length(ψ0), sigdigits=6)) / $(round(Espread/length(ψ0), sigdigits=6))"

            # compute particle density and spread
            df_lobs = DataFrame()
            Nexp = expect(ψ0, "N")
            df_lobs[!, "i"] = 1:length(Nexp)
            df_lobs[!, "n_i"] = Nexp
            Ntot = real(sum(Nexp))
            Nsqexp = expect(ψ0, "Nsq")
            df_lobs[!, "\\braket{\\hat n_i^2}"] = Nsqexp
            df_lobs[!, "\\braket{(\\hat n_i-n_i)^2}"] = Nsqexp .- Nexp .^ 2
            CSV.write("$(cfg["io"]["output_directory"])/lobs_$ctr.csv", df_lobs)

            Nspread = sum(Nsqexp - Nexp .^ 2)
            df[!, "\\sum_i\\braket{\\hat n_i}"] = [sum(Nexp)]
            df[!, "\\sum_i\\braket{\\hat n_i^2}"] = [sum(Nsqexp)]
            @info "Total Charge / spread: $(round(Ntot, sigdigits=2)) / $(round(Nspread,sigdigits=6))"

            df_all = vcat(df_all, df; cols=:union)

            file = h5open(file_name_wavefunction, "w")
            write(file, "psi", ψ0)
            close(file)

            ctr += 1
        end
        # write and exit main observables
        CSV.write("$(cfg["io"]["output_directory"])/main_obs.csv", df_all)

        return 0
    end

    main()
end