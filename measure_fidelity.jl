using ITensors, ITensorMPS, JSON, DataFrames, CSV, HDF5
include("z3space_nofrac.jl")

let
    # load mps from existing .h5 file
    find_str = "*.h5"
    in_path = "jJ_axis/"

    filenames = split(readchomp(`find $in_path -path $find_str`))  # get selected mps files

    jJs = []
    bds = []
    Ls = []
    for fn in filenames
        cfg = JSON.parsefile(replace(fn, "wf_0.h5" => "config.json"))
        push!(jJs, cfg["parameters"]["jJ"])
        push!(bds, cfg["dmrg"]["bond_dimension"][1])
        push!(Ls, cfg["parameters"]["L"])
    end
    jJs_unique = unique(jJs)
    bds_unique = sort(unique(bds))
    Ls_unique = reverse(sort(unique(Ls)))


    df_all = DataFrame()

    # loop over equal bond dimensions and system sizes
    for L in Ls_unique, bd in bds_unique
        # if L == 100
        #     continue
        # end
        # if bd == 16
        #     continue
        # end
        slice = (bds .== bd) .&& (Ls .== L)

        # sort the parameter values
        jJs_slice = jJs[slice]
        fn_slice = filenames[slice]

        # order the slices
        order = sortperm(jJs_slice)
        jJs_slice = jJs_slice[order]
        fn_slice = fn_slice[order]

        # loop over parameters
        for (idfn, fn) in enumerate(fn_slice[1:length(fn_slice)-1])
            file = h5open(fn, "r")
            ψ0 = read(file, "psi", MPS)
            close(file)

            file = h5open(fn_slice[idfn+1], "r")
            ψ1 = read(file, "psi", MPS)
            close(file)

            normalize!(ψ0)
            normalize!(ψ1)

            df = DataFrame()

            @info L, bd, jJs_slice[idfn], abs(inner(ψ0, ψ1))

            df[!, "L"] = [L]
            df[!, "\\chi"] = [bd]
            df[!, "J_J"] = [jJs_slice[idfn]]
            val = inner(ψ0, ψ1)
            df[!, "{\\rm Re}\\braket{\\psi_0 | \\psi_1}"] = [real(val)]
            df[!, "{\\rm Im}\\braket{\\psi_0 | \\psi_1}"] = [imag(val)]

            df_all = vcat(df_all, df; cols=:union)

            # write and exit main observables
            CSV.write("$in_path/state_fidelity.csv", df_all)
        end
    end
end