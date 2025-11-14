using ITensors, ITensorMPS, JSON, DataFrames, CSV, HDF5
include("z3space_nofrac.jl")

let
    # load mps from existing .h5 file
    find_str = "*.h5"
    in_path = (ARGS != String[]) ? ARGS[1] : "phase_diagram_fine_M64/"

    filenames = split(readchomp(`find $in_path -path $find_str`))  # get selected mps files

    df_all = DataFrame()

    # loop over parameters
    for (idfn, fn) in enumerate(sort(filenames))

        fn_repl = replace(fn, "wf_0.h5" => "config.json")
        cfg = JSON.parsefile(fn_repl)

        fn_repl_spec = replace(fn, "wf_0.h5" => "ES.csv")
        fn_repl_svn = replace(fn, "wf_0.h5" => "SvN.csv")
        if isfile(fn_repl_spec) && isfile(fn_repl_svn)
            println("files $fn_repl_spec and $fn_repl_svn present, skipping...")
            continue
        end

        file = h5open(fn, "r")
        ψ = read(file, "psi", MPS)
        close(file)

        SvNs = []
        data_df = DataFrame()
        for b=2:length(ψ)-1
            orthogonalize!(ψ, b)
            U, S, V = svd(ψ[b], (linkind(ψ, b-1), siteind(ψ,b)))
            SvN = 0.0
            for n=1:dim(S, 1)
                p = S[n,n]^2
                SvN -= p * log(p)
            end
            push!(SvNs, SvN)

            indices = inds(S)
            for (id, block) in enumerate(nzblocks(S))
                idx = indices[1].space[id].first[1].val
                spec = diag(Matrix(S[block]).^2)
                nval = idx
                df = DataFrame(Dict("bond" => b, "charge" => idx, "spectrum" => [spec]))
                append!(data_df, df)
            end
        end
        df = DataFrame(Dict("bond" => 2:length(ψ)-1, "SvN" => SvNs))

        println(fn_repl_spec)
        CSV.write(fn_repl_spec, data_df)
        println(fn_repl_svn)
        CSV.write(fn_repl_svn, df)
    end
end