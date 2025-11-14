using ITensors, ITensorMPS, JSON, DataFrames, CSV, HDF5
include("z3space_naive.jl")

let
    filenames = [ARGS[1]]

    df_all = DataFrame()

    # loop over parameters
    for (idfn, fn) in enumerate(sort(filenames))
        println(fn)

        fn_repl = replace(fn, "wf_0.h5" => "config.json")
        cfg = JSON.parsefile(fn_repl)

        fn_repl = replace(fn, "wf_0.h5" => "corr.csv")
        if isfile(fn_repl)
            println("file $fn_repl present, skipping...")
            continue
        end

        file = h5open(fn, "r")
        ψ = read(file, "psi", MPS)
        close(file)

        normalize!(ψ)

        c1 = correlation_matrix(ψ, "Ydagger", "Y")
        c2 = correlation_matrix(ψ, "N", "N")
        c3 = correlation_matrix(ψ, "Xdagger", "X")

        @info "Correlation matrices measured."

        df_all = DataFrame()
        for i1 in eachindex(ψ), i2 in eachindex(ψ)
            df = DataFrame()
            df[!, "i_1"] = [i1]
            df[!, "i_2"] = [i2]
            df[!, "{\\rm Re}\\braket{\\hat Y^\\dag_{i_1} \\hat Y_{i_2}}"] = [real(c1[i1, i2])]
            df[!, "{\\rm Im}\\braket{\\hat Y^\\dag_{i_1} \\hat Y_{i_2}}"] = [imag(c1[i1, i2])]
            df[!, "{\\rm Re}\\braket{\\hat N_{i_1} \\hat N_{i_2}}"] = [real(c2[i1, i2])]
            df[!, "{\\rm Im}\\braket{\\hat N_{i_1} \\hat N_{i_2}}"] = [imag(c2[i1, i2])]
            df[!, "{\\rm Re}\\braket{\\hat X^\\dag_{i_1} \\hat X_{i_2}}"] = [real(c3[i1, i2])]
            df[!, "{\\rm Im}\\braket{\\hat X^\\dag_{i_1} \\hat X_{i_2}}"] = [imag(c3[i1, i2])]

            df_all = vcat(df_all, df; cols=:union)
        end

        # write and exit main observables
        fn_repl = replace(fn, "wf_0.h5" => "corr.csv")
        @info fn_repl
        CSV.write("$fn_repl", df_all)
    end
end