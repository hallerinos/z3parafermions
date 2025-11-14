function generate_Nmatrix(; n_2e=2)
    nmax = 2 * n_2e
    ni = -nmax:2:nmax
    # nf2_3 = 2/3:2:nmax
    nf2_3 = 2/3:2:nmax÷2
    nf2_3 = vcat(-nf2_3, nf2_3)
    # nf4_3 = 4/3:2:nmax
    nf4_3 = 4/3:2:nmax÷2
    nf4_3 = vcat(-nf4_3, nf4_3)
    nvals = vcat(ni, nf2_3, nf4_3)
    Nmatrix = diagm(nvals)
    return Nmatrix
end

function ITensors.space(::SiteType"Rotor"; conserve_qns=false, n_2e=2)
    nvals = Integer.(3/2*diag(generate_Nmatrix(n_2e=n_2e)))
    if conserve_qns
        nvals = Integer.(3/2*diag(generate_Nmatrix(n_2e=n_2e)))
        return [QN("N", i) => 1 for i = nvals]
    end
    lhd = length(nvals)
    return lhd
end

function generate_Nsqmatrix(; n_2e=2)
    nvals = diag(generate_Nmatrix(; n_2e=n_2e))
    Nsqmatrix = diagm(nvals.^2)
    return Nsqmatrix
end

function generate_exp2πNmatrix(; n_2e=2)
    nvals = diag(generate_Nmatrix(; n_2e=n_2e))
    exp2πN = diagm(exp.(1im*2*π.*nvals))
    return exp2πN
end

function generate_Xmatrix(; n_2e=2)
    nvals = diag(generate_Nmatrix(; n_2e=n_2e))
    lhd = length(nvals)

    Xmatrix = zeros(lhd, lhd)
    for (id1, n1) in enumerate(nvals), (id2, n2) in enumerate(nvals)
        if n2 ≈ n1 + 2/3
            Xmatrix[id1, id2] = 1
        end
    end
    return Xmatrix
end

function generate_Ymatrix(; n_2e=2)
    nvals = diag(generate_Nmatrix(; n_2e=n_2e))
    lhd = length(nvals)

    Ymatrix = zeros(lhd, lhd)
    for (id1, n1) in enumerate(nvals), (id2, n2) in enumerate(nvals)
        if n2 ≈ n1 + 2
            Ymatrix[id1, id2] = 1
        end
    end
    return Ymatrix
end

function generate_CosπNmatrix(; n_2e=2)
    nvals = diag(generate_Nmatrix(; n_2e=n_2e))
    CosπNmatrix = diagm(cos.(π.*nvals))
    return CosπNmatrix
end

function generate_Idmatrix(; n_2e=2)
    Idmatrix = Matrix(1.0I, size(generate_Nmatrix(; n_2e=n_2e))...)
    return Idmatrix
end

### Adding the matrices as local operators to my local Hilberspace ###

_op(::OpName"N", ::SiteType"Rotor"; n_2e=2) = generate_Nmatrix(; n_2e=n_2e)
_op(::OpName"Nsq", ::SiteType"Rotor"; n_2e=2) = generate_Nsqmatrix(; n_2e=n_2e)
_op(::OpName"X", ::SiteType"Rotor"; n_2e=2) = generate_Xmatrix(; n_2e=n_2e)
_op(::OpName"Xdagger", f::SiteType"Rotor"; n_2e=2) = transpose(generate_Xmatrix(; n_2e=n_2e))
_op(::OpName"Y", ::SiteType"Rotor"; n_2e=2) = generate_Ymatrix(; n_2e=n_2e)
_op(::OpName"Ydagger", f::SiteType"Rotor"; n_2e=2) = transpose(generate_Ymatrix(; n_2e=n_2e))
_op(::OpName"CosπN", ::SiteType"Rotor"; n_2e=2) = generate_CosπNmatrix(; n_2e=n_2e)
_op(::OpName"exp2πN", ::SiteType"Rotor"; n_2e=2) = generate_exp2πNmatrix(; n_2e=n_2e)
_op(::OpName"Id", ::SiteType"Rotor"; n_2e=2) = generate_Idmatrix(; n_2e=n_2e)

function ITensors.op(on::OpName, st::SiteType"Rotor", s::ITensors.Index)
    return itensor(_op(on, st; n_2e=((dim(s)-1)÷4)), s', dag(s))
end

ITensors.val(::ValName"0", ::SiteType"Rotor"; n_2e=2) = n_2e+1
ITensors.state(::StateName"0", ::SiteType"Rotor"; n_2e=2) = generate_Idmatrix(; n_2e=n_2e)[:, n_2e+1]

function ITensors.state(sn::StateName, st::SiteType"Rotor", s::ITensors.Index)
    return itensor(state(sn, st; n_2e=((dim(s)-1)÷4)), s)
end