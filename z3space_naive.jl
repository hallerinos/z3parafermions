using ITensors, ITensorMPS, LinearAlgebra

function ITensors.space(::SiteType"Rotor"; conserve_qns=false, nmax=4)
    if conserve_qns
        range = -Int(nmax):Int(nmax)
        return [QN("N", i) => 1 for i = -range]
    end
    lhd = 2 * nmax + 1
    return lhd
end

function generate_Nmatrix(; nmax=4)
    nvals = -2*nmax/3:2/3:2*nmax/3
    Nmatrix = diagm(nvals)
    return Nmatrix
end

function generate_Nsqmatrix(; nmax=4)
    nvals = -2*nmax/3:2/3:2*nmax/3
    Nsqmatrix = diagm(nvals).^2
    return Nsqmatrix
end

function generate_exp2πNjmatrix(; nmax=4)
    nvals = -2*nmax/3:2/3:2*nmax/3
    exp2πNj = diagm(exp.(2*π.*nvals))
    return exp2πNj
end

function generate_Xmatrix(; nmax=4)
    lhd = 2 * nmax + 1
    nvals = diag(generate_Nmatrix(; nmax=nmax))

    Xmatrix = zeros(lhd, lhd)
    for (id1, n1) in enumerate(nvals), (id2, n2) in enumerate(nvals)
        if n2 ≈ n1 + 2/3
            Xmatrix[id1, id2] = 1
        end
    end
    return Xmatrix
end

function generate_Ymatrix(; nmax=4)
    lhd = 2 * nmax + 1
    nvals = diag(generate_Nmatrix(; nmax=nmax))

    Ymatrix = zeros(lhd, lhd)
    for (id1, n1) in enumerate(nvals), (id2, n2) in enumerate(nvals)
        if n2 ≈ n1 + 2
            Ymatrix[id1, id2] = 1
        end
    end
    return Ymatrix
end

function generate_CosπNmatrix(; nmax=4)
    nvals = diag(generate_Nmatrix(; nmax=nmax))
    CosπNmatrix = diagm(cos.(π.*nvals))
    return CosπNmatrix
end

function generate_Idmatrix(; nmax=4)
    lhd = 2 * nmax + 1
    Idmatrix = Matrix(1.0I, lhd, lhd)
    return Idmatrix
end

### Adding the matrices as local operators to my local Hilberspace ###

_op(::OpName"N", ::SiteType"Rotor"; nmax=4) = generate_Nmatrix(; nmax=nmax)
_op(::OpName"Nsq", ::SiteType"Rotor"; nmax=4) = generate_Nsqmatrix(; nmax=nmax)
_op(::OpName"X", ::SiteType"Rotor"; nmax=4) = generate_Xmatrix(; nmax=nmax)
_op(::OpName"Xdagger", f::SiteType"Rotor"; nmax=4) = transpose(generate_Xmatrix(; nmax=nmax))
_op(::OpName"Y", ::SiteType"Rotor"; nmax=4) = generate_Ymatrix(; nmax=nmax)
_op(::OpName"Ydagger", f::SiteType"Rotor"; nmax=4) = transpose(generate_Ymatrix(; nmax=nmax))
_op(::OpName"CosπN", ::SiteType"Rotor"; nmax=4) = generate_CosπNmatrix(; nmax=nmax)
_op(::OpName"exp2πN", ::SiteType"Rotor"; nmax=4) = generate_exp2πNjmatrix(; nmax=nmax)
_op(::OpName"Id", ::SiteType"Rotor"; nmax=4) = generate_Idmatrix(; nmax=nmax)

function ITensors.op(on::OpName, st::SiteType"Rotor", s::ITensors.Index)
    return itensor(_op(on, st; nmax=((dim(s)-1)÷2)), s', dag(s))
end

ITensors.val(::ValName"0", ::SiteType"Rotor"; nmax=4) = nmax+1
ITensors.state(::StateName"0", ::SiteType"Rotor"; nmax=4) = generate_Idmatrix(; nmax=nmax)[:, nmax+1]

function ITensors.state(sn::StateName, st::SiteType"Rotor", s::ITensors.Index)
    return itensor(state(sn, st; nmax=((dim(s)-1)÷2)), s)
end