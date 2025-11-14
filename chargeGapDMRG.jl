using ITensors, ITensorMPS
using HDF5
using LinearAlgebra

function fraction_to_float(arg::String)
    if occursin("/", arg)
        # Split the string at "/"
        parts = split(arg, "/")
        
        # Convert the numerator and denominator to integers
        numerator = parse(Float64, parts[1])
        denominator = parse(Float64, parts[2])
        
        # Perform the division and return the result as a float
        return numerator / denominator
    else
        return parse(Float64, arg)
    end
end

##############################################################################################
##################### Defining the local Hilberspace and operators ###########################
##############################################################################################

### Define local Hilberspace: In case of no specifed quantum number, we return just local_hilbertspace_dimension ###
# If we want conservd quantum numbers we have to return an array of QN typies created by the function QN(nametag, value). 
# The operation QN(...)=>dim fixes the dimension of the subspace with a given quantum number. 

cqns = true
local_hilbertspace_dimension = 13
function ITensors.space(::SiteType"Rotor"; conserve_qns=false)
    if conserve_qns
        range = -Int(round(local_hilbertspace_dimension/2)):Int(round(local_hilbertspace_dimension/2))
        return [QN("N",i)=>1 for i=-range]
    end
    return local_hilbertspace_dimension
end

### Creating the matrices that symbolize the local operators ###

Nmatrix =  diagm(-round(local_hilbertspace_dimension-1)/3:2/3:round(local_hilbertspace_dimension-1)/3)
@show Nmatrix
return

Xmatrix = zeros(local_hilbertspace_dimension, local_hilbertspace_dimension)
for i in 1:(local_hilbertspace_dimension-1)
    Xmatrix[i,i+1] = 1
end

Ymatrix = zeros(local_hilbertspace_dimension, local_hilbertspace_dimension)
for i in 1:(local_hilbertspace_dimension-3)
    Ymatrix[i,i+3] = 1
end

CosπNdiag = []
for i in -round((local_hilbertspace_dimension-1)/2):round((local_hilbertspace_dimension-1)/2)
    if i%3 == 0
        push!(CosπNdiag, 1)
    else
        push!(CosπNdiag, -0.5) 
    end
end
CosπNmatrix = diagm(convert(Array{Float64,1}, CosπNdiag))

Idmatrix = Matrix(1.0I, local_hilbertspace_dimension, local_hilbertspace_dimension)

### Adding the matrices as local operators to my local Hilberspace ###

ITensors.op(::OpName"N", ::SiteType"Rotor") = Nmatrix
ITensors.op(::OpName"X", ::SiteType"Rotor") = Xmatrix
ITensors.op(::OpName"Xdagger", f::SiteType"Rotor") = transpose(Xmatrix)
ITensors.op(::OpName"Y", ::SiteType"Rotor") = Ymatrix
ITensors.op(::OpName"Ydagger", f::SiteType"Rotor") = transpose(Ymatrix)
ITensors.op(::OpName"CosπN", ::SiteType"Rotor") = CosπNmatrix
ITensors.op(::OpName"id", ::SiteType"Rotor") = Idmatrix

### Defining the constructors creating the operators acting on the wave function  ###

function RotorHamiltonian(Qg, jΔ, jt, jJ, lattice)
    L = length(lattice)
    os = OpSum()
    factor1 = -2.0*Qg
    factor2 = Qg^2
    factor3 = -2.0*jΔ
    for j in 1:L
        os += ("N", j, "N", j)
        os += (factor1, "N", j)
        os += (factor2, "id", j)
        os += (factor3, "CosπN", j)
        if j != L
            os += (-jt, "X", j, "Xdagger", j+1)
            os += (-jt, "Xdagger", j, "X", j+1)
            os += (-jJ, "Y", j, "Ydagger", j+1)
            os += (-jJ, "Ydagger", j, "Y", j+1)
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

function X(lattice, i)
    op = OpSum()
    for j in 1:length(lattice)
        if i == j
            op += "X", j
        end
    end
    return MPO(op, lattice)
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

#############################################################################
###################### DMRG #################################################
#############################################################################


function DMRGparamerter()
    return [100, [16], [1E-10]] # nsweeps, maxdim, cutoff
end

function initialState(L, lattice; chargeSector=0)
    state = [7 for i=1:L]

    if chargeSector == 2/3
        state[Int(L/2)] = 8
    elseif chargeSector == 2
        state[Int(L/2)] = 10
    end

    ψInit= MPS(lattice, state)
    return ψInit
end

function DMRG(Hamiltonian, sysprm, lattice; chargeSector=0)

    nsweeps, maxdim, cutoff = DMRGparamerter()
    L, Qg, jΔ, jt, jJ = sysprm

    if cqns
        ψInit = initialState(L, lattice; chargeSector=chargeSector)
    else
        ψInit = randomMPS(lattice)
    end

    E0, ψ0 = dmrg(Hamiltonian, ψInit; nsweeps, maxdim, cutoff, observer=DMRGObserver(;energy_tol=1E-8, minsweeps=10, energy_type=Float64))

    file_name_wavefunction = "Psi0_N=$(chargeSector)_L=$(Int(L))_Qg=$(round(Qg, digits=3))_jDelta=$(round(jΔ, digits=3))_jt=$(round(jt, digits=3))_jJ=$(round(jJ, digits=3)).h5"
    file = h5open(file_name_wavefunction, "w")
    write(file, "psi", ψ0)
    close(file)

    return E0, ψ0
end

function main()
    if ARGS != String[]
        L = parse(Int, ARGS[1])
        Qg = fraction_to_float(ARGS[2])
        jΔ = fraction_to_float(ARGS[3])
        jt = fraction_to_float(ARGS[4])
        jJ = fraction_to_float(ARGS[5])
    else
        L = 10
        Qg = 1
        jΔ = 1
        jt = 1
        jJ = 1
    end

    sysprm = [L, Qg, jΔ, jt, jJ]
    lattice = siteinds("Rotor", L; conserve_qns=cqns)
    H = RotorHamiltonian(Qg, jΔ, jt, jJ, lattice)
    DMRG(H, sysprm, lattice; chargeSector=0)
    #DMRG(H, sysprm, lattice; chargeSector=2/3)
    #DMRG(H, sysprm, lattice; chargeSector=2)
end

main()