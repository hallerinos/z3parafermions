import ITensorMPS.AbstractProjMPO
import ITensors.@debug_check
import ITensors.@timeit_debug
import ITensors.@printf
"""
A ProjMPO0 computes and stores the projection of an
MPO into a basis defined by an MPS, leaving a
certain number of site indices of the MPO unprojected.
Which sites are unprojected can be shifted by calling
the `position!` method.

Drawing of the network represented by a ProjMPO0 `P(H)`,
showing the case of `nsite(P)==2` and `position!(P,psi,4)`
for an MPS `psi`:

```
o--o--o--o--o-  -o--o--o--o--o--o <psi|
|  |  |  |  |    |  |  |  |  |  |
o--o--o--o--o----o--o--o--o--o--o H
|  |  |  |  |    |  |  |  |  |  |
o--o--o--o--o-  -o--o--o--o--o--o |psi>
```
"""
mutable struct ProjMPO0 <: AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  H::MPO
  LR::Vector{ITensor}
end
ProjMPO0(H::MPO) = ProjMPO0(0, length(H) + 1, 0, H, Vector{ITensor}(undef, length(H)))
