module Gem

export GemBasis, GemHamiltonian, GemModel
export gemisbounded, gemsolve!, gemplot, gemplot!, gemplotr, gemplotr!, gemgetwf

using Plots: plot, plot!
using LinearAlgebra: eigen
using QuadGK: gauss

global gem_xxx, gem_www = gauss(40, 0, 1)
function makegauss(N::Int)
    global gem_xxx, gem_www = gauss(N, 0, 1)
    nothing
end


include("others/quadgauss.jl")

"""
Calculate the integral from zero to positive-infinity.
"""
function integauss1d(f, xxx::T, www::T; xc=1.0) where {T<:Vector{Float64}}
    tmp1 = quadgauss(x -> f(xc * x), xxx, www)
    tmp2 = quadgauss(x -> 1 / x^2 * f(xc / x), xxx, www)
    return xc * (tmp1 + tmp2)
end

# ---------------------------basic functions of Gaussian Expansion Method-------------------------

function double_factorial(n)
    if n == 1 || n == 2
        return n
    end
    return n * double_factorial(n - 2)
end

function N_gem(νn, l)
    return sqrt(2^(l + 2) * (2 * νn)^(l + 3 / 2) / (sqrt(π) * double_factorial(2 * l + 1)))
end

function φ_gem(r, νn, l)
    return N_gem(νn, l) * r^l * exp(-νn * r^2)
end

function NMatrix_gem(νn1, νn2, l)
    return (2 * sqrt(νn1 * νn2) / (νn1 + νn2))^(l + 3 / 2)
end

function TMatrix_gem(νn1, νn2, l, μ, hbar)
    return hbar^2 / μ * (2 * l + 3) * νn1 * νn2 / (νn1 + νn2) * (2 * sqrt(νn1 * νn2) / (νn1 + νn2))^(l + 3 / 2)
end

function VMatrix_gem(V, νn1, νn2, l, xxx::T, www::T) where {T<:Vector{Float64}}
    return N_gem(νn1, l) * N_gem(νn2, l) * integauss1d(r -> (r^(2 * l) * exp(-(νn1 + νn2) * r^2) * V(r) * r^2), xxx, www)
end

#-----------------------------functions related to gaussian basises-----------------------------------
mutable struct GemBasis
    nmax::Int8
    r1::Float64
    rnmax::Float64
end

function gembasisn(gb::GemBasis, n)::Float64
    nmax, r1, rnmax = gb.nmax, gb.r1, gb.rnmax
    return 1 / r1^2 * (r1 / rnmax)^((2n - 2) / (nmax - 1))
end

function gembasisl(gb::GemBasis)::Vector{Float64}
    nmax, r1, rnmax = gb.nmax, gb.r1, gb.rnmax
    return map(n -> 1 / r1^2 * (r1 / rnmax)^((2n - 2) / (nmax - 1)), 1:nmax)
end

#---------------------------functions related to Hamiltonian stuff-------------------------------------
mutable struct GemChannel
    id::Int8
    μ::Float64
    ΔE::Float64
    l::Int8
end

mutable struct GemHamiltonian
    N::Int8
    T::Vector{GemChannel}
    V

    function GemHamiltonian(vμ, vΔE, vl, Vf)
        if !(length(vμ) == length(vΔE) && length(vμ) == length(vl))
            error("The input vectors should be in the same length!")
        end
        len = length(vμ)
        tmp = Vector{GemChannel}(undef, len)
        for i in 1:len
            tmp[i] = GemChannel(i, Float64(vμ[i]), Float64(vΔE[i]), Int8(vl[i]))
        end
        new(len, tmp, Vf)
    end
end

mutable struct GemModel
    Tmat::Matrix{Float64}
    Vmat::Matrix{Float64}
    ΔEmat::Matrix{Float64}
    Nmat::Matrix{Float64}

    evals::Vector{ComplexF64}
    evecs::Matrix{ComplexF64}

    function GemModel(gh::GemHamiltonian, gb::GemBasis; hbar=1)
        nc, nb = gh.N, gb.nmax
        gbb = gembasisl(gb)

        tmp_Tmat = zeros(Float64, nc * nb, nc * nb)
        tmp_ΔEmat = zeros(Float64, nc * nb, nc * nb)
        tmp_Nmat = zeros(Float64, nc * nb, nc * nb)

        # assign static Matrix: T,ΔE,N
        for cl in 1:nc
            tmp_μ, tmp_ΔE, tmp_l = gh.T[cl].μ, gh.T[cl].ΔE, gh.T[cl].l
            for i in 1:nb
                νn_i = gbb[i]
                for j in 1:nb
                    νn_j = gbb[j]
                    tmp_ΔEmat[i+(cl-1)*nb, j+(cl-1)*nb] = tmp_ΔE * NMatrix_gem(νn_i, νn_j, tmp_l)
                    tmp_Nmat[i+(cl-1)*nb, j+(cl-1)*nb] = NMatrix_gem(νn_i, νn_j, tmp_l)
                    tmp_Tmat[i+(cl-1)*nb, j+(cl-1)*nb] = TMatrix_gem(νn_i, νn_j, tmp_l, tmp_μ, hbar)
                end
            end
        end

        new(tmp_Tmat, zeros(Float64, nc * nb, nc * nb), tmp_ΔEmat, tmp_Nmat, zeros(ComplexF64, nc * nb), zeros(ComplexF64, nc * nb, nc * nb))
    end
end



function gemsolve!(gm::GemModel, gh::GemHamiltonian, gb::GemBasis, xxx::T=gem_xxx, www::T=gem_www; kws...) where {T<:Vector{Float64}}
    nc, nb = gh.N, gb.nmax
    gbb = gembasisl(gb)
    for cli in 1:nc
        tmp_l = gh.T[cli].l
        for clj in 1:nc
            for i in 1:nb
                νn_i = gbb[i]
                for j in 1:nb
                    νn_j = gbb[j]
                    gm.Vmat[i+(cli-1)*nb, j+(clj-1)*nb] = VMatrix_gem(r -> gh.V(r, cli, clj; kws...), νn_i, νn_j, tmp_l, xxx, www)
                end
            end
        end
    end
    gm.evals, gm.evecs = eigen(gm.Tmat + gm.ΔEmat + gm.Vmat, gm.Nmat)
    nothing
end

function gemisbounded(gm::GemModel)::Bool
    i, res = 1, false
    while i <= length(gm.evals)
        if isreal(gm.evals[i]) && real(gm.evals[i]) < 0
            return true
        elseif real(gm.evals[i]) >= 0
            break
        end
        i += 1
    end
    return res
end

function gemplot(gm::GemModel, gh::GemHamiltonian, gb::GemBasis, cl, index, rmin, rmax, xxx=gem_xxx, www=gem_www; kwargs...)
    nc, nb = gh.N, gb.nmax
    gbb = gembasisl(gb)

    function wf(r; cl=1, index=1)
        tmp = 0.0
        for i in eachindex(gbb)
            tmp += gm.evecs[i+nb*(cl-1), index] * φ_gem(r, gbb[i], gh.T[cl].l)
        end
        return tmp
    end

    NormalConstant = 0.0
    for i in 1:nc
        NormalConstant += integauss1d(r -> real(r * wf(r; cl=i, index=index))^2, xxx, www)
    end

    sym = sign(real(wf((rmin + rmax) / 2; cl=1, index=index)))

    plot(r -> real(sym * wf(r; cl=cl, index=index) / sqrt(4 * π * NormalConstant)), rmin, rmax; kwargs...)
end

function gemplot!(gm::GemModel, gh::GemHamiltonian, gb::GemBasis, cl, index, rmin, rmax, xxx=gem_xxx, www=gem_www; kwargs...)
    nc, nb = gh.N, gb.nmax
    gbb = gembasisl(gb)

    function wf(r; cl=1, index=1)
        tmp = 0.0
        for i in eachindex(gbb)
            tmp += gm.evecs[i+nb*(cl-1), index] * φ_gem(r, gbb[i], gh.T[cl].l)
        end
        return tmp
    end

    NormalConstant = 0.0
    for i in 1:nc
        NormalConstant += integauss1d(r -> real(r * wf(r; cl=i, index=index))^2, xxx, www)
    end

    sym = sign(real(wf((rmin + rmax) / 2; cl=1, index=index)))

    plot!(r -> real(sym * wf(r; cl=cl, index=index) / sqrt(4 * π * NormalConstant)), rmin, rmax; kwargs...)
end

function gemplotr(gm::GemModel, gh::GemHamiltonian, gb::GemBasis, cl, index, rmin, rmax, xxx=gem_xxx, www=gem_www; kwargs...)
    nc, nb = gh.N, gb.nmax
    gbb = gembasisl(gb)

    function wf(r; cl=1, index=1)
        tmp = 0.0
        for i in eachindex(gbb)
            tmp += gm.evecs[i+nb*(cl-1), index] * φ_gem(r, gbb[i], gh.T[cl].l)
        end
        return tmp
    end

    NormalConstant = 0.0
    for i in 1:nc
        NormalConstant += integauss1d(r -> real(r * wf(r; cl=i, index=index))^2, xxx, www)
    end

    sym = sign(real(wf((rmin + rmax) / 2; cl=1, index=index)))

    plot(r -> real(sym * r * wf(r; cl=cl, index=index) / sqrt(4 * π * NormalConstant)), rmin, rmax; kwargs...)
end

function gemplotr!(gm::GemModel, gh::GemHamiltonian, gb::GemBasis, cl, index, rmin, rmax, xxx=gem_xxx, www=gem_www; kwargs...)
    nc, nb = gh.N, gb.nmax
    gbb = gembasisl(gb)

    function wf(r; cl=1, index=1)
        tmp = 0.0
        for i in eachindex(gbb)
            tmp += gm.evecs[i+nb*(cl-1), index] * φ_gem(r, gbb[i], gh.T[cl].l)
        end
        return tmp
    end

    NormalConstant = 0.0
    for i in 1:nc
        NormalConstant += integauss1d(r -> real(r * wf(r; cl=i, index=index))^2, xxx, www)
    end

    sym = sign(real(wf((rmin + rmax) / 2; cl=1, index=index)))

    plot!(r -> real(sym * r * wf(r; cl=cl, index=index) / sqrt(4 * π * NormalConstant)), rmin, rmax; kwargs...)
end

function gemgetwf(gm::GemModel, gh::GemHamiltonian, gb::GemBasis, index)
    nc, nb = gh.N, gb.nmax
    gbb = gembasisl(gb)

    function wf(r; cl=1, index=1)
        tmp = 0.0
        for i in eachindex(gbb)
            tmp += gm.evecs[i+nb*(cl-1), index] * φ_gem(r, gbb[i], gh.T[cl].l)
        end
        return tmp
    end


    res = Array{Any}(undef, nc)
    for i in 1:nc
        res[i] = (r -> wf(r; cl=i, index=index))
    end
    return res
end


end