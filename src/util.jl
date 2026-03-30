using Plots, LaTeXStrings, CategoricalArrays, Printf, ColorSchemes
using GLMakie; GLMakie.activate!(inline=false)
using WaterLily
using Distributions: mean
using JLD2

set_plots_style!(; fontsize=14, linewidth=1) = Plots.default(
    fontfamily = "Computer Modern",
    linewidth = linewidth,
    framestyle = :box,
    grid = true,
    minorgrid = true,
    left_margin = Plots.Measures.Length(:mm, 5),
    right_margin = Plots.Measures.Length(:mm, 5),
    bottom_margin = Plots.Measures.Length(:mm, 5),
    top_margin = Plots.Measures.Length(:mm, 5),
    titlefontsize = fontsize,
    legendfontsize = fontsize,
    tickfontsize = fontsize,
    labelfontsize = fontsize,
    legend = :bottomleft,
)

δ1(i,::Val{N}) where N = CartesianIndex(ntuple(j -> j==i ? 2 : 1, N))
δ1(i,I::CartesianIndex{N}) where N = δ1(i, Val{N}())

function cbc_spectrum(cbc_path="data/cbc_spectrum.dat", cbc_t=1)
    cbc_spectrum = readdlm(cbc_path)
    k_cbc = 100 * cbc_spectrum[:, 1]
    e_cbc = 1e-6 * cbc_spectrum[:, 1+cbc_t]
    E = interpolate((k_cbc,), e_cbc, Gridded(Linear()))
    return k_cbc, E
end

function plot_spectra!(p, L, N, u;
    cbc_path="cbc_spectrum.dat", cbc_t=1, fig_path=nothing, label=L"~%$(N[1])^3")
    cg = cgrad(:lightrainbow, 10, categorical=true)
    if !isnothing(cbc_path)
        label_cbc = any(contains(p.series_list[i][:label], "CBC") for i in 1:length(p.series_list)) ? nothing : "CBC (exp)"
        k_cbc, E = cbc_spectrum(cbc_path, cbc_t)
        Plots.plot!(p, k_cbc, E(k_cbc), label=label_cbc, color=:black)
    end

    k, tke = spectrum(u, L)
    Plots.plot!(p, k[2:end], tke[2:end], label=label, color=cg.colors[length(p.series_list)],
        marker=:circle, markersize=3, markevery=1, markerstrokewidth=0, legendmarkersize=0.1
    )
    Plots.vline!(p, [2π/(L/(N/2))], label=:none, ls=:dash, color=:grey)

    Plots.plot!(p, xaxis=:log10, yaxis=:log10, xlims=(10,2e3), ylims=(1e-8,1e-3),
        xlabel=L"\kappa", ylabel=L"E(\kappa)",framestyle=:box, grid=true, minorgrid=true, size=(900,600)
    )
    !isnothing(fig_path) && (savefig(p, fig_path); println("Figure stored in $(fig_path)"))
    return p
end

"""
Returns the (flattened) Kronecker product between multiple vectors.
Note that `kron(vectors...)` is different, since the Kronecker product orders dimensions inversely than how Julia orders tensors.
https://discourse.julialang.org/t/reshaping-a-matrix-for-reverse-kronecker-product-or-schmidt-decomposition/14702
"""
⨂(vectors...) = [prod.(collect(Base.product(vectors...)))...]
"""
Returns the tensor product of multiple vectors in Matrix form. Multiplications are not aactually performed,
    but the elements of the product are stored in each matrix row.
"""
⨂m(vectors...) = hcat(collect.(vcat(collect(Base.product(vectors...))...))...)'