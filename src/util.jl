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
)


load!(flow::Flow, fname::String; dir="./") = load!(flow.p, flow.u, fname; dir)
function load!(p, u, fname; dir="./")
    obj = jldopen(joinpath(dir,fname))
    f = typeof(p).name.wrapper
    p .= obj["p"] |> f
    u .= obj["u"] |> f
    obj["t"]
end
write!(fname, p, u, t; dir="./") = jldsave(
    joinpath(dir, fname);
    p=Array(p),
    u=Array(u),
    t=t
)
write!(fname, flow::Flow; dir="./") = write!(
    fname,
    flow.p,
    flow.u,
    WaterLily.time(flow);
    dir
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

    Plots.plot!(p, xaxis=:log10, yaxis=:log10, xlims=(10,1e3), ylims=(1e-6,1e-3),
        xlabel=L"\kappa", ylabel=L"E(\kappa)",framestyle=:box, grid=true, minorgrid=true, size=(900,600)
    )
    !isnothing(fig_path) && (savefig(p, fig_path); println("Figure stored in $(fig_path)"))
    return p
end

function ω!(cpu_array, sim)
    a,dt = sim.flow.σ,sim.L/sim.U
    WaterLily.@inside a[I] = WaterLily.ω_mag(I,sim.flow.u)
    copyto!(cpu_array, a[inside(a)]) # copy to CPU
end

function ω_viz(sim; t_end=nothing, dt_viz=0.001, dt_sim=0.5, video=false, isovalue=0.1)
    function viz_step!(sim; dt_viz)
        sim_step!(sim, sim_time(sim)+dt_viz; remeasure=false, verbose=true)
        ω[] = ω!(dat,sim)
    end

    dat = sim.flow.σ[inside(sim.flow.σ)] |> Array; # CPU buffer array
    ω = ω!(dat, sim) |> Observable
    f = Figure(size=(1200,1200), figure_padding=0)
    N = mean(size(sim.flow.σ))-2
    ax = Axis3(f[1, 1]; aspect=:equal, limits=(1,N,1,N,1,N))
    hidedecorations!(ax)

    # colormap = to_colormap(:plasma)
    # colormap[1] = RGBAf(0,0,0,0)
    # volume!(ax, ω,  algorithm = :absorption, absorption=1f0, colormap=colormap)
    volume!(ax, ω, algorithm=:iso, colormap=[:green], isovalue=isovalue)

    if !isnothing(t_end) # time loop for animation
        if video
            GLMakie.record(f, "hit.mp4", 1:round(Int, (t_end-sim_time(sim))/dt_sim); framerate=30, compression=5) do frame
                viz_step!(sim; dt_viz)
            end
        else
            display(f)
            while sim_time(sim) < t_end
                viz_step!(sim; dt_viz)
            end
        end
    end
    # save("hit.png", ax.scene; px_per_unit = 4)
    display(f)
    return f, ax
end

function ω_contour!(σ, u; levels=20, colormap=:viridis)
    WaterLily.@inside σ[I] = WaterLily.ω_mag(I, u)
    dots = WaterLily.inside(σ).indices |> x -> CartesianIndices(x[1:2])
    indz = last(size(σ))÷2
    f = Figure(size=(1200,1200), figure_padding=5); ax = Axis(f[1, 1])
    GLMakie.contourf!(ax, @views σ[dots,indz]'; levels, colormap)
    display(f)
    return f, ax
end
function σ_contour(σ; levels=20, colormap=:viridis)
    f = Figure(size=(1011,1011), figure_padding=5); ax = Axis(f[1, 1], aspect = AxisAspect(1))
    GLMakie.contourf!(ax, @views σ; levels, colormap)
    display(GLMakie.Screen(), f)
    return f, ax
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



# find files in dir according to pattern
using Glob
function rdir(dir, patterns)
    results = String[]
    patterns = [Glob.FilenameMatch("*" * p * "*") for p in patterns]
    for (root, _, files) in walkdir(dir)
        fpaths = joinpath.(root, files)
        length(fpaths) == 0 && continue
        matches = length(patterns) > 0 ? [filter(x -> occursin(p, x), fpaths) for p in patterns] : fpaths
        push!(results, vcat(matches...)...)
    end
    results
end