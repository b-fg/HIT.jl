using Revise
using HIT, WaterLily, Printf, LaTeXStrings, Plots, Random, CUDA
using WaterLily: dot, sgs!, size_u, @loop, @inside, inside, inside_u, quick, cds
import WaterLily: CFL
Random.seed!(99) # seed random turbulence generator

smagorinsky(I::CartesianIndex{m} where m; S, Cs, Δ) = @views (Cs*Δ)^2*sqrt(2dot(S[I,:,:],S[I,:,:])) # define the Smagorinsky-Lilly model

# Create the isotropic turbulence box by using `generate_hit` to generate the intial condition. Then we copy it to the velocity field (sim.flow.u)
function hit(L, N, M; load=false, length_scale=1, velocity_scale=1, cbc_path="cbc_spectrum.dat", ν=1.48e-5, mem=Array, T=Float32)
    sim = Simulation((N,N,N), (0,0,0), length_scale; U=velocity_scale, ν, T, mem, perdir=(1,2,3))
    if load
        load!(sim.flow; fname=joinpath(@__DIR__, "data/", "flow_N$(N)_t42.00.jld2"))
    else
        u0 = generate_hit(L,N,M; cbc_path, mem) |> stack
        Ni,D = size_u(sim.flow.u)
        for d in 1:D
            WaterLily.@loop sim.flow.u[I,d] = u0[I-δ1(d,I),d] over I in WaterLily.inside_u(Ni,d)
        end
    end
    WaterLily.perBC!(sim.flow.u, sim.flow.perdir)
    return sim
end

T = Float32 # run with single (Float32) or double (Float64) precision
mem = CuArray # run on CPU (Array) or GPU (CuArray)

# Experiment: Comte-Bellot & Corrsin 1971, https://doi.org/10.1017/S0022112071001599
M = 5.08/100 # grid size [m]
L = 9*2π/100 # length of HIT cube [m], L = 11M
velocity_scale = 10 # velocity related to the bulk flow (U₀ in paper) [m/s]

N = 2^5 # cells per direction
modes = 2^11 # number of modes for initial isotropic turbulence condition, following Saad et al 2016, https://doi.org/10.2514/1.J055230
ν_air = 1.48e-5 # same as Rozema et al 2015, https://doi.org/10.1063/1.4928700 (dry air at 15C)
Re = M*velocity_scale/ν_air # 33866
length_scale = (M/L)*N # for CTU, paper uses M, which here we scale with L/N. The experiment domain is L=11M.
ν_numerical = length_scale/Re # for numerical Re, we use U=1

t0_ctu, t1_ctu, t2_ctu = 42.0, 98.0, 171.0 # in convective time units (CTU), t_ctu=length_scale/velocity_scale = M/U
Cs = T(0.17) # Smagorinsky constant
Δ = sqrt(1^2+1^2+1^2)|>T # Filter width
λ = cds # convective scheme: cds or quick
dt = 0.5 # constant time step (not in CTU!)

# Others
Cs_str = @sprintf("%2.2f", Cs)
udf = Cs > 0 ? sgs! : nothing
cbc_path = joinpath(@__DIR__, "data/", "cbc_spectrum.dat")
set_plots_style!(; fontsize=22, linewidth=2)
WaterLily.CFL(a::Flow;Δt_max=10) = dt # set a constant time step
save, load = false, true

function main()
    println("N=$(N), LES=$(udf), Cs=$(Cs_str), λ=$(λ)")
    sim = hit(L, N, modes; load, length_scale, velocity_scale, cbc_path, ν=ν_numerical, mem, T)
    u_inside = @views sim.flow.u[inside_u(sim.flow.u),:]
    t_str = @sprintf("%2.2f", t0_ctu)
    save && save!(joinpath(@__DIR__,"data/", "flow_N$(N)_t$(t_str).jld2"), sim.flow)
    p = plot_spectra!(Plots.plot(dpi=600, title=L"$N=%$(N)$"), L, N, u_inside|>Array;
        cbc_path, cbc_t=1, label=L"t=%$t_str"
    )

    N_t,n = size_u(sim.flow.u)
    S = zeros(T, N_t..., n, n) |> mem # working array holding a tensor for each cell

    sim_step!(sim, t1_ctu-t0_ctu; verbose=true, remeasure=false, λ, udf, νₜ=smagorinsky, S, Cs, Δ)
    t_str = @sprintf("%2.2f", sim_time(sim)+t0_ctu)
    save && save!(joinpath(@__DIR__,"data/", "flow_N$(N)_t$(t_str).jld2"), sim.flow)
    p = plot_spectra!(p, L, N, u_inside|>Array;
        cbc_path, cbc_t=2, label=L"t=%$t_str"
    )

    sim_step!(sim, sim_time(sim)+(t2_ctu-t1_ctu); verbose=true, remeasure=false, λ, udf, νₜ=smagorinsky, S, Cs, Δ)
    t_str = @sprintf("%2.2f", sim_time(sim)+t0_ctu)
    save && save!(joinpath(@__DIR__,"data/", "flow_N$(N)_t$(t_str).jld2"), sim.flow)
    p = plot_spectra!(p, L, N, u_inside|>Array;
        cbc_path, cbc_t=3, label=L"t=%$t_str",
        fig_path=joinpath(@__DIR__,"plots/", "Ek_N$(N)_modes$(modes)_Cs$(Cs_str)_$(λ)_t$(t_str).png"),
    )

    return sim
end

sim = main(); return

## Visualization
# N_t,n = size_u(sim.flow.u)
# S = zeros(T, N_t..., n, n) |> mem
## 3D
# viz!(sim, ω!; t_end=sim_time(sim)+400, λ, udf, udf_kwargs=Dict(:νₜ=>smagorinsky, :S=>S, :Cs=>Cs, :Δ=>Δ),
#     isovalue=0.14, algorithm=:iso, colormap=[:green],); return
## 2D
# viz!(sim, ω!; t_end=sim_time(sim)+400, d=2,
#     λ, udf, udf_kwargs=Dict(:νₜ=>smagorinsky, :S=>S, :Cs=>Cs, :Δ=>Δ)); return