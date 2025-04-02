using Revise
using HIT, JLD2
using WaterLily: Flow
using WaterLily
import EllipsisNotation: Ellipsis; const dots = Ellipsis()

N = 2^5 # cells per direction
M = 5.08/100 # grid size [m]
L = 9*2π/100 # length of HIT cube [m], L = 11M
velocity_scale = 10 # velocity related to the bulk flow (U₀ in paper) [m/s]

T = Float32 # run with single (Float32) or double (Float64) precision
mem = Array # run on CPU (Array) or GPU (CuArray)
cbc_path = joinpath(@__DIR__, "data", "cbc_spectrum.dat")
sim = Simulation((N,N,N),(0,0,0),velocity_scale)

# t_str = "42.00"
t_str = "98.53"
load!(sim.flow, joinpath(@__DIR__, "data/", "flow_N$(N)_t$(t_str).jld2"))

