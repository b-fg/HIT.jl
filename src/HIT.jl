module HIT
using DelimitedFiles, Interpolations, Distributions, Random, OutMacro, FFTW
import EllipsisNotation: Ellipsis; const dots = Ellipsis()
import WaterLily: dot

include("util.jl")
include("dataset.jl")

export generate_hit, spectrum, cbc_spectrum, plot_spectra!, ω_viz, σ_contour, filter_sharp
export write!, load!, write_x!, write_y!, load, load_x, load_y, Dataset, set_plots_style!, δ1, ⨂, ⨂m

"""
    generate_hit(L,N,M; mem=Array)

Homogeneous isotropic turbulence initial condition.
    `L`: domain size
    `N`: number of grid points per direction
    `M`: number of modes
"""
function generate_hit(L,N,M; cbc_path="data/cbc_spectrum.dat", mem=Array)
    dx = L/N # cell size
    wn1 = 2π/L # smallest wavenumber represented by this spectrum, determined here from cbc spectrum properties
    # Compute random angles in unit sphere
    ν = rand(Uniform(0,1), M)
    ϕm = rand(Uniform(0,2π), M)
    θm = @. acos(2ν-1)
    ψm = rand(Uniform(-π/2,π/2), M)
    # Highest wave number that can be represented on this grid (nyquist limit): 2\pi/2dx
    wnn = π/dx
    # Wavenumber step
    dk = (wnn - wn1) / M
    # Wavenumber at cell centers
    wn = @. wn1 + 0.5 * dk + $(collect(0:M-1)) * dk
    # Wavenumber vector from random angles
    kx = @. sin(θm) * cos(ϕm) * wn
    ky = @. sin(θm) * sin(ϕm) * wn
    kz = @. cos(θm) * wn
    # Create divergence vector
    ktx = @. sin(kx*dx/2)/dx
    kty = @. sin(ky*dx/2)/dx
    ktz = @. sin(kz*dx/2)/dx
    # Enforce Mass Conservation
    ϕm1 = rand(Uniform(0,2π), M)
    ν1 = rand(Uniform(0,1), M)
    θm1 = @. acos(2ν1-1)
    ζx = @. sin(θm1) * cos(ϕm1)
    ζy = @. sin(θm1) * sin(ϕm1)
    ζz = @. cos(θm1)
    sxm = @. ζy * ktz - ζz * kty
    sym = @. -(ζx * ktz - ζz * ktx)
    szm = @. ζx * kty - ζy * ktx
    smag = @. sqrt(sxm * sxm + sym * sym + szm * szm)
    @. sxm = sxm / smag
    @. sym = sym / smag
    @. szm = szm / smag

    # Verify that the wave vector and sigma are perpendicular
    @assert isapprox(sum(dot(ktx, sxm) + dot(kty, sym) + dot(ktz, szm)), 0; atol=eps(Float32)) "wave vector and sigma are not perpendicular"

    # Get CBC spectrum
    k_cbc, E = cbc_spectrum(cbc_path)
    # Generate turbulence at cell centers
    um = @. sqrt(E(wn)*dk)
    u,v,w = zeros(N,N,N), zeros(N,N,N), zeros(N,N,N)
    c = dx/2 .+ collect(0:N-1)*dx # cell centers

    arg, bmx, bmy, bmz = zeros(M), zeros(M), zeros(M), zeros(M)
    for k=1:N, j=1:N, i=1:N
        @. arg = kx * c[i] + ky * c[j] + kz * c[k] - ψm
        @. bmx = 2.0 * um * cos(arg - kx * dx / 2.0)
        @. bmy = 2.0 * um * cos(arg - ky * dx / 2.0)
        @. bmz = 2.0 * um * cos(arg - kz * dx / 2.0)

        u[i,j,k] = dot(bmx, sxm)
        v[i,j,k] = dot(bmy, sym)
        w[i,j,k] = dot(bmz, szm)
    end
    return (u, v, w) .|> mem
end

spectrum(u, L::Number) = spectrum(u, Tuple(L for i in 1:last(size(u))))
function spectrum(u, L::Tuple)
    N,D = size(u)[1:end-1], last(size(u))
    @assert length(L) == D
    dx = L ./ N
    k0_norm = mean(2π./L)
    kmax_norm = mean(2π./(L./(N./2)))
    N_norm = round(Int,mean(N))
    wn = collect(fftfreq(N[d], dx[d]) * N[d]/dx[d] for d in 1:D) # or wave numbers, 0..(N-2)/2,-N/2,...,-1, vcat(0:(N[i]-2)/2,-N[i]/2:-1)
    uk = collect(fft(u[dots,d])/prod(N) for d in 1:D)
    tke = 0.5sum(uk[dots,d].*conj(uk[dots,d]) for d in 1:D) |> real # TKE distributed across n-dimensional wavenumber space
    tke_sum = zeros(1:N_norm) # spherically integrated TKE with M modes of resolution
    for IJK in CartesianIndices(tke)
        rk = sqrt(sum(wn[d][IJK[d]]^2 for d in 1:D)) |> x->round(Int,x)
        tke_sum[rk+1] += tke[IJK]
    end
    k = collect(k0_norm * i for i in 0:N_norm-1)
    return k, tke_sum./k0_norm
end

function filter_sharp(u, wn_c)
    N,D = size(u)[1:end-1], last(size(u))
    wn = collect(fftfreq(N[d]) * N[d] for d in 1:D) # or wave numbers, 0..(N-2)/2,-N/2,...,-1, vcat(0:(N[i]-2)/2,-N[i]/2:-1)
    uk = collect(fft(u[dots,d]) for d in 1:D)

    wn_m = ⨂m(wn...) # wn matrix
    rk = map(x->sqrt(sum(x.^2)), eachrow(wn_m)) # resulting wavenumber
    sharp_filter = map(x -> x > wn_c/2 ? 0.0 : 1.0, rk) # filter mask
    for d in 1:D
        uk[d][:] .*= sharp_filter
    end
    return stack(ifft(uk[d])|>real for d in 1:D)
end

end # module HIT
