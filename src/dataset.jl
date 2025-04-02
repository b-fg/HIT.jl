using JLD2

struct Dataset{Xn,Yn}
    dir::String
    x_features::NTuple{Xn,String}
    y_targets::NTuple{Yn,String}
    x_files::Vector{String}
    y_files::Vector{String}
    function Dataset(dir; x_features=(), y_targets=())
        mkpath(joinpath(dir, "x")); mkpath(joinpath(dir, "y"))
        x_files = rdir(joinpath(dir, "x"), x_features)
        y_files = rdir(joinpath(dir, "y"), y_targets)
        new{length(x_features), length(y_targets)}(dir, x_features, y_targets, x_files, y_files)
    end
end

load_x(ds::Dataset, i::Integer) = load(ds, ds.x_files[i])
load_y(ds::Dataset, i::Integer) = load(ds, ds.y_files[i])
load(ds::Dataset, i::Integer) = load(ds, ds.x_files[i], ds.y_files[i])
load(ds::Dataset, files...) = map(files) do fname
    obj = jldopen(fname)
    obj[ds.x_features...]
end

function write!(ds::Dataset, x, y; extra="")
    @assert length(ds.x_files) == length(ds.y_files)
    extra != "" && (extra = "_"*extra)
    write_x!(ds, x; extra)
    write_y!(ds, y; extra)
end
function write_x!(ds::Dataset, x; extra="")
    @assert length(x) == length(ds.x_features)
    n = length(ds.x_files)+1
    fname = joinpath(ds.dir, "x", join(ds.x_features,"_") * "_$n" * "$(extra).jld2")
    jldsave(fname;
        (Symbol(ds.x_features[i]) => x[i] for i in eachindex(x))...
    )
    push!(ds.x_files, fname)
end
function write_y!(ds::Dataset, y; extra="")
    @assert length(y) == length(ds.y_targets)
    n = length(ds.y_files)+1
    fname = joinpath(ds.dir, "y", join(ds.y_targets,"_") * "_$n" * "$(extra).jld2")
    jldsave(fname;
        (Symbol(ds.y_targets[i]) => y[i] for i in eachindex(y))...
    )
    push!(ds.y_files, fname)
end