using BenchmarkTools
using BenchmarkPlots
using StatsPlots  

LabAD = joinpath(dirname(@__DIR__), "LabAD")
include(joinpath(LabAD, "test", "test.jl"))
include(joinpath(LabAD, "solution", "forward.jl"))
include(joinpath(LabAD, "solution", "reverse_simple.jl"))
include(joinpath(LabAD, "data.jl"))
include(joinpath(LabAD, "models.jl"))
include("reverse_vectorized.jl")


## First order
num_data = 200
num_features = 10
num_hidden = 2

X, y, w = generate_data(num_data, num_features, num_hidden, false)
# Identity activation
L = loss(cross_entropy, relu_softmax, X, y)


#-----------------------FIRST ORDER-----------------------#
res_forward = @benchmark Forward.gradient($L, $w)
res_reverse_simple = @benchmark SimpleReverse.gradient($L, $w)
res_reverse = @benchmark VectReverse.gradient($L, $w)


#-----------------------SECOND ORDER----------------------#
res_forward_scd = @benchmark Forward.hessian($L, $w)
res_reverse_scd = @benchmark VectReverse.hessian($L, $w)

using Statistics

#-----------------------FIRST ORDER-----------------------#
function summary_stats(res)
    (
        time_mean = mean(res).time / 1e6,    # en millisecondes
        allocs   = minimum(res).allocs,
        bytes    = minimum(res).memory / 1024 # en Ko
    )
end

sum_forward        = summary_stats(res_forward)
sum_reverse_simple = summary_stats(res_reverse_simple)
sum_reverse        = summary_stats(res_reverse)

#-----------------------SECOND ORDER----------------------#
function summary_stats_scd(res)
    (
        time_mean = mean(res).time / 1e6,    # en millisecondes
        allocs   = minimum(res).allocs,
        bytes    = minimum(res).memory / (1024 * 1024) # en MiB
    )
end

sum_forward_scd    = summary_stats_scd(res_forward_scd)
sum_reverse_scd    = summary_stats_scd(res_reverse_scd)



using DataFrames, PrettyTables
#-----------------------FIRST ORDER-----------------------#
df = DataFrame(
    Method = ["Forward", "Reverse simple", "Reverse vectorized"],
    Mean_ms = [sum_forward.time_mean,
                    sum_reverse_simple.time_mean,
                    sum_reverse.time_mean],
    Alloc  = [sum_forward.allocs,
                    sum_reverse_simple.allocs,
                    sum_reverse.allocs],
    Mem_Ko   = [sum_forward.bytes,
                    sum_reverse_simple.bytes,
                    sum_reverse.bytes]
)

pretty_table(df; backend = Val(:latex))

function profile_vr(L, w, n=10_000)
    for i in 1:n
        VectReverse.gradient(L, w)
    end
end

@profview profile_vr(L, w)

#-----------------------SECOND ORDER----------------------#
df = DataFrame(
    Method = ["Forward", "Reverse vectorized"],
    Mean_ms = [sum_forward_scd.time_mean,
                    sum_reverse_scd.time_mean],
    Alloc  = [sum_forward_scd.allocs,
                    sum_reverse_scd.allocs],
    Mem_MiB   = [sum_forward_scd.bytes,
                    sum_reverse_scd.bytes]
)

pretty_table(df; backend = Val(:latex))

function profile_vr_scd(L, w, n=10_000)
    for i in 1:n
        VectReverse.hessian(L, w)
    end
end

@profview profile_vr_scd(L, w)