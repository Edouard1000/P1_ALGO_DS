module VectReverse
import ..Flatten
import Base: hash, ==

mutable struct VectNode
    op::Union{Nothing, Symbol}
    args::Vector{VectNode}
    value::Union{Float64, Array{Float64}}             # Float64 pour scalaire, Array pour vect/mat
    localjac::Union{Nothing, Tuple{Vararg{Any}}}      # éléments scalaires ou arrays
    derivative::Union{Float64, Array{Float64}}        # même type que value
end

_make_value(x::Number) = Float64(x)
_make_value(x::AbstractArray) = convert.(Float64, x)

function VectNode(op, args, value, localjac)
    v = _make_value(value)
    if isa(v, Float64)
        der = 0.0
    else
        der = zeros(Float64, size(v))
    end
    return VectNode(op, args, v, localjac, der)
end

VectNode(x::Number) = VectNode(nothing, VectNode[], x, nothing)
VectNode(x::AbstractArray) = VectNode(nothing, VectNode[], x, nothing)

function hash(v::VectNode, h::UInt)
    return hash(objectid(v), h)
end
==(a::VectNode, b::VectNode) = a === b

isscalar(v::VectNode) = isa(v.value, Float64)
shape(v::VectNode) = isa(v.value, Float64) ? () : size(v.value)

ones_like(node::VectNode) = isa(node.value, Float64) ? 1.0 : ones(size(node.value))
zeros_like(node::VectNode) = isa(node.value, Float64) ? 0.0 : zeros(size(node.value))

_promote_operand(x::Union{Number,AbstractArray}) = isa(x, Number) ? Float64(x) : convert.(Float64, x)


localjac_dict = Dict{Symbol, Function}()

localjac_dict[:+]  = (x,y) -> ( ones_like(x), ones_like(y) )
localjac_dict[:-]  = (x,y) -> ( ones_like(x), -ones_like(y) )
localjac_dict[:*]  = (x,y) -> ( y.value, x.value )
localjac_dict[:/]  = (x,y) -> ( isa(y.value, Float64) ? (1.0 / y.value) : (1.0 ./ y.value),
                                 isa(x.value, Float64) ? (- x.value / (y.value^2)) : (- x.value ./ (y.value .^ 2)) )
localjac_dict[:exp]  = x -> ( isa(x.value, Float64) ? exp(x.value) : exp.(x.value), )
localjac_dict[:log]  = x -> ( isa(x.value, Float64) ? (1.0 / x.value) : (1.0 ./ x.value), )
localjac_dict[:tanh] = x -> ( isa(x.value, Float64) ? (1.0 - tanh(x.value)^2) : (1 .- tanh.(x.value).^2), )

function Base.broadcasted(op::Function, x::VectNode)
    sym = Symbol(op)
    haskey(localjac_dict, sym) || error("Opération $sym non supportée (unary).")
    jac = localjac_dict[sym](x)
    val = isa(x.value, Float64) ? op(x.value) : op.(x.value)
    return VectNode(sym, [x], val, jac)
end

function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    sym = Symbol(op)
    haskey(localjac_dict, sym) || error("Opération $sym non supportée (binary).")
    jac = localjac_dict[sym](x, y)
    xv = x.value; yv = y.value
    val = (isa(xv, Float64) && isa(yv, Float64)) ? op(xv, yv) : op.(xv, yv)
    return VectNode(sym, [x, y], val, jac)
end

function Base.broadcasted(op::Function, x::VectNode, y::Union{Number,AbstractArray})
    sym = Symbol(op)
    haskey(localjac_dict, sym) || error("Opération $sym non supportée (x,V).")
    yv = _promote_operand(y) 
    y_node = isa(yv, Float64) ? VectNode(yv) : VectNode(yv)
    jac, _ = localjac_dict[sym](x, y_node)
    xv = x.value
    val = (isa(xv, Float64) && isa(yv, Float64)) ? op(xv, yv) : op.(xv, yv)
    return VectNode(sym, [x], val, (jac, ))
end

function Base.broadcasted(op::Function, x::Union{Number,AbstractArray}, y::VectNode)
    sym = Symbol(op)
    haskey(localjac_dict, sym) || error("Opération $sym non supportée (V,y).")
    xv = _promote_operand(x)
    x_node = isa(xv, Float64) ? VectNode(xv) : VectNode(xv)
    _, jac = localjac_dict[sym](x_node, y)
    yv = y.value
    val = (isa(xv, Float64) && isa(yv, Float64)) ? op(xv, yv) : op.(xv, yv)
    return VectNode(sym, [y], val, (jac, ))
end

function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{n}) where {n}
    val = isa(x.value, Float64) ? x.value^n : x.value .^ n
    jac = ( isa(x.value, Float64) ? (n * x.value^(n-1)) : (n .* x.value .^ (n-1)), )
    return VectNode(:^, [x], val, jac)
end

Base.:+(x::VectNode, y::VectNode) = x.+y
Base.:+(x::Union{AbstractArray, Number}, y::VectNode) = x.+y
Base.:+(x::VectNode, y::Union{AbstractArray, Number}) = x.+y

Base.:-(x::VectNode, y::VectNode) = x.-y
Base.:-(x::Union{AbstractArray, Number}, y::VectNode) = x.-y
Base.:-(x::VectNode, y::Union{AbstractArray, Number}) = x.-y

Base.:/(x::VectNode, y::VectNode) = x./y
Base.:/(x::Union{AbstractArray, Number}, y::VectNode) = x./y
Base.:/(x::VectNode, y::Union{AbstractArray, Number}) = x./y

localjac_dict2 = Dict{Symbol, Function}()

localjac_dict2[:*] = (x, y) -> function (x, y)
	x_isScalar , y_isScalar = isscalar(x), isScalar(y)
	if x_isScalar || y_isScalar
		return localjac_dict[:*](x, y)
	else
		sx, sy = shape(x), shape(y)
		# TODO
	end
end

function Base.:*(x::VectNode, y::VectNode)
	jac = localjac_dict2[:*]
	return VectNode(*, [x, y], x.value * y.value, jac)
end

Base.:*(x::VectNode, y::Union{Number,AbstractArray}) = begin
	jac, _ = localjac_dict2[:*]
	return VectNode(*, [x], x.value * y.value, (jac, ))
end

Base.:*(x::Union{Number, AbstractArray}, y::VectNode) = begin
	_, jac = localjac_dict2[:*]
	return VectNode(*, [y], x.value * y.value, (jac, ))
end

function _accumulate_derivative!(arg::VectNode, df, jac)
    if isa(arg.derivative, Float64)
        if isa(df, Float64) && isa(jac, Float64)
            arg.derivative += df * jac
        else
            error("Shape mismatch: arg.derivative is scalar but product needs array")
        end
    else
        arg.derivative .+= df .* jac
    end
    return nothing
end

function topo_sort!(visited::Set{VectNode}, topo::Vector{VectNode}, f::VectNode)
    if !(f in visited)
        push!(visited, f)
        for arg in f.args
            topo_sort!(visited, topo, arg)
        end
        push!(topo, f)
    end
end

function _backward!(f::VectNode)
    if isnothing(f.op) || isnothing(f.localjac) return end
    for (i, arg) in enumerate(f.args)
        jac = f.localjac[i]
        _accumulate_derivative!(arg, f.derivative, jac)
    end
end

function backward!(f::VectNode)
    visited = Set{VectNode}()
    topo = VectNode[]
    topo_sort!(visited, topo, f)
    reverse!(topo)
    # set output derivative to ones of matching shape
    if isa(f.derivative, Float64)
        f.derivative = 1.0
    else
        f.derivative .= ones(size(f.value))
    end
    for n in topo
        _backward!(n)
    end
    return f
end

function gradient!(f, g::Flatten, x::Flatten)
    x_nodes = Flatten(VectNode.(x.components))
    expr = f(x_nodes)
    backward!(expr)
    for i in eachindex(x.components)
        dnode = x_nodes.components[i]
        if isa(x.components[i], Number)
            g.components[i] = isa(dnode.derivative, Float64) ? dnode.derivative : dnode.derivative[1]
        else
            g.components[i] .= (isa(dnode.derivative, Float64) ? fill(dnode.derivative, size(g.components[i])) : dnode.derivative)
        end
    end
    return g
end

gradient(f, x) = gradient!(f, zero(x), x)

end

using Test
using LinearAlgebra

# --------------------------
# 1) Test scalaire
# --------------------------
function test_scalar()
	x = VectNode(2.0)
	y = VectNode(3.0)
	z = (2.0 .* x) .* y + 5.0
	backward!(z)
	@test z.value == 17.0
	@test x.derivative == 6.0
	@test y.derivative == 4.0
end

# --------------------------
# 2) Test vecteur
# --------------------------
function test_vector()
    x = VectNode([1.0, 2.0, 3.0])
    y = VectNode([2.0, 3.0, 4.0])
    z = x .* y .+ 1.0
    backward!(z)
    @test z.value == [3.0, 7.0, 13.0]
    @test x.derivative == y.value
    @test y.derivative == x.value
end

# --------------------------
# 3) Test matrice
# --------------------------
function test_matrix()
    x = VectNode([1.0 2.0; 3.0 4.0])
    y = VectNode([2.0 0.5; 1.0 3.0])
    z = x .* y .+ 1.0
    backward!(z)
    @test z.value == [3.0 2.0; 4.0 13.0]
    @test x.derivative == y.value
    @test y.derivative == x.value
end

# --------------------------
# 4) Test combinaison scalaire + vecteur
# --------------------------
function test_scalar_vector()
    y = VectNode([1.0, 2.0, 3.0])
    z = 2 .* y .+ 1.0
    backward!(z)
    @test z.value == [3.0, 5.0, 7.0]
    @test y.derivative == fill(2.0, 3) # ∂z/∂y_i = x
end

x_t = rand(1, 3)
w1_t = rand(3, 5)
w2_t = rand(5, 2)
x = VectNode(x_t)
w1 = VectNode(w1_t)
w2 = VectNode(w2_t)
y = x * w1 * w2
y_t = x_t * w1_t * w2_t
y_t
y.value
backward!(y) #do not work for the moment

# --------------------------
# Run all tests
# --------------------------

x = [1, 0, 2]
y = [1, 2, 2]

x*y

test_scalar()
test_vector()
test_matrix()
test_scalar_vector()


