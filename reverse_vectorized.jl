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

x = [1 2 3; 4 5 6; 7 8 9]
y = [1 2 3]
x .- y

VectNode(x::Number) = VectNode(nothing, VectNode[], x, nothing)
VectNode(x::AbstractArray) = VectNode(nothing, VectNode[], x, nothing)

function hash(v::VectNode, h::UInt)
    return hash(objectid(v), h)
end
==(a::VectNode, b::VectNode) = a === b

Base.length(v::VectNode) = length(v.value)
Base.size(v::VectNode) = size(v.value)

isscalar(v::VectNode) = isa(v.value, Float64)
shape(v::VectNode) = isa(v.value, Float64) ? () : size(v.value)

ones_like(node::VectNode) = isa(node.value, Float64) ? 1.0 : ones(size(node.value))
zeros_like(node::VectNode) = isa(node.value, Float64) ? 0.0 : zeros(size(node.value))

_promote_operand(x::Union{Number,AbstractArray}) = isa(x, Number) ? Float64(x) : convert.(Float64, x)

function extend(A::Union{Matrix, Vector, Number}, sx::Tuple)
    A_val = A isa Vector ? reshape(A, (length(A), 1)) :
             A isa Number ? reshape([A], (1, 1)) : A
    sa = size(A_val)
	if length(sx) != 2
		error("sx must be the shape of a matrix !")
    elseif sa == sx
        return A_val
    elseif sa[1] == 1 && sa[2] == 1
        return repeat(A_val, sx...)
    elseif sa[2] == 1 && length(sx) == 2
        return repeat(A_val, 1, sx[2])
    elseif sa[1] == 1 && length(sx) == 2
        return repeat(A_val, sx[1], 1)
    else
        error("can't extend matrix of size $(sa) to $(sx)")
    end
end

function compress(A::Union{Matrix}, sx::Tuple)
    sa = size(A)
	if sx[1] == sa[1]
		sum(A, dims=2)
	elseif sx[2] == sa[2]
		sum(A, dims=1)
	else error("can't compress matrix $sa to $sx")
	end
end


function compute_diag_prod(A::Matrix{Float64}, B::AbstractMatrix{Float64})
	sa = size(A)
	sb = size(B)

	if sa[2] != sb[1]
		error("can't compute product between $(sa) and $(sb)")
	end

	toreturn = zeros(Float64, sa[1])
	for i in 1:sa[1]
		toreturn[i] = sum(A[i, :] .* B[:, i])
	end

	return toreturn
end


localjac_dict = Dict{Symbol, Function}()

localjac_dict[:+]  = (x,y) -> ( ones_like(x), ones_like(y) )
localjac_dict[:-]  = (x,y) -> begin
	if shape(x) != shape(y) && !isscalar(x) && !isscalar(y)
			return ()
	end
	return ( ones_like(x), -ones_like(y) )
end
localjac_dict[:*]  = (x,y) -> ( y.value, x.value )
localjac_dict[:/]  = (x,y) -> begin
	if shape(x) != shape(y) && !isscalar(x) && !isscalar(y)
		return ()
	end
	return (
	isa(y.value, Float64) ? (1.0 / y.value) : (1.0 ./ y.value),
    isa(x.value, Float64) ? (- x.value / (y.value^2)) : (- x.value ./ (y.value .^ 2))
	)
end
localjac_dict[:exp]  = x -> ( isa(x.value, Float64) ? exp(x.value) : exp.(x.value), )
localjac_dict[:log]  = x -> ( isa(x.value, Float64) ? (1.0 / x.value) : (1.0 ./ x.value), )
localjac_dict[:tanh] = x -> ( isa(x.value, Float64) ? (1.0 - tanh(x.value)^2) : (1 .- tanh.(x.value).^2), )
localjac_dict[:relu] = x -> ( x.value .> 0,  )
localjac_dict[:sum] = x -> (ones_like(x), )
localjac_dict[:maximum] = x -> begin
    mask = zeros(size(x.value))
    idx = argmax(x.value)
    mask[idx] = 1.0
    return (mask,)
end

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
Base.:-(x::VectNode) = 0 .- x

Base.:/(x::VectNode, y::VectNode) = x./y # à modifier si en faisant x / y on veut x * inv(y)
Base.:/(x::Union{AbstractArray, Number}, y::VectNode) = x./y
Base.:/(x::VectNode, y::Union{AbstractArray, Number}) = x./y

Base.isless(x::VectNode, y::Number) = x.value .< y
Base.isless(x::Number, y::VectNode) = x .< y.value
Base.isless(x::VectNode, y::VectNode) = x.value .< y.value

localjac_dict2 = Dict{Symbol, Function}()

localjac_dict2[:*] = (f) -> begin
	x = f.args[1] ; y = f.args[2]
	y_val, x_val, f_der = y.value, x.value, f.derivative
	x_der = f_der * (y_val'); y_der = (x_val') * f_der
	x.derivative .+= x_der; y.derivative .+= y_der 
end

localjac_dict2[:sum] = (f) -> begin
	x = f.args[1] ; sx = shape(x); sf = shape(f)
	if sf[1] == 1
		x.derivative .+= repeat(f.derivative, sx[1], 1)
	else
		x.derivative .+= repeat(f.derivative, 1, sx[2])
	end
end

localjac_dict2[:maximum] = (f) -> begin
	x = f.args[1] ; sx = shape(x); sf = shape(f)
	mask = zeros(Float64, sx)
	if sf[1] == 1
		for j in 1:sx[2]
			col = view(x.value, :, j)
			imax = argmax(col)
			mask[imax, j] = 1.0
		end
		x.derivative .+= mask .* repeat(f.derivative, sx[1], 1)
	else
		for i in 1:sx[1]
			row = view(x.value, i, :)
			jmax = argmax(row)
			mask[i, jmax] = 1.0
		end
		x.derivative .+= mask .* repeat(f.derivative, 1, sx[2])
	end 
end

localjac_dict2[:/] = (f) -> begin
	x = f.args[1] ; z = f.args[2]; 
	dl_dx = f.derivative ./ extend(z.value, shape(x))
	dl_dz = - compute_diag_prod(x.value, f.derivative') ./ (z.value .^2)
	x.derivative .+= dl_dx
	z.derivative .+= dl_dz
end

localjac_dict2[:-] = (f) -> begin
	x = f.args[1]; z = f.args[2]
	x.derivative .+= f.derivative
	z.derivative .+= compress(f.derivative, shape(z))
end

function Base.:*(x::VectNode, y::VectNode)
	if isscalar(x) || isscalar(y) return x.*y end
	return VectNode(Symbol(*), [x, y], x.value * y.value, ())
end

Base.:*(x::VectNode, y::Union{Number,AbstractArray}) = begin
	if isscalar(x) || isa(y, Number) return x.*y end
	y_node = VectNode(y)
	return VectNode(Symbol(*), [x, y_node], x.value * y, ())
end

Base.:*(x::Union{Number, AbstractArray}, y::VectNode) = begin
	if isa(x, Number) || isscalar(y) return x.*y end
	x_node = VectNode(x)
	return VectNode(Symbol(*), [x_node, y], x * y.value, ())
end

function Base.sum(v::VectNode; dims=nothing)
    if isnothing(dims)
        if isscalar(v)
            return v
        else
            return VectNode(:sum, [v], sum(v.value), localjac_dict[:sum](v))
        end
    else
        return VectNode(:sum, [v], sum(v.value, dims=dims), ())
    end
end

function Base.maximum(v::VectNode; dims=nothing)
    if isnothing(dims)
        if isscalar(v)
            return v
        else
            return VectNode(:maximum, [v], maximum(v.value), localjac_dict[:maximum](v))
        end
    else
        return VectNode(:maximum, [v], maximum(v.value, dims=dims), ())
    end
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

function _accumulate_derivative!(args::Vector{VectNode}, f, jac)
    if isa(jac, Tuple) && length(jac) == 0 # needs special endeling as *
		localjac_dict2[f.op](f)
	else
		for (i, arg) in enumerate(args)
			jac = f.localjac[i]
        	_accumulate_derivative!(arg, f.derivative, jac)
		end
	end
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
    _accumulate_derivative!(f.args, f, f.localjac)
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

gradient(f, x) = begin
	g = deepcopy(x)
	gradient!(f, g, x)
end

end


