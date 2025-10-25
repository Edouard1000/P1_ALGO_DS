module VectReverse
import ..Flatten
import Base: hash, ==

# -------------------- definition -------------------- #
mutable struct VectNode
    op::Union{Nothing, String}
    args::Vector{VectNode}
    value::Union{Float64, Array{Float64}}
    derivative::Union{Float64, Array{Float64}}      
	memory::Union{Nothing, Float64, Tuple{AbstractArray, AbstractArray}, AbstractArray, Vector{Int}}
end

# -------------------- init -------------------- #
function VectNode(op, args, value)
    v = _make_value(value)
    return VectNode(op, args, v, zeros_like(v), nothing)
end
VectNode(x::Number) = VectNode(nothing, VectNode[], x)
VectNode(x::AbstractArray) = VectNode(nothing, VectNode[], x)

# -------------------- static functions -------------------- #
_make_value(x::Union{Number, AbstractArray}) = isa(x, Number) ? Float64(x) : convert.(Float64, x)
ones_like(val::Union{Float64, Array{Float64}} ) = isa(val, Float64) ? 1.0 : ones(size(val))
zeros_like(val::Union{Float64, Array{Float64}} ) = isa(val, Float64) ? 0.0 : zeros(size(val))

function extend(A::Union{Matrix}, sx::Tuple)
    sa = size(A)
    if sa == sx
        return A
	elseif sa[2] == 1 && length(sx) == 2
        return repeat(A, 1, sx[2])
    elseif sa[1] == 1 && length(sx) == 2
        return repeat(A, sx[1], 1)
    else
        error("can't extend matrix of size $(sa) to $(sx)")
    end
end

function compress(A::Matrix, sx::Tuple)
    sa = size(A)
	if sx[1] == sa[1]
		sum(A, dims=2)
	elseif sx[2] == sa[2]
		sum(A, dims=1)
	else error("can't compress matrix $sa to $sx")
	end
end

function compress(A::Float64, sx::Tuple)
    return A
end

function compress(A::Vector, sx::Tuple)
	if isempty(sx)
		return sum(A)
	else
		return A
	end
end

function compute_diag_prod(A::Matrix{Float64}, B::AbstractMatrix{Float64})
	sa = size(A); sb = size(B)
	if sa[2] != sb[1] error("can't compute product between $(sa) and $(sb)") end
	toreturn = zeros(Float64, sa[1])
	for i in 1:sa[1]
		toreturn[i] = sum(A[i, :] .* B[:, i])
	end
	return toreturn
end

# -------------------- class functions -------------------- #
hash(v::VectNode, h::UInt) = hash(objectid(v), h)
==(a::VectNode, b::VectNode) = a === b
Base.length(v::VectNode) = length(v.value)
Base.size(v::VectNode) = size(v.value)
isscalar(v::VectNode) = isa(v.value, Float64)
shape(v::VectNode) = isa(v.value, Float64) ? () : size(v.value)
ones_like(node::VectNode) = isa(node.value, Float64) ? 1.0 : ones(size(node.value))
zeros_like(node::VectNode) = isa(node.value, Float64) ? 0.0 : zeros(size(node.value))

# ............... backward rules ............... #

check_param = Dict{String, Function}()
check_param["*"] = (sx, sy) -> begin
	if length(sx) != 2 || length(sy) > 2 && sx[2] != sy[1] error("multiplication is not implemented for $sx and $sy") end
end
check_param[".*"] = (sx, sy) -> begin
	if sx != () && sy != () && (sx != sy) error("multiplication broadcasting is not implemented for $sx and $sy") end
end
check_param["sum"] = (sx) -> begin
	if length(sx) != 2 error("sum not define for $sx") end
end
check_param["sum_all"] = (sx) -> begin
	return
end
check_param["maximum"] = (sx) -> begin
	if length(sx) != 2 error("maximum along a dim is not implemented for $sx") end
end
check_param["maximum_all"] = (sx) -> begin
	return
end
check_param["./"] = (sx, sy) -> begin
	if sy != () && sx != () && length(sx) != 2 && sx != sy error("division not implemented for $sx and $sz") end
end
check_param[".-"] = (sx, sy) -> begin
	if sx != () && sy != () && length(sx) != 2 && sx != sy error("soustraction not implemented for $sx and $sz") end
end
check_param[".+"] = (sx, sy) -> begin
	if !isscalar(x) && !isscalar(y) && (shape(x) != shape(y)) error("addition broadcasting for different size not implemented yet") end
end
check_param[".exp"] = (sx) -> begin
	return
end
check_param[".log"] = (sx) -> begin
	return
end
check_param[".tanh"] = (sx) -> begin
	return
end
check_param[".relu"] = (sx) -> begin
	return
end
check_param[".^"] = (sx) -> begin
	return
end

local_back_rule = Dict{String, Function}()
local_back_rule["*"] = (f) -> begin
	x = f.args[1] ; y = f.args[2]
	sx, sy = shape(x), shape(y)
	y_val, x_val, f_der = y.value, x.value, f.derivative
	x_der = f_der * (y_val'); y_der = (x_val') * f_der
	x.derivative .+= x_der; y.derivative .+= y_der 
end
local_back_rule[".*"] = (f) -> begin
	x = f.args[1] ; y = f.args[2] ; sx = shape(x) ; sy = shape(y)
	x.derivative = x.derivative .+ f.derivative .* y.value
	y.derivative = y.derivative .+ f.derivative .* x.value
end
local_back_rule["sum"] = (f) -> begin
	x = f.args[1] ; sx = shape(x)
	x.derivative .+= extend(f.derivative, sx)
end
local_back_rule["sum_all"] = (f) -> begin
	x = f.args[1]
	x.derivative = x.derivative .+ f.derivative
end

local_back_rule["maximum"] = (f) -> begin
	x = f.args[1]; sx = shape(x); sf = shape(f)
	if sf[1] == 1
		if isnothing(f.memory)
			idxs = Vector{Int}(undef, sx[2])
			for j in 1:sx[2]
				col = view(x.value, :, j)
				idxs[j] = argmax(col)
			end
			f.memory = idxs
		end
		for j in 1:sx[2]
			x.derivative[f.memory[j], j] += f.derivative[j]
		end
	else
		if isnothing(f.memory)
			idxs = Vector{Int}(undef, sx[1])
			for i in 1:sx[1]
				row = view(x.value, i, :)
				idxs[i] = argmax(row)
			end
			f.memory = idxs
		end
		for i in 1:sx[1]
			x.derivative[i, f.memory[i]] += f.derivative[i]
		end
	end
end
local_back_rule["maximum_all"] = (f) -> begin
	x = f.args[1]
	if isnothing(f.memory) f.memory = argmax(x.value) end
	idx = f.memory
	x.derivative[idx] += f.derivative
end
local_back_rule["./"] = (f) -> begin
	x = f.args[1] ; z = f.args[2]; sx = shape(x); sz = shape(z)
	if shape(x) != shape(z) && !isscalar(x) && !isscalar(z)
		if isnothing(f.memory) f.memory = (extend(z.value, shape(x)), z.value .^ 2) end
		dl_dx = f.derivative ./ f.memory[1]
		dl_dz = - compute_diag_prod(x.value, f.derivative') ./ f.memory[2]
		x.derivative .+= dl_dx
		z.derivative .+= dl_dz
	else
		if isnothing(f.memory) f.memory = x.value ./ (z.value .^2) end
		x.derivative = x.derivative .+ f.derivative ./ z.value
		z.derivative = z.derivative .- f.derivative .* f.memory
	end
end
local_back_rule[".-"] = (f) -> begin
	x = f.args[1]; z = f.args[2]; sx = shape(x); sz = shape(z)
	x.derivative = x.derivative .+ f.derivative
	z.derivative = z.derivative .- compress(f.derivative, shape(z)) # pour le broadcasting à taille différente !
end
local_back_rule[".+"]  = (f) -> begin
	x = f.args[1]; y=f.args[2]
	x.derivative .+= f.derivative 
	y.derivative .+= f.derivative
end
local_back_rule[".exp"]  = (f) -> begin
	x = f.args[1]
	if isnothing(f.memory) f.memory = exp.(x.value) end
	x.derivative = x.derivative .+ f.derivative .* f.memory
end
local_back_rule[".log"]  = (f) -> begin
	x = f.args[1]
	if isnothing(f.memory) f.memory = 1.0 ./ x.value end
	x.derivative = x.derivative .+ f.derivative .* f.memory
end
local_back_rule[".tanh"] = (f) -> begin
	x = f.args[1]
	if isnothing(f.memory) f.memory = 1 .- tanh.(x.value).^2 end
	x.derivative = x.derivative .+ f.derivative .* f.memory
end
local_back_rule[".relu"] = (f) -> begin
	x = f.args[1]
	if isnothing(f.memory) f.memory = x.value .> 0 end
	x.derivative = x.derivative .+ f.derivative .* f.memory
end
local_back_rule[".^"] = (f) -> begin
	x = f.args[1] ; n = f.args[2].value # on suppose que l'exposant ne peut pas etre un parametre
	if isnothing(f.memory) f.memory = n .* x.value .^ (n-1) end
	x.derivative = x.derivative .+ f.derivative .* f.memory
end

# ............... broadcasting ............... #
function Base.broadcasted(op::Function, x::VectNode)
    sym = "." * string(op)
	haskey(check_param, sym) ? check_param[sym](shape(x)) : error("symbole $sym not implemented")
    return VectNode(sym, [x], op.(x.value))
end

function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    sym = "." * string(op)
	haskey(check_param, sym) ? check_param[sym](shape(x), shape(y)) : error("symbole $sym not implemented")
    return VectNode(sym, [x, y], op.(x.value, y.value))
end

function Base.broadcasted(op::Function, x::VectNode, y::Union{Number,AbstractArray})
    sym = "." * string(op)
    y_node = VectNode(y)
	haskey(check_param, sym) ? check_param[sym](shape(x), shape(y_node)) : error("symbole $sym not implemented")
    return VectNode(sym, [x, y_node], op.(x.value, y_node.value))
end

function Base.broadcasted(op::Function, x::Union{Number,AbstractArray}, y::VectNode)
    sym = "." * string(op)
    x_node = VectNode(x)
	haskey(check_param, sym) ? check_param[sym](shape(x_node), shape(y)) : error("symbole $sym not implemented")
    return VectNode(sym, [x_node, y], op.(x_node.value, y.value))
end

function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{n}) where {n}
    val = x.value .^ n
	n_node = VectNode(n)
	haskey(check_param, ".^") ? check_param[".^"](shape(x)) : error("symbole .^ not implemented")
    return VectNode(".^", [x, n_node], val)
end

# ............... translation rules ............... #

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

function Base.:*(x::VectNode, y::VectNode)
	if isscalar(x) || isscalar(y) return x.*y end
	haskey(check_param, "*") ? check_param["*"](shape(x), shape(y)) : error("symbole * not implemented")
	return VectNode("*", [x, y], x.value * y.value)
end

Base.:*(x::VectNode, y::Union{Number,AbstractArray}) = begin
	if isscalar(x) || isa(y, Number) return x.*y end
	y_node = VectNode(y)
	haskey(check_param, "*") ? check_param["*"](shape(x), shape(y_node)) : error("symbole * not implemented")
	return VectNode("*", [x, y_node], x.value * y)
end

Base.:*(x::Union{Number, AbstractArray}, y::VectNode) = begin
	if isa(x, Number) || isscalar(y) return x.*y end
	x_node = VectNode(x)
	haskey(check_param, "*") ? check_param["*"](shape(x_node), shape(y)) : error("symbole * not implemented")
	return VectNode("*", [x_node, y], x * y.value)
end

function Base.sum(v::VectNode; dims=nothing)
    if isnothing(dims)
        if isscalar(v)
            return v
        else
			haskey(check_param, "sum_all") ? check_param["sum_all"](shape(v)) : error("symbole sum_all not implemented")
            return VectNode("sum_all", [v], sum(v.value))
        end
    else
		haskey(check_param, "sum") ? check_param["sum"](shape(v)) : error("symbole sum not implemented")
        return VectNode("sum", [v], sum(v.value, dims=dims))
    end
end

function Base.maximum(v::VectNode; dims=nothing)
    if isnothing(dims)
        if isscalar(v)
            return v
        else
			haskey(check_param, "maximum_all") ? check_param["maximum_all"](shape(v)) : error("symbole maximum_all not implemented")
            return VectNode("maximum_all", [v], maximum(v.value))
        end
    else
		haskey(check_param, "maximum") ? check_param["maximum"](shape(v)) : error("symbole maximum not implemented")
        return VectNode("maximum", [v], maximum(v.value, dims=dims))
    end
end

# ............... compute gradient  ............... #

function topo_sort!(visited::Set{VectNode}, topo::Vector{VectNode}, f::VectNode)
    if !(f in visited)
        push!(visited, f)
        for arg in f.args
            topo_sort!(visited, topo, arg)
        end
        push!(topo, f)
    end
end

function backward!(f::VectNode)
    visited = Set{VectNode}()
    topo = VectNode[]
    topo_sort!(visited, topo, f)
    reverse!(topo)
    if isa(f.derivative, Float64)
        f.derivative = 1.0
    else
        f.derivative .= ones(size(f.value))
    end
    for n in topo
        if !isnothing(n.op)
    		local_back_rule[n.op](n)
		end
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


