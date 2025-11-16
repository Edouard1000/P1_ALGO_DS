module VectReverse
import ..Flatten
import Base: hash, ==
using SparseArrays


# -------------------- definition -------------------- #
mutable struct VectNode
    op::Union{Nothing, String}
    args::Vector{VectNode}
    value::Union{Float64, Array{Float64}}
    derivative::Union{Nothing, Float64, Array{Float64}} # dl/df  #Jv(L)
	forward_derivative::Union{Nothing, Float64, Array{Float64}} #df/dx 
    second_order_derivative::Union{Nothing, Float64, Array{Float64}} #Hv(L)
	memory::Union{Nothing, Float64, Tuple{AbstractArray, AbstractArray}, AbstractArray, Vector{Int}}
	need_grad::Bool
	need_hess::Bool
end
# -------------------- init -------------------- #
function VectNode(op, args, value, need_grad = true, need_hess = true, f_grad = nothing)
    v = _make_value(value)
	grad = (need_grad ? zeros_like(v) : nothing)
    second_order_derivative = (need_hess ? zeros_like(v) : nothing)
	f_gradi = isnothing(f_grad) ? nothing : _make_value(f_grad) 
	return VectNode(op, args, v, grad, f_gradi, second_order_derivative, nothing, need_grad, need_hess)
end
VectNode(x::Number, need_grad = true, need_hess = true, f_grad = nothing) = VectNode(nothing, VectNode[], x, need_grad, need_hess, f_grad)
VectNode(x::AbstractArray, need_grad = true, need_hess = true, f_grad = nothing) = VectNode(nothing, VectNode[], x, need_grad, need_hess, f_grad)

# -------------------- static functions -------------------- #
_make_value(x::Union{Number, AbstractArray}) = isa(x, Number) ? Float64(x) : convert.(Float64, x)
ones_like(val::Union{Float64, Array{Float64}} ) = isa(val, Float64) ? 1.0 : ones(size(val))
zeros_like(val::Union{Float64, Array{Float64}} ) = isa(val, Float64) ? 0.0 : zeros(size(val))

function extend(A::Float64, sx::Tuple)
	return isempty(sx) ? A : fill(A, sx)
end

function extend(A::Matrix, sx::Tuple)
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

function extend(A::Vector, sx::Tuple)
    sa = size(A)
    if sa == sx
        return A
    end

	if length(sx) == 2
		if sx[1] == length(A) && sx[2] > 1
			return repeat(reshape(A, :, 1), 1, sx[2])
		elseif sx[2] == length(A) && sx[1] > 1
			return repeat(reshape(A, 1, :), sx[1], 1)
		end
	elseif length(sx) == 1 && sx[1] == length(A)
		return A
	end
	error("can't extend vector of size $(sa) to $(sx)")
end


function compress(A::Matrix, sx::Tuple)
    sa = size(A)
	if isempty(sx)
		return sum(A)
	elseif sx[1] == sa[1]
		return sum(A, dims=2)
	elseif sx[2] == sa[2]
		return sum(A, dims=1)
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

function tensormul(H::Array{Float64,3}, d::Array{Float64,1})
    n, n2, p = size(H)
    @assert n == n2 && length(d) == n "Dimensions mismatch"
    R = zeros(n, p)
    for k = 1:p
        R[:,k] = H[:,:,k] * d
    end
    return R
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
	return
end
check_param["sum_all"] = (sx) -> begin
	return
end
check_param["maximum"] = (sx) -> begin
    return
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
	if sx != () && sy != () && (sx != sy) error("addition broadcasting for different size not implemented yet") end
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
	if x.need_grad x.derivative .+= f_der * (y_val') end
	if y.need_grad y.derivative .+= (x_val') * f_der  end
end
local_back_rule[".*"] = (f) -> begin
	x = f.args[1] ; y = f.args[2] ; sx = shape(x) ; sy = shape(y)
	if x.need_grad x.derivative = x.derivative .+ f.derivative .* y.value end
	if y.need_grad y.derivative = y.derivative .+ f.derivative .* x.value end
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
        if x.need_grad x.derivative .+= f.derivative ./ f.memory[1] end
        if z.need_grad z.derivative .-= compress(x.value .* f.derivative, sz) ./ f.memory[2] end
    else
        if isnothing(f.memory) f.memory = x.value ./ (z.value .^2) end
        if x.need_grad x.derivative = x.derivative .+ f.derivative ./ z.value end
        if z.need_grad z.derivative = z.derivative .- f.derivative .* f.memory end
    end
end
local_back_rule[".-"] = (f) -> begin
	x = f.args[1]; z = f.args[2]; sx = shape(x); sz = shape(z)
	if x.need_grad x.derivative = x.derivative .+ f.derivative end
	if z.need_grad z.derivative = z.derivative .- compress(f.derivative, shape(z)) end# pour le broadcasting à taille différente !
end
local_back_rule[".+"]  = (f) -> begin
	x = f.args[1]; y=f.args[2]
	if x.need_grad x.derivative .+= f.derivative end
	if y.need_grad y.derivative .+= f.derivative end
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

local_forw_rule = Dict{String, Function}()

local_forw_rule["*"] = (f) -> begin
    x = f.args[1]; y = f.args[2]
    f.forward_derivative = x.forward_derivative * y.value + x.value * y.forward_derivative
end

local_forw_rule[".*"] = (f) -> begin
    x = f.args[1]; y = f.args[2]
    f.forward_derivative = x.forward_derivative .* y.value .+ y.forward_derivative .* x.value
end

local_forw_rule["sum"] = (f) -> begin
    x = f.args[1]; sf = shape(f)
    f.forward_derivative = compress(x.forward_derivative, sf)
end
local_forw_rule["sum_all"] = (f) -> begin
    x = f.args[1]
    f.forward_derivative = compress(x.forward_derivative, ())
end
local_forw_rule["maximum"] = (f) -> begin
    x = f.args[1]; sx = shape(x); sf = shape(f)
	f.forward_derivative = zeros_like(f)
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
            f.forward_derivative[1, j] = x.forward_derivative[f.memory[j], j]
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
            f.forward_derivative[i, 1] = x.forward_derivative[i, f.memory[i]]
        end
    end
end

local_forw_rule["maximum_all"] = (f) -> begin
    x = f.args[1]
    if isnothing(f.memory) f.memory = argmax(x.value) end
    f.forward_derivative = x.forward_derivative[f.memory]
end

local_forw_rule["./"] = (f) -> begin
    x = f.args[1]; z = f.args[2]; sx = shape(x); sz = shape(z)
    if shape(x) != shape(z) && !isscalar(x) && !isscalar(z)
        z_val = extend(z.value, sx)
        z_der = extend(z.forward_derivative, sx)
        f.forward_derivative = x.forward_derivative .* (1 ./ z_val) .- (x.value .* (z_der .* (1 ./ (z_val .^ 2))))
    else
        f.forward_derivative = x.forward_derivative .* (1 ./ z.value) .- (x.value .* (z.forward_derivative .* (1 ./ (z.value .^ 2))))
    end
end

local_forw_rule[".-"] = (f) -> begin
    x = f.args[1]; z = f.args[2]; sx = shape(x)
    f.forward_derivative = x.forward_derivative .- extend(z.forward_derivative, sx)
end

local_forw_rule[".+"] = (f) -> begin
    x = f.args[1]; y = f.args[2]
    f.forward_derivative = x.forward_derivative .+ y.forward_derivative
end

local_forw_rule[".exp"] = (f) -> begin
    x = f.args[1]
    f.forward_derivative = exp.(x.value) .* x.forward_derivative
end

local_forw_rule[".log"] = (f) -> begin
    x = f.args[1]
    f.forward_derivative = (1.0 ./ x.value) .* x.forward_derivative
end

local_forw_rule[".tanh"] = (f) -> begin
    x = f.args[1]
    f.forward_derivative = (1 .- tanh.(x.value).^2) .* x.forward_derivative
end

local_forw_rule[".relu"] = (f) -> begin
    x = f.args[1]
    f.forward_derivative = (x.value .> 0) .* x.forward_derivative
end

local_forw_rule[".^"] = (f) -> begin
    x = f.args[1]; n = f.args[2].value
    f.forward_derivative = (n .* (x.value .^ (n .- 1))) .* x.forward_derivative
end
local_forw_over_rev_rule = Dict{String, Function}()

local_forw_over_rev_rule["*"] = (f) -> begin
    x, y = f.args
    if x.need_hess
        x.second_order_derivative .+= f.second_order_derivative * y.value' .+ f.derivative * y.forward_derivative'
    end
    if y.need_hess
        y.second_order_derivative .+= x.value' * f.second_order_derivative .+ x.forward_derivative' * f.derivative
    end
end

local_forw_over_rev_rule[".*"] = (f) -> begin
    x, y = f.args
    if x.need_hess
        x.second_order_derivative .+= f.second_order_derivative .* y.value .+ f.derivative .* y.forward_derivative
    end
    if y.need_hess
        y.second_order_derivative .+= f.second_order_derivative .* x.value .+ f.derivative .* x.forward_derivative
    end
end

local_forw_over_rev_rule[".+"] = (f) -> begin
    x, y = f.args
    
    if isscalar(f)
        if x.need_hess
            x.second_order_derivative += f.second_order_derivative
        end
        if y.need_hess
            y.second_order_derivative += f.second_order_derivative
        end
    else
        if x.need_hess
            x.second_order_derivative .+= f.second_order_derivative
        end
        if y.need_hess
            y.second_order_derivative .+= f.second_order_derivative
        end
    end
end

local_forw_over_rev_rule[".-"] = (f) -> begin
    x, y = f.args
    
    if isscalar(f)
        if x.need_hess
            x.second_order_derivative += f.second_order_derivative
        end
        if y.need_hess
            y.second_order_derivative -= f.second_order_derivative
        end
    else
        if x.need_hess
            x.second_order_derivative .+= f.second_order_derivative
        end
        if y.need_hess
            y.second_order_derivative .-= compress(f.second_order_derivative, shape(y))
        end
    end
end

local_forw_over_rev_rule["./"] = (f) -> begin
    x, z = f.args
    
    if isscalar(f)
        
        z_val = z.value
        z2 = z_val^2
        
        if x.need_hess
            # Pas de '.+=' pour un scalaire !
            x.second_order_derivative += f.second_order_derivative / z_val - f.derivative * (z.forward_derivative / z2)
        end
        
        if z.need_hess
            x_val = x.value
            t1 = -f.second_order_derivative * (x_val / z2)
            
            t2_num = x.forward_derivative * z2 - 2.0 * x_val * z_val * z.forward_derivative
            t2_den = z2^2 # (z_val^2)^2 == z_val^4. t2_den = z2^2 est correct.
            t2 = -f.derivative * (t2_num / t2_den)
            
            # Pas de '.+=' pour un scalaire !
            z.second_order_derivative += t1 + t2
        end
        
    else
 
        z2 = z.value .^ 2
        
        if x.need_hess
            x.second_order_derivative .+= f.second_order_derivative ./ z.value .- f.derivative .* (z.forward_derivative ./ z2)
        end
        
        if z.need_hess
            t1 = -f.second_order_derivative .* (x.value ./ z2)
            t2_num = z2 .* x.forward_derivative .- 2.0 .* x.value .* z.value .* z.forward_derivative
            t2 = -f.derivative .* (t2_num ./ (z2 .^ 2)) 
            
            z.second_order_derivative .+= compress(t1 .+ t2, shape(z))
        end
    end
end

local_forw_over_rev_rule["sum"] = (f) -> begin
    x = f.args[1]
    if x.need_hess
        x.second_order_derivative .+= extend(f.second_order_derivative, shape(x))
    end
end

local_forw_over_rev_rule["sum_all"] = (f) -> begin
    x = f.args[1]
    if x.need_hess
        x.second_order_derivative .+= extend(f.second_order_derivative, size(x.value))
    end
end
local_forw_over_rev_rule[".exp"] = (f) -> begin
    x = f.args[1]
    e = exp.(x.value)
    if x.need_hess
        x.second_order_derivative .+= f.second_order_derivative .* e .+ f.derivative .* (e .* x.forward_derivative)
    end
end

local_forw_over_rev_rule[".log"] = (f) -> begin
    x = f.args[1]
    if x.need_hess
        x.second_order_derivative .+= f.second_order_derivative ./ x.value .+ f.derivative .* ((-1.0 ./ (x.value.^2)) .* x.forward_derivative)
    end
end

local_forw_over_rev_rule[".tanh"] = (f) -> begin
    x = f.args[1]
    t = tanh.(x.value)
    s = 1 .- t.^2  
    if x.need_hess
        x.second_order_derivative .+= f.second_order_derivative .* s .+
                                 f.derivative .* ((-2 .* t .* s) .* x.forward_derivative)
    end
end


local_forw_over_rev_rule[".relu"] = (f) -> begin
    x = f.args[1]
    
    if x.need_hess
        mask = isnothing(f.memory) ? (x.value .> 0) : f.memory

        if isscalar(f)
            x.second_order_derivative += f.second_order_derivative * mask
        else
            x.second_order_derivative .+= f.second_order_derivative .* mask
        end
    end
end

local_forw_over_rev_rule[".^"] = (f) -> begin
    x = f.args[1]
    n = f.args[2].value
    if x.need_hess
        xn1 = x.value .^ (n - 1)
        xn2 = (n > 1) ? x.value .^ (n - 2) : ones_like(x)
        
        t1 = f.second_order_derivative .* (n .* xn1) 
        t2 = f.derivative .* (n*(n-1) .* xn2 .* x.forward_derivative) 
        
        x.second_order_derivative .+= t1 .+ t2
    end
end

local_forw_over_rev_rule["maximum"] = (f) -> begin
    x = f.args[1]
    sx, sf = shape(x), shape(f)


    if x.need_hess
        if sf[1] == 1
            for j in 1:sx[2]
                idx = f.memory[j]
                
                x.second_order_derivative[idx, j] += f.second_order_derivative[1, j]
            end
        else
            for i in 1:sx[1]
                idx = f.memory[i]

                x.second_order_derivative[i, idx] += f.second_order_derivative[i, 1]
            end
        end
    end
end


local_forw_over_rev_rule["maximum_all"] = (f) -> begin
    x = f.args[1]
    if isnothing(f.memory)
        f.memory = argmax(x.value)
    end
    
    if x.need_hess
        idx = f.memory
        x.second_order_derivative[idx] += f.second_order_derivative
    end
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
    y_node = VectNode(y, false, false)
	haskey(check_param, sym) ? check_param[sym](shape(x), shape(y_node)) : error("symbole $sym not implemented")
    return VectNode(sym, [x, y_node], op.(x.value, y_node.value))
end

function Base.broadcasted(op::Function, x::Union{Number,AbstractArray}, y::VectNode)
    sym = "." * string(op)
    x_node = VectNode(x, false, false)
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
	y_node = VectNode(y, false, false)
	haskey(check_param, "*") ? check_param["*"](shape(x), shape(y_node)) : error("symbole * not implemented")
	return VectNode("*", [x, y_node], x.value * y)
end

Base.:*(x::Union{Number, AbstractArray}, y::VectNode) = begin
	if isa(x, Number) || isscalar(y) return x.*y end
	x_node = VectNode(x, false, false)
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

function forward!(f::VectNode)
    if isempty(f.args)
        if isnothing(f.forward_derivative)
            f.forward_derivative = zeros_like(f)
        end
        return
    end
    for arg in f.args
        if isnothing(arg.forward_derivative)
            forward!(arg)
        end
    end
    local_forw_rule[f.op](f)
end

function second_order_backward(f::VectNode)
    
    visited = Set{VectNode}()
    topo = VectNode[]
    topo_sort!(visited, topo, f)  
    reverse!(topo)                 

    for node in topo
        if !isnothing(node.op)
            local_forw_over_rev_rule[node.op](node)
        end
    end
    return f
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

function backward!(f::VectNode)
    visited = Set{VectNode}()
    topo = VectNode[]
    topo_sort!(visited, topo, f)
    reverse!(topo)
    if isa(f.derivative, Float64)
        f.derivative = 1.0
    else
        error("you're suppose to have a loss function !")
    end
    for n in topo
        if !isnothing(n.op)
    		local_back_rule[n.op](n)
		end
    end
    return f
end

function hvp(f, x, v)
    x_nodes = Flatten([VectNode(xi, true, true, vi) for (xi, vi) in zip(x.components, v.components)])
    last_node = f(x_nodes)
    
    forward!(last_node)

    backward!(last_node)    
            
    second_order_backward(last_node)  

    hv = zero(v)
    for i in eachindex(x.components)
        hv.components[i] .= x_nodes.components[i].second_order_derivative
    end
    return hv
end

function hessian(f, x::Flatten)
    n = length(x)
    H = zeros(n, n)
    basis = zero(x)

    for i in 1:n
        v = zero(basis)
        v[i] = 1.0

        hv = hvp(f, x, v)
        H[:, i] .= reduce(vcat, vec.(hv.components))
    end

    return H
end

function slow_grad!(f, g::Flatten, x::Flatten)
    for i in 1:length(x.components)
        xi = x.components[i]
        if isa(xi, Number)
            x_nodes = Flatten(VectNode.(x.components))
            x_nodes.components[i].forward_derivative = 1.0
            expr = f(x_nodes)
            forward!(expr)
            val = expr.forward_derivative
            g.components[i] = isa(val, Number) ? val : val[1]
        else
            for idx in eachindex(xi)
                x_nodes = Flatten(VectNode.(x.components))
                seed = zeros_like(x_nodes.components[i].value)
                seed[idx] = 1.0
                x_nodes.components[i].forward_derivative = seed
                expr = f(x_nodes)
                forward!(expr)
                val = expr.forward_derivative
                if isa(g.components[i], Number)
                    g.components[i] = isa(val, Float64) ? val : val[1]
                else
                    g.components[i][idx] = isa(val, Float64) ? val : val[1]
                end
            end
        end
    end
	return g
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
	g = zero(x)
	gradient!(f, g, x)
end

end


