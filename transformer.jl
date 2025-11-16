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
	hessian_vector::Union{Nothing, Float64, Array{Float64}}
	memory::Any
	need_grad::Bool
	need_hess::Bool
end
# -------------------- init -------------------- #
function VectNode(op, args, value, need_grad = true, need_hess = false, f_grad = nothing)
    v = _make_value(value)
	grad = (need_grad ? zeros_like(v) : nothing)
	f_gradi = isnothing(f_grad) ? nothing : _make_value(f_grad)
	hv = (need_hess ? zeros_like(v) : nothing)
	return VectNode(op, args, v, grad, f_gradi, hv, nothing, need_grad, need_hess)
end
VectNode(x::Number, need_grad = true, need_hess = false, f_grad = nothing) = VectNode(nothing, VectNode[], x, need_grad, need_hess, f_grad)
VectNode(x::AbstractArray, need_grad = true, need_hess = false, f_grad = nothing) = VectNode(nothing, VectNode[], x, need_grad, need_hess, f_grad)

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

# -------------------- class functions -------------------- #
hash(v::VectNode, h::UInt) = hash(objectid(v), h)
==(a::VectNode, b::VectNode) = a === b
Base.length(v::VectNode) = length(v.value)
Base.size(v::VectNode) = size(v.value)
isscalar(v::VectNode) = isa(v.value, Float64)
shape(v::VectNode) = isa(v.value, Float64) ? () : size(v.value)
Base.size(v::VectNode) = isa(v.value, Float64) ? () : size(v.value)
Base.size(v::VectNode, int::Int) = isa(v.value, Float64) ? () : size(v.value)[int]
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
	if sx != () && sy != () && length(sx) != 2 && sx != sy error("addition broadcasting for different size not implemented yet") end
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
check_param[".sqrt"] = (sx) -> begin
    return 
end
check_param["sel_cols"] = (sx) -> begin
	if length(sx) != 2 error("sel_cols not implemented for $sx") end
end
check_param["'"] = (sx) -> begin
    if length(sx) != 2 error("adjoint not implemented for $sx") end
end
check_param["getindex"] = (sx) -> begin
    if sx == () error("getindex not implemented for $sx") end
end
check_param["applymask"] = (sx) -> begin
    if length(sx) != 2 error("applymask not implemented for $sx") end
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
local_back_rule[".+"] = (f) -> begin
    x = f.args[1]; y = f.args[2]; sx = shape(x); sy = shape(y)
    if x.need_grad x.derivative = x.derivative .+ f.derivative end
    if y.need_grad y.derivative = y.derivative .+ compress(f.derivative, sy) end
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
local_back_rule["sel_cols"] = (f) -> begin
	x = f.args[1] 
	idxs = f.memory
	for (j, idx) in enumerate(idxs)
		x.derivative[:, idx] .+= f.derivative[:, j]
	end
end
local_back_rule["'"] = (f) -> begin
    x = f.args[1]
    x.derivative = x.derivative .+ f.derivative'
end
local_back_rule["getindex"] = (f) -> begin
    x = f.args[1]
    keys = f.memory
    x.derivative[keys...] = x.derivative[keys...] .+ f.derivative
end
local_back_rule["applymask"] = (f) -> begin
    x = f.args[1]
    mask = f.memory
    x.derivative[mask .!= 1] .+= f.derivative[mask .!= 1]
end
local_back_rule["vcat"] = (f) -> begin
    offset = 0
    for h in f.args
        d = size(h.value, 1)  # nombre de lignes de ce head
        h.derivative = h.derivative .+ f.derivative[offset+1:offset+d, :]
        offset += d
    end
end
local_back_rule[".sqrt"] = (f) -> begin
    x = f.args[1]
    if isnothing(f.memory)
        f.memory = 1.0 ./ (2 .* sqrt.(x.value))
    end
    x.derivative .+= f.derivative .* f.memory
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

function Base.adjoint(v::VectNode)
    check_param["'"](shape(v))
    return VectNode("'", [v], v.value')
end

function Base.getindex(v::VectNode, keys...)
    r = VectNode("getindex", [v], getindex(v.value, keys...))
    r.memory = keys
    return r
end

sel_cols(M::VectNode, idxs::Vector{Int}) = begin
	check_param["sel_cols"](shape(M))
	function sel_cols(M::Matrix{Float64}, idxs::Vector{Int})
		n_tokens = length(idxs)
		d_model = size(M, 1)
		embeded_vectors = Matrix{Float64}(undef, d_model, n_tokens)
		for (i, idx) in enumerate(idxs)
			if idx == -1
				embeded_vectors[:, i] .= 0.0
			else
				embeded_vectors[:, i] = M[:, idx]
			end
		end
		return embeded_vectors
	end
	v = VectNode("sel_cols", [M], sel_cols(M.value, idxs))
	v.memory = idxs
	return v
end


function applymask(M::VectNode, mask::Matrix)
    check_param["applymask"](shape(M))
    masked_value = copy(M.value)
    masked_value[mask .== 1] .= -Inf
    r = VectNode("applymask", [M], masked_value)
    r.memory = mask
    return r
end

function Base.vcat(heads::VectNode...)
    values = vcat([h.value for h in heads]...)
    r = VectNode("vcat", collect(heads), values) # to verify ! 
    return r
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
        error("you're suppose to have a loss function !")
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
	g = zero(x)
	gradient!(f, g, x)
end


# ---------------------------- #
# ---- partie transformer ---- #
# ---------------------------- #



import ..Flatten
using LinearAlgebra
using Random
path = "/home/thoma/unnif/Q7/LINMA2472 (algo_in_data_sciences)/LINMA2472/HomeworkAD/text.txt"

struct Word_vect_encoder_decoder
    vocab_encoder::Vector{Tuple{String, String, String}}
    vocab::Vector{String}
    idx_encoder::Dict{String, Int64}
    idx_decoder::Dict{Int64, String}
end

function get_dict(vocab)
    world_to_idx = Dict{String, Int64}()
    idx_to_word = Dict{Int64, String}()
    for (i, string) in enumerate(vocab)
        world_to_idx[string] = i
        idx_to_word[i] = string
    end
    return world_to_idx, idx_to_word
end

function word_to_idx(encoded_sentence::Vector{String}, word_to_idx_dict::Dict{String, Int})
    return [get(word_to_idx_dict, s, -1) for s in encoded_sentence]
end

function idx_to_word(idxs::Union{Vector{Int}, Int}, idx_to_word_dict::Dict{Int, String})
    if isa(idxs, Int)
        return get(idx_to_word_dict, idxs, "<?>")
    else
        return [get(idx_to_word_dict, i, "<?>") for i in idxs]
    end
end

function string_to_list(s::String)
    out = String[]
    for c in s
        if c == ' '
            push!(out, "</w>")
        else
            push!(out, lowercase(string(c)))
        end
    end
    push!(out, "</w>")
    return out
end

function train_encode(strings::Union{Vector{String}, String}, max_it::Int)
    strings = (isa(strings, String) ? string_to_list(strings) : strings)
    vocab = String[]
    vocab_set = Set{String}()
    encoder = Vector{Tuple{String, String, String}}()

    for string in strings
        if !(string in vocab_set)
            push!(vocab, string)
            push!(vocab_set, string)
        end
    end

    for _ in 1:max_it
        if length(strings) < 2
            break
        end

        max_val = 0
        max_stat = nothing
        stats = Dict{Tuple{String, String}, Int}()

        i = 1
        while i < length(strings)
            j = findfirst(x -> x == "</w>", strings[i:end])
            j = isnothing(j) ? length(strings) - i + 1 : j
            word = strings[i:(i + j - 1)]

            for k in 1:(length(word) - 1)
                a, b = word[k], word[k + 1]
                if occursin("</w>", a)
                    continue
                end
                pair = (a, b)
                stats[pair] = get(stats, pair, 0) + 1
                if stats[pair] > max_val
                    max_val = stats[pair]
                    max_stat = pair
                end
            end
            i += j
        end

        if max_stat === nothing || max_val < 1
            break
        end

        new_strings = String[]
        i = 1
        while i <= length(strings)
            if i < length(strings)
                a, b = strings[i], strings[i + 1]
                if occursin("</w>", a)
                    push!(new_strings, a)
                    i += 1
                    continue
                end
                if (a, b) == max_stat
                    push!(new_strings, a * b)
                    i += 2
                    continue
                end
            end
            push!(new_strings, strings[i])
            i += 1
        end

        push!(encoder, (max_stat[1], max_stat[2], max_stat[1] * max_stat[2]))
        if !(max_stat[1] * max_stat[2] in vocab_set)
            push!(vocab, max_stat[1] * max_stat[2])
            push!(vocab_set, max_stat[1] * max_stat[2])
        end

        strings = new_strings
    end

    word_to_idx_d, idx_to_word_d = get_dict(vocab)  
    return Word_vect_encoder_decoder(encoder, vocab, word_to_idx_d, idx_to_word_d), word_to_idx(strings, word_to_idx_d)
end

function encode(strings::Union{Vector{String}, String}, encoder_decoder::Word_vect_encoder_decoder)
    encoder = encoder_decoder.vocab_encoder
    strings = (isa(strings, String) ? string_to_list(strings) : strings)

    for (a, b, ab) in encoder
        new_string = String[]
        j = 1
        while j <= length(strings)
            if j < length(strings) && strings[j] == a && strings[j+1] == b
                push!(new_string, ab)
                j += 2
            else
                push!(new_string, strings[j])
                j += 1
            end
        end
        strings = new_string
    end

    return word_to_idx(strings, encoder_decoder.idx_encoder)
end

function decode(idxs::Union{Vector{Int}, Int}, encoder_decoder::Word_vect_encoder_decoder)
    tokens = idx_to_word(idxs, encoder_decoder.idx_decoder)
    if isa(tokens, String)
        tok = tokens
        return occursin("</w>", tok) ? replace(tok, "</w>" => "") : tok
    else
        s = ""
        for tok in tokens
            if occursin("</w>", tok)
                s *= replace(tok, "</w>" => "") * " "
            else
                s *= tok
            end
        end
        return rstrip(s)
    end
end

function read_corpus(path::String)
    v = Vector{Vector{String}}()
    open(path, "r") do f
        text = read(f, String)
        words = split(text)
        v = [vcat([lowercase(string(c)) for c in collect(w)], ["</w>"]) for w in words]
    end
    return [s for sub in v for s in sub]
end

# --- utilitaires
function softmax(M, dim::Int) # dim=1 -> softmax par ligne (pour scores attention), dim=2 -> par colonne
    if dim == 1
        m = maximum(M, dims=2)                   # max par ligne -> taille (n,1)
        ex = exp.(M .- m)                        # broadcast soustraction ligne-wise
        return ex ./ sum(ex, dims=2)             # normalize par ligne
    elseif dim == 2
        m = maximum(M, dims=1)                   # max par colonne -> taille (1,n)
        ex = exp.(M .- m)
        return ex ./ sum(ex, dims=1)             # normalize par colonne
    else
        throw(ArgumentError("dim must be 1 or 2"))
    end
end

# mean générique compatible avec dims
function mymean(M; dims=nothing)
    if isnothing(dims)
        if isa(M, Number)
            return M
        else
            return sum(M) / length(M)
        end
    else
        if dims == 1
            return sum(M, dims=1) ./ size(M, 1)   # moyenne sur les lignes -> renvoie (1, n_col)
        elseif dims == 2
            return sum(M, dims=2) ./ size(M, 2)   # moyenne sur les colonnes -> renvoie (n_row, 1)
        else
            throw(ArgumentError("dims must be 1 or 2"))
        end
    end
end
mymean(M, dim) = mymean(M; dims=dim)

# applique un masque (1 -> interdit)
function applymask(M::Matrix, mask)
    M_masked = copy(M)
    M_masked[mask .== 1] .= -Inf
    return M_masked
end

# --- attention single-head (Q, K, V ont shape (d_k, n_tokens))
# retourne une matrice (d_k, n_tokens)
function attention(Q, K, V, d_k::Int; need_mask::Bool=true)
    # scores : (n_tokens, n_tokens)
    M = (K' * Q) / sqrt(float(d_k))          
    if need_mask
        mask = tril(ones(size(M)), -1)             
        M = applymask(M, mask)
    end
    A = softmax(M, 2)                             
    return V * A                                  # (d_k, n_tokens) à transposé ?
end

# --- multi-head
function multihead_attention_block(Z, Q, K, V, WO, nbr_of_head::Int; with_mask::Bool=true)
    d_model, n_tokens = size(Z)
    @assert d_model % nbr_of_head == 0
    d_k = div(d_model, nbr_of_head)
    heads = Vector{}(undef, nbr_of_head)
    for i in 1:nbr_of_head
        idx = ((i-1)*d_k + 1):(i*d_k)
        Qi = Q[idx, :]    # (d_k, n)
        Ki = K[idx, :]
        Vi = V[idx, :]
        heads[i] = attention(Qi, Ki, Vi, d_k; need_mask=with_mask)  # (d_k, n)
    end
    concat = vcat(heads...)                         # (d_model, n)
    return WO * concat                              # (d_model, n) ; WO doit être (d_model, d_model_out) typiquement (d_model,d_model)
end

# --- util sel_cols
function sel_cols(M::Matrix{Float64}, idxs::Vector{Int})
    n_tokens = length(idxs)
    d_model = size(M, 1)
    embeded_vectors = Matrix{Float64}(undef, d_model, n_tokens)
    for (i, idx) in enumerate(idxs)
        if idx == -1
            embeded_vectors[:, i] .= 0.0
        else
            embeded_vectors[:, i] = M[:, idx]
        end
    end
    return embeded_vectors
end

function transformer_embedding(X::Vector{Int}, W_emb, P)
    n_tokens = length(X)
    _, max_pos = size(P)
    @assert n_tokens == max_pos "Positional matrix P is not of good size"
    return sel_cols(W_emb, X) .+ P                  # (d_model, n_tokens)
end

# --- normalization & FFN (LayerNorm sur la dimension des features -> mean over rows, dims=1)
function Layer_norm(Y; eps::Float64=1e-5)
    mu = mymean(Y, dims=1)                            # (1, n_tokens)
    sig2 = mymean((Y .- mu).^2, dims=1)               # (1, n_tokens)
    return (Y .- mu) ./ sqrt.(sig2 .+ eps)         # broadcasting ok -> (d_model, n_tokens)
end

relu(x) = max(0, x)                                # utilisé en broadcasting relu.(...)

function FFN(Z, W1, b1, W2, b2)
    hidden = relu.(W1 * Z .+ b1)                   # W1 (d_ff, d_model) * Z (d_model, n) => (d_ff, n)
    return W2 * hidden .+ b2                       # (d_model, n)
end

function decoder_output(Z, W_out)
    logits = W_out * Z                              # (vocab_size, n_tokens)
    probs = softmax(logits, 2)                      # softmax par colonne (chaque colonne = distribution sur vocab pour une position)
    return logits, probs
end

# --- modèle principal
function transformer_model(X::Vector{Int}, W::Flatten, param::Dict)
    W_emb = W.components[1]                          # (d_model, vocab_size)
    P = W.components[2]                              # (d_model, n_tokens)
    l = param["nbr_of_layers"]
    nbr_of_head = param["nbr_of_head"]

    Z = transformer_embedding(X, W_emb, P)          # (d_model, n_tokens)

    for j in 0:(l-1)
        base = 3 + 8*j
        WQ, WK, WV, WO = W.components[base+0], W.components[base+1], W.components[base+2], W.components[base+3]
        W1, W2, b1, b2 = W.components[base+4], W.components[base+5], W.components[base+6], W.components[base+7]

        Q, K, V = WQ * Z, WK * Z, WV * Z            # chaque -> (d_model, n_tokens)
        att = multihead_attention_block(Z, Q, K, V, WO, nbr_of_head; with_mask=false)             # ! (deleteme)

        Z = Layer_norm(Z + att)                     # résiduel + layernorm

        Z = Layer_norm(Z + FFN(Z, W1, b1, W2, b2))
    end

    W_out = W.components[end]                       # (vocab_size, d_model)
    return decoder_output(Z, W_out)
end

# --- génération des poids (inchangé sauf quelques types)
function generate_random_W(param::Dict, vocab_size::Int)
    nbr_of_layers = param["nbr_of_layers"]
    nbr_of_head = param["nbr_of_head"]
    n_tokens = param["sample_size"] - 1
    d_model = param["d_model"]

    @assert d_model % nbr_of_head == 0

    Random.seed!(42)
    components = Any[]

    push!(components, randn(d_model, vocab_size) * sqrt(1/d_model))   # W_emb -> (d_model, vocab_size)
    push!(components, randn(d_model, n_tokens) * sqrt(1/d_model))     # P -> (d_model, n_tokens)

    for _ in 1:nbr_of_layers
        WQ = randn(d_model, d_model) * sqrt(1/d_model)
        WK = randn(d_model, d_model) * sqrt(1/d_model)
        WV = randn(d_model, d_model) * sqrt(1/d_model)
        WO = randn(d_model, d_model) * sqrt(1/d_model)

        d_ff = 4*d_model
        W1 = randn(d_ff, d_model) * sqrt(2/d_model)
        W2 = randn(d_model, d_ff) * sqrt(2/d_ff)
        b1 = zeros(d_ff, 1)
        b2 = zeros(d_model, 1)
        append!(components, [WQ, WK, WV, WO, W1, W2, b1, b2])
    end

    push!(components, randn(vocab_size, d_model) * sqrt(1/d_model))  # W_out -> (vocab_size, d_model)
    return Flatten(components)
end

# --- samples & loss (inchangés, quelques petits ajustements de robustesse)
function ssamples(encoded_corpus::Vector{Int}, sample_size::Int)
    n = length(encoded_corpus)
    out = Vector{Vector{Int}}()
    for i in 1:(n - sample_size + 1)
        push!(out, encoded_corpus[i:(i + sample_size - 1)])
    end
    return out
end

function cross_entropy(Y_est, Y)
    @assert size(Y_est) == size(Y)
    return -sum(Y .* log.(Y_est .+ 1e-9)) / size(Y, 2)
end

function grad_step!(l, W::Flatten, step_size::Float64)
    g = gradient(l, W)
    for i in eachindex(W.components)
        W.components[i] .-= step_size .* g.components[i]
    end
end


function transformer(corpus, param) # build a transformer (return W) and train it on corpus 
    encoder_decoder, encoded_corpus = train_encode(corpus, param["max_tokenisation_step"])
    samples = ssamples(encoded_corpus, param["sample_size"])
    W = generate_random_W(param, length(encoder_decoder.vocab))

    loss(W) = begin
        s = 0.0
        for i in 1:(length(samples))
            x = samples[i][1:end-1]  
            y_true = samples[i][2:end]
            _, probs = transformer_model(x, W, param)

            Y = zeros(size(probs))
            for (col, idx) in enumerate(y_true)
                if idx != -1
                    Y[idx, col] = 1
                end
            end

            s += cross_entropy(probs, Y)
        end
        return s / length(samples)
    end

    for i in 1:param["nbr_of_obt_step"]
        println("$i ) loss = $(loss(W)) , step_size = $(param["step_size"])")
        param["step_size"] *= 0.999
        grad_step!(loss, W, param["step_size"])
    end

    return encoder_decoder, W
    
end

function create_dumb_text(path::String; n_tokens::Int=1000, seed::Integer=42)
    Random.seed!(seed)
    letters = ['a','b','c']
    # transition rules for next letter given previous letter
    # from 'a': b 90%, c 10%
    # from 'b': c 100%
    # from 'c': a/b/c each 1/3
    next_letter(prev) = begin
        r = rand()
        if prev == 'a'
            return r < 0.5 ? 'b' : 'c'
        elseif prev == 'b'
            return r < 0.5 ? 'a' : 'c'
        else # prev == 'c'
            return r < 0.5 ? 'a' : 'b'
        end
    end

    # initialize first letter uniformly
    first = letters[rand(1:3)]
    seq = Vector{Char}(undef, 2 * n_tokens)  # letter, space, letter, space...
    cur = first
    for i in 1:n_tokens
        seq[2*i-1] = cur
        seq[2*i] = ' '
        cur = next_letter(cur)
    end

    text = String(seq)
    open(path, "w") do io
        write(io, text)
    end
    return path
end

# create_dumb_text("/home/thoma/unnif/Q7/LINMA2472 (algo_in_data_sciences)/LINMA2472/HomeworkAD/text.txt", n_tokens = 100)


param = Dict{String, Union{Int, Float64}}()
param["nbr_of_layers"] = 2
param["nbr_of_head"] = 8
param["sample_size"] = 128  # nbr_of_token = sample_size - 1 !
param["d_model"] = param["nbr_of_head"] * 64
param["max_tokenisation_step"] = 10000
param["nbr_of_obt_step"] = 10
param["step_size"] = 0.01

encoder_decoder, W = transformer(read_corpus(path)[1:10000], param)

# test = encode("a b", encoder_decoder)
# test2 = encode("a c", encoder_decoder)
# logits, probs = transformer_model(test, W, param)
# logits2, probs2 = transformer_model(test2, W, param)
# println("\n ---- \n")
# println(probs[:, 2])
# println(probs2[:, 2])
# println(encoder_decoder.vocab)
# println("0 = ", probs[:, 1] .- probs2[:, 1])

using Serialization

function save_model(path::String, encoder_decoder, W)
    open(path, "w") do io
        serialize(io, (encoder_decoder, W))
    end
end

function load_model(path::String)
    open(path, "r") do io
        return deserialize(io) # returns (encoder_decoder, W)
    end
end

# example: save current encoder_decoder and W
save_model("/home/thoma/unnif/Q7/LINMA2472 (algo_in_data_sciences)/LINMA2472/HomeworkAD/transformer_model.bin", encoder_decoder, W)
encoder_decoder_loaded, W_loaded = load_model("/home/thoma/unnif/Q7/LINMA2472 (algo_in_data_sciences)/LINMA2472/HomeworkAD/transformer_model.bin")

# test = encode("b b", encoder_decoder_loaded)
# logits, probs = transformer_model(test, W_loaded, param)
# println("\n ---- \n")
# println(probs[:, 1])
# println(encoder_decoder.vocab)