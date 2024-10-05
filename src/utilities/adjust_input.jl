"""
$(SIGNATURES)

# Arguments
- `input`: input data
- `input_dim::Int`: input dimension
- `tdim::Int`: time dimension
- `margin::Int=2`: margin for time dimension (error protection)

# Returns
- `input::Array{T,2}`: adjusted input
"""
function adjust_input(input, input_dim, tdim; margin=2)
    if input_dim == tdim 
        warn("Input dimension is equal to time dimension. This function will treat the input as a vector of length input_dim * tdim")
    end

    @assert tdim > 3 "Time dimension must be greater than 3"

    for tdim_ in tdim-margin:tdim+margin
        if size(input,1) == input_dim && size(input,2) == tdim_
            # Input is correct size
            return input
        elseif size(input,1) == tdim_ && size(input,2) == input_dim
            # Transpose input
            return input'
        elseif ndims(input) == 1 && length(input) == tdim_ && input_dim ==1
            # Input is a vector of length tdim_, reshape to (1, tdim_)
            return reshape(input, 1, tdim_)
        elseif ndims(input) ==1 && length(input) == input_dim && tdim_ ==1
            # Input is a vector of length input_dim, reshape to (input_dim, 1)
            return reshape(input, input_dim, 1)
        elseif ndims(input) ==1 && length(input) == input_dim * tdim_
            # Input is a vector, reshape to (input_dim, tdim_)
            return reshape(input, input_dim, tdim_)
        end
    end

    # If none of the conditions are met, throw an error
    @debug "Size of input: $(size(input))"
    @debug "Input dimension: $(input_dim)"
    @debug "Time dimension: $(tdim)"
    error("Input dimensions are not compatible with input_dim and tdim_")
end

