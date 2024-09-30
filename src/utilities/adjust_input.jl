"""
$(SIGNATURES)
"""
function adjust_input(input, input_dim, tdim)
    if input_dim == tdim 
        warn("Input dimension is equal to time dimension. This function will treat the input as a vector of length input_dim * tdim")
    end

    if size(input,1) == input_dim && size(input,2) == tdim
        # Input is correct size
        return input
    elseif size(input,1) == tdim && size(input,2) == input_dim
        # Transpose input
        return input'
    elseif ndims(input) == 1 && length(input) == tdim && input_dim ==1
        # Input is a vector of length tdim, reshape to (1, tdim)
        return reshape(input, 1, tdim)
    elseif ndims(input) ==1 && length(input) == input_dim && tdim ==1
        # Input is a vector of length input_dim, reshape to (input_dim, 1)
        return reshape(input, input_dim, 1)
    elseif ndims(input) ==1 && length(input) == input_dim * tdim
        # Input is a vector, reshape to (input_dim, tdim)
        return reshape(input, input_dim, tdim)
    else
        @debug "Size of input: $(size(input))"
        @debug "Input dimension: $(input_dim)"
        @debug "Time dimension: $(tdim)"
        error("Input dimensions are not compatible with input_dim and tdim")
    end
end

