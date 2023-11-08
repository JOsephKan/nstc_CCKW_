# import library
using LinearAlgebra

# module of Fourier transform

module Fourier_trans

    export DFT, IDFT

    function operator(arr :: Array{Float64, 2}, dimension :: Int8)
        # Declare section
        N        :: Int16               # dimension to do Fourier transform
        n, k, eo :: Array{Float64, 2}   # coefficient and operator

        # Execute section
        N = round(size(arr)[dimension])

        n = LinRange(0, N-1, N)
        k = LinRange(0, N-1, N) * (2*pi/N)

        eo = k'*n / N
        return eo
    end

    function DFT(arr :: Array{Float64, 2})
        # Declare section
        eo     :: Array{Float64, 2} # operator 
        Ck, Sk :: Array{Float64, 2} # Array of DFT

        # Execute section
        eo = operator(arr, 2)
        Ck = real.(eo) * arr'
        Sk = imag.(eo) * arr'

        return Ck, Sk
    end

    function IDFT(C_k :: Array{Float64, 2}, S_k :: Array{Float64, 2}, arr :: Array{Float64, 2})
        # Declare section
        eo :: Array{Float64, 2}
        Co, So :: Array{Float64, 2}
        # Execute section
        eo = operator(arr, 1)
        Co = (real.(eo) * C_k)'
        So = (imag.(eo) * S_k)'
    
        return Co, So
    end

end

module power_spectrum

    import ..Fourier_trans: DFT

    function Nyquist(arr :: Array{Float64, 2}, arr_o :: Array{Float64, 2}, axis :: Int8)
        # Declare section
        arr_new :: Array{Float64, 2}

        # Execute section
        arr_new = arr[1:round(size(arr_o, axis))]
        arr_new *= 2

        return arr_new
    end

    function power_coe(arr)
        # Declare section
        Ck, Sk     :: Array{Float64, 2}
        A, B, a, b :: Array{Float64, 2}

        # Execute section
        Ck ,Sk = DFT(arr)
        Ck = Nyquist(Ck, arr, 2)
        Sk = Nyquist(Sk, arr, 2)

        A, B = DFT(Ck)
        a, b = DFT(Sk)
        A = Nyquist(A, arr, 1)
        B = Nyquist(B, arr, 1)
        a = Nyquist(a, arr, 1)
        b = Nyquist(b, arr, 1)

        return [A, B, a, b]
    end

    function power_spec(arr)
        # Declare section
        power_pos, power_neg :: Array{Float64, 2}
        A, B, a, b :: Array{Float64, 2}

        # Execute section
        A, B, a, b = power_coe(arr)
        power_pos = 1/8 * (
            A .^ 2 + B .^ 2 + a .^ 2 + b .^ 2
        ) + 1/4 * (a.*B - b.*A)
        power_neg = 1/8 * (
            A .^ 2 + B .^ 2 + a .^ 2 + b .^ 2
        ) - 1/4 * (a.*B - b.*A)

        return power_pos, power_neg
    end
end