# import package
import numpy as np
from numba import njit

# define DFT class
class Fourier_trans:
    def __init__(self):
        self.N = None
        self.n = None
        self.k = None
        self.C_k = None
        self.S_k = None
        self.c_i = None
        self.s_i = None
        self.N_i = None
        self.n_i = None
        self.k_i = None
        self.C_k_i = None
        self.S_k_i = None

    def DFT(self, arr):
        self.N = arr.shape[1]

        self.n = np.matrix(np.linspace(0, self.N - 1, self.N))
        self.k = 2 * np.pi * np.matrix(np.linspace(0, self.N - 1, self.N)) / self.N

        self.C_k = np.cos(+self.k.T * self.n) * np.matrix(arr).T / (self.N)
        self.S_k = np.sin(+self.k.T * self.n) * np.matrix(arr).T / (self.N)

        return self.C_k, self.S_k

    def IDFT(self, C_k, S_k, arr):
        self.N_i = round(arr.shape[0])
        self.n_i = np.matrix(np.linspace(0, self.N_i - 1, self.N_i))
        self.k_i = (
            np.matrix(np.linspace(0, self.N_i - 1, self.N_i)) * 2 * np.pi / self.N_i
        )
        self.c_i = np.cos(self.k_i.T * self.n_i[:, : round(self.N_i / 2)])
        self.s_i = np.sin(self.k_i.T * self.n_i[:, : round(self.N_i / 2)])
        self.C_k_i = (self.c_i * C_k).T
        self.S_k_i = (self.s_i * S_k).T

        return self.C_k_i, self.S_k_i

# define power spectrum class
class power_spectrum(Fourier_trans):
    def __init__(self):
        self.C_k_p = None
        self.S_k_p = None
        self.A_c = None
        self.B_c = None
        self.a_c = None
        self.b_c = None
        self.A = None
        self.B = None
        self.a = None
        self.b = None
        self.power_pos = None
        self.power_neg = None

    def Nyquist(self, arr, arr_o, axis):
        arr_new = arr[0 : round(arr_o.shape[axis] / 2), :]
        arr_new = arr_new * 2
        return arr_new

    def power_coe(self, arr):
        self.C_k_p, self.S_k_p = self.DFT(arr)
        self.C_k_p = self.Nyquist(self.C_k_p, arr, 1)
        self.S_k_p = self.Nyquist(self.S_k_p, arr, 1)

        self.A_c, self.B_c = self.DFT(self.C_k_p)
        self.a_c, self.b_c = self.DFT(self.S_k_p)

        self.A_c = self.Nyquist(self.A_c, arr, 0) 
        self.B_c = self.Nyquist(self.B_c, arr, 0) 
        self.a_c = self.Nyquist(self.a_c, arr, 0) 
        self.b_c = self.Nyquist(self.b_c, arr, 0) 

        return np.asarray([self.A_c, self.B_c, self.a_c, self.b_c])

    def power_spec(self, arr):
        self.A, self.B, self.a, self.b = self.power_coe(arr)

        self.power_pos = 1 / 8 * (
            np.power(self.A, 2)
            + np.power(self.B, 2)
            + np.power(self.a, 2)
            + np.power(self.b, 2)
        ) + 1 / 4 * (np.multiply(self.a, self.B) - np.multiply(self.b, self.A))
        self.power_neg = 1 / 8 * (
            np.power(self.A, 2)
            + np.power(self.B, 2)
            + np.power(self.a, 2)
            + np.power(self.b, 2)
        ) - 1 / 4 * (np.multiply(self.a, self.B) - np.multiply(self.b, self.A))

        return self.power_pos, self.power_neg

class reconstruction(power_spectrum):
    def __init__(self):
        self.A_r = None
        self.B_r = None
        self.a_r = None
        self.b_r = None
        self.east_r = None
        self.east_i = None
        self.west_r = None
        self.west_i = None
        self.east = None
        self.west = None
        self.re_wave_i = None
        self.re_wave_r = None
        self.re_wave = None
        self.A_e = None
        self.B_e = None
        self.a_e = None
        self.b_e = None
        self.real_east = None
        self.real_west = None
        self.imag_east = None
        self.imag_west = None
        self.real_wind = None
        self.imag_wind = None
        self.wind_C = None
        self.wind_S = None
        self.wind_C_i = None
        self.wind_S_i = None
        self.wind_inv1 = None
        self.wind_R = None
        self.wind_I = None

    def recon_wave(self, arr):
        self.A_r, self.B_r, self.a_r, self.b_r = self.power_coe(arr)
        self.east_r, self.east_i = self.IDFT(self.A_r, self.B_r, arr)
        self.west_r, self.west_i = self.IDFT(self.a_r, self.b_r, arr)

        self.east = self.east_r + self.east_i
        self.west = self.west_r + self.west_i

        self.re_wave_r, self.re_wave_i = self.IDFT(self.east, self.west, arr.T)
        self.re_wave = self.re_wave_r + self.re_wave_i

        return self.re_wave

    def e_w_trans(self, arr_c):
        self.A_e, self.B_e, self.a_e, self.b_e = arr_c
        self.real_east = 1 / 4 * (self.A_e - self.b_e)
        self.imag_east = 1 / 4 * (-self.B_e - self.a_e)
        self.real_west = 1 / 4 * (self.A_e + self.b_e)
        self.imag_west = 1 / 4 * (self.B_e - self.a_e)
        return np.array(
            [self.real_east, self.imag_east, self.real_west, self.imag_west]
        )

    def east_recon(self, arr, arr_item):
        self.real_wind, self.imag_wind = arr_item
        self.wind_C = (1 + 0j) * (self.real_wind) + (0 + 1j) * (self.imag_wind)
        self.wind_S = (1 + 0j) * (self.imag_wind) - (0 + 1j) * (self.real_wind)
        self.wind_C_i, self.wind_S_i = self.IDFT(self.wind_C, self.wind_S, arr)
        self.wind_inv1 = self.wind_C_i - self.wind_S_i
        self.wind_R = np.real(self.wind_inv1)
        self.wind_I = np.imag(self.wind_inv1)

        return np.asarray([self.wind_R, self.wind_I])

def genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0, Ahe=[50, 25, 12]):
    """
    Function to derive the shallow water dispersion curves. Closely follows NCL version.

    input:
        nWaveType : integer, number of wave types to do
        nPlanetaryWave: integer
        rlat: latitude in radians (just one latitude, usually 0.0)
        Ahe: [50.,25.,12.] equivalent depths
            ==> defines parameter: nEquivDepth ; integer, number of equivalent depths to do == len(Ahe)

    returns: tuple of size 2
        Afreq: Frequency, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn: Zonal savenumber, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        
    notes:
        The outputs contain both symmetric and antisymmetric waves. In the case of 
        nWaveType == 6:
        0,1,2 are (ASYMMETRIC) "MRG", "IG", "EIG" (mixed rossby gravity, inertial gravity, equatorial inertial gravity)
        3,4,5 are (SYMMETRIC) "Kelvin", "ER", "IG" (Kelvin, equatorial rossby, inertial gravity)
    """
    nEquivDepth = len(Ahe) # this was an input originally, but I don't know why.
    pi    = np.pi
    radius = 6.37122e06    # [m]   average radius of earth
    g     = 9.80665        # [m/s] gravity at 45 deg lat used by the WMO
    omega = 7.292e-05      # [1/s] earth's angular vel
    # U     = 0.0   # NOT USED, so Commented
    # Un    = 0.0   # since Un = U*T/L  # NOT USED, so Commented
    ll    = 2.*pi*radius*np.cos(np.abs(rlat))
    Beta  = 2.*omega*np.cos(np.abs(rlat))/radius
    fillval = 1e20
    
    # NOTE: original code used a variable called del,
    #       I just replace that with `dell` because `del` is a python keyword.

    # Initialize the output arrays
    Afreq = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))
    Apzwn = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))

    for ww in range(1, nWaveType+1):
        for ed, he in enumerate(Ahe):
            # this loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(Beta)*(g*he)**(0.25) This is close to pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed   
            L = np.sqrt(c/Beta)  # was: (g*he)**(0.25)/np.sqrt(Beta), this is Rossby radius of deformation        

            for wn in range(1, nPlanetaryWave+1):
                s  = -20.*(wn-1)*2./(nPlanetaryWave-1) + 20.
                k  = 2.0 * pi * s / ll
                kn = k * L 

                # Anti-symmetric curves  
                if (ww == 1):       # MRG wave
                    if (k < 0):
                        dell  = np.sqrt(1.0 + (4.0 * Beta)/(k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)
                    
                    if (k == 0):
                        deif = np.sqrt(c * Beta)
                    
                    if (k > 0):
                        deif = fillval
                    
                
                if (ww == 2):       # n=0 IG wave
                    if (k < 0):
                        deif = fillval
                    
                    if (k == 0):
                        deif = np.sqrt( c * Beta)
                    
                    if (k > 0):
                        dell  = np.sqrt(1.+(4.0*Beta)/(k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)
                    
                
                if (ww == 3):       # n=2 IG wave
                    n=2.
                    dell  = (Beta*c)
                    deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2)
                    # do some corrections to the above calculated frequency.......
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2 + g*he*Beta*k/deif)
                    
    
                # symmetric curves
                if (ww == 4):       # n=1 ER wave
                    n=1.
                    if (k < 0.0):
                        dell  = (Beta/c)*(2.*n+1.)
                        deif = -Beta*k/(k**2 + dell)
                    else:
                        deif = fillval
                    
                if (ww == 5):       # Kelvin wave
                    deif = k*c

                if (ww == 6):       # n=1 IG wave
                    n=1.
                    dell  = (Beta*c)
                    deif = np.sqrt((2. * n+1.) * dell + (g*he)*k**2)
                    # do some corrections to the above calculated frequency
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he)*k**2 + g*he*Beta*k/deif)
                
                eif  = deif  # + k*U since  U=0.0
                P    = 2.*pi/(eif*24.*60.*60.)  #  => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.
            
                Apzwn[ww-1,ed-1,wn-1] = s
                if (deif != fillval):
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would re-calculate now
                    Afreq[ww-1,ed-1,wn-1] = 1./P
                else:
                    Afreq[ww-1,ed-1,wn-1] = fillval
    return  Afreq, Apzwn