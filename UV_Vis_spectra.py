"""
This is some simple code to read in spectra exported as .csv files by the software of Varian Cary 
UV-Vis spectrometers.
Created by Thomas Wolf, 01/02/2021
"""

import numpy as np
import matplotlib.pyplot as plt

############################################################################################################
## Classes and functions ###################################################################################
############################################################################################################

class UV_Vis_spectra(object):
    """
    Class to extract spectra from .csv exports of spectra from Varian Cary UV-Vis spectrometers.
    Arguments:
    fname: Name of the .csv file containing the spectra
    """
    def __init__(self,fname):
        with open(fname) as f:
           lines = f.read().splitlines()
       
        header, data, self.info = self.split_file(lines)
        self.make_spectra(header, data)

    def split_file(self,lines):
        """
        Function to split the raw file into the header, the spectral data, and the measurement info.
        Arguments:
        lines: raw lines from the .csv file
        Returns:
        header: header describing the columns
        data: spectral data as ascii lines
        info: measurement info as ascii lines
        """
        header0 = lines[0].split(',')#[:-1]
        header1 = lines[1].split(',')#[:-1]
        header = [header0, header1]
        data = []
        print(lines[1203])
        for i in np.arange(2,len(lines)):
            if lines[i] == '':
                info = lines[i+1:]
                break
            data.append(lines[i])
        return header, data, info
    
    def make_spectra(self, header, data):
        """
        Function to convert spectral data and headers into Spectrum objects
        Arguments:
        header: Spectrum headers
        data: spectral data as ascii lines
        """
        dat = np.zeros((len(data),len(header[0])))
        for i in np.arange(len(data)):
           line = data[i].split(',')
           for j in np.arange(len(line)-1):
               dat[i,j] = float(line[j])

        self.Spectra = []
        for i in np.arange(0,len(header[0])-1,2):
           spec = Spectrum(dat[:,i], dat[:,i+1], header[0][i])
           self.Spectra.append(spec)
            
    def plot(self, arr, xlim=None):
        """
        Function to plot spectra.
        Arguments:
        arr: list with indices of spectra in the list to be plotted
        xlim: x axis limits
        """
        plt.figure()
        plt.title('UV-Vis absorption spectra')
        for item in arr:
            spec = self.Spectra[item]
            plt.plot(spec.WL_nm, spec.Abs_OD, label=spec.name)
        if xlim != None:
            plt.xlim(xlim)
        plt.legend(loc='best')
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Absorbance / OD')
        plt.show()

class Spectrum(object):
    """
    Class to store wavelength and absorbance data of individual spectra in.
    Arguments:
    x: Wavelength in nm
    y: absorbance in OD
    """
    def __init__(self, x, y,name):
        self.name = name
        self.WL_nm = x
        self.Abs_OD = y
