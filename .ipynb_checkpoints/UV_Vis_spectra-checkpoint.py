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
                # This line accommodates situations where there spectra with different
                # numbers of points in the same file:
                if line[j] == '':
                    dat[i,j] = np.nan
                else:
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
        
    def eval_epsilon_xsec(self, specinds, backginds, cell_length, press_conc, press, T=293):
        """
        Function to convert internsity scales into decadic extinction coefficients and cross-sections
        Arguments:
        specinds:    List of indices of spectra to average over.
        backinds:    List of indices of spectra to subtract from spectra defined by specinds
        cell_length: Optical path length of the absorption cell_length [cm]
        press_conc:  Value for pressure [torr] or concentration [mol/l] of the sample
        press:       Boolean defining if the press_conc value refers to a pressure or not
        T:           Temperature of the sample [K]
        
        Returns:
        cross-section: Spectrum object involving beside the intensity [OD] also the decadic extinction 
                       coefficient epsilon [l/(mol*cm)] and the absorption cross-section xsect [cm^2]
        """
        if press==True:
            pPa = press_conc*133 # [Pa]
            R = 8.314 # J/(mol*K)]
            c = pPa/(R*T)*1E-3 # [mol/l]
        else:
            c = press_conc
        
        for i, ind in enumerate(specinds):
            spec = self.Spectra[ind]
            if i==0:
                int = np.zeros_like(spec.WL_nm)
            int += spec.Abs_OD
        int = int/len(specinds)
        
        if backginds == None:
            bkg = np.zeros_like(int)
        else:
            for i, ind in enumerate(backginds):
                spec = self.Spectra[ind]
                if i==0:
                    bkg = np.zeros_like(spec.WL_nm)
                bkg += spec.Abs_OD
            bkg = bkg/len(backginds)
        
        int = int - bkg
        
        epsilon = int/(c*cell_length) # [l/(mol*cm)]
        xsect = sigm = epsilon*np.log(10)*1000/6.022E23 # [cm^2]
        
        cross_sections = Spectrum(spec.WL_nm, int, self.Spectra[specinds[0]].name, epsilon, xsect)
        
        return cross_sections
            
        
        
class Spectrum(object):
    """
    Class to store wavelength and absorbance data of individual spectra in.
    Arguments:
    x: Wavelength in nm
    y: absorbance in OD
    """
    def __init__(self, x, y, name, eps=None, xsect=None ):
        self.name = name
        inds = np.where(~np.isnan(x))
        self.WL_nm = x[inds]
        self.Abs_OD = y[inds]
        if eps is not None:
           self.epsilon_l_mol_cm = eps
        if xsect is not None:
           self.xsect_cm2 = xsect
           
    def plot_xsect(self, xlim=None):
        """
        Function to plot absorption cross-sections.
        Arguments:
        xlim: x axis limits
        """
        plt.figure()
        plt.plot(self.WL_nm, self.xsect_cm2*1E18)
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Cross-section / Mbarn')
        if xlim!=None:
            plt.xlim(xlim)
        plt.title(self.name)
        plt.show()
