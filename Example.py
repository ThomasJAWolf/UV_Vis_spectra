"""
This is a simple example code to demonstrate use of the UV_Vis_spectra class.
Created by Thomas Wolf, 01/03/21
"""
from UV_Vis_spectra import *

############################################################################################################
## Example code ############################################################################################
############################################################################################################

filename = '201230_2-bromothiophene.csv'
spectra = UV_Vis_spectra(filename)
print(len(spectra.Spectra))
spectra.plot([0,4],[200,300])
