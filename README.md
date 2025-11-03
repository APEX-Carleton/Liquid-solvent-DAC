# Liquid-solvent-DAC

Source code for "Liquid solvent Direct Air Capture's cost and carbon dioxide removal vary with ambient environmental conditions".

Manuscript DOI: https://doi.org/10.1038/s43247-024-01773-1

Description of files:
***************************
get_captureMatrix.py

This script expects 2 arguments, i.e. "python3 get_captureMatrix.py arg1 arg2"

The first argument is a .npz file containing arrays of climate variables, labelled as 'airTemperatureRange' (Kelvins), 'relativeHumidityRange' (between 0 and 1), 'CO2ppmRange' (in parts per million), and 'pressureRange' (in Pascals).

The second argument is the name of the output .npz file.
***************************
myfunctions.py

This wonderfully named file is a function library comprising the process model. It should be in the same folder as 'get_captureMatrix.py' 


