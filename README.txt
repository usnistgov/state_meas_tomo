
Joint Quantum State and Measurement tomography
==============================================

This software package performs joint quantum state and measurement tomography 
as described in arXiv:17xx.xxxxx. Included are three example scripts that 
simulate data for one or two trapped ion systems with either symmetric or
asymmetric measurements: 

analysis_scripts/paper_simulations.py: produces all data and
    histograms shown in arXiv:17xx.xxxx with seed = 0. Also provides
    an example of symmetric measurements.
    
analysis_scripts/asym_simulations.py: produces simulated data from
    asymmetric measruements by similar methods as in previous script.
    
analysis_scripts/load_tutorial/load_simulations.py: gives an example
    of loading data with asymmetric measurements.


Features
--------
- Produces point estimates of unknown quantum states and POVMs
- Bounds expectation values of states even when POVMs are not informationally 
  complete
- Performs bootstrap resampling to establish confidence intervals
- Produces histogram plots and data shown in arXiv:17xx.xxxx


Installation
------------
Once the software is downloaded, update the following lines in each script so 
that installdir1 is the directory where the software is saved:

sys.path.append('installdir1/state_meas_tomo/')
sys.path.append('installdir1/state_meas_tomo/analysis_scripts')
sys.path.append('installdir1/state_meas_tomo/partial_tomography')

The software was written for Python 3.4. To create a Python 3.4 environment,
we recommend using Anaconda (https://www.anaconda.com/download/). Once 
installed, an environment can be created by running the following in the 
terminal:

For Mac/Linux:
$ conda create —-name py34 python=3.4
$ source activate

For Windows:
> conda create --name py34 python=3.4 matplotlib numpy scipy".
> activate py34

For more information and instructions see: 
https://conda.io/docs/user-guide/tasks/manage-environments.html

All three scripts mentioned above can be run from the terminal:

python xxxxx_simulations.py

from a Python Shell, or a developement environment like Spyder 
(included with Anaconda). 

Without expectation value bounding:
No further installation is required. (In the current version of each script,
the expecation value bounding is commented out.)

With expecation value bounding:
In order to use the expectation-value bounding features (described in Sec. V)
you additionally must have MATLAB installed and the included python engine 
extracted, see:
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

- To extract the engine, from within the py34 environment, run:

$ cd "matlabroot\extern\engines\python"
$ python setup.py install —prefix="installdir2"

where "matlabroot" is the location of the MATLAB installation (use function
matlabroot in MATLAB), and "installdir2" is the directory you've specified to 
to install the engine.

Depending on your operating system permissions, you may need to build
or install the MATLAB python engine in non-default locations.  For
instructions see:
https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html

You must also add installdir2 to the python path in each script by
updating the following lines:
                                                               
sys.path.append('installdir2/lib/python3.4/site-packages/')
sys.path.append('installdir2/lib/python3.4/site-packages/matlab')

- The software also requires YALMIP and SeDuMi, which must be
  installed in MATLAB.  Download and follow installation instructions
  from:

YALMIP: https://yalmip.github.io

SeDuMi: http://sedumi.ie.lehigh.edu
                                                             
- To run all three sample scripts with the expecation value bounding, first
uncomment line 27 in analysis.py:

# import matlab.engine  # uncomment for expectation-value bounding

and then uncomment the bottom lines of the chosen script.


Outputs
-------
Each PartialMaxLike instance (contained in analysis.py module) that uses the
.tomography or .bootstrapAnalysis methods creates a "autosave_name.hist" file,
where name is the attribuite given for the instance (defaults to the date), 
which contains the PartialMaxLike instance with all the histograms, analysis 
parameters, estimites, and bootstrap log (if created).

The expectationValues function in analysis_tools.py creates a text file called 
"results-name.txt" output, which contains the expectation value lower and upper 
bounds for each specified observable as well as the confidence intervals and 
medians from the bootstrap resamples. It also creates .out file with the list
of lower and upper bounds for each bootstrap resample.

The remaining functions in analysis_tools.py create figure 2 and 5 in
arXiv:17xx.xxxxx and other figures that may be useful. \SGc{Can we
make the example scripts automatically make these figures?}


Support
-------
If you are having issues, please let us know.

Contact: Scott Glancy (sglancy@nist.gov)
