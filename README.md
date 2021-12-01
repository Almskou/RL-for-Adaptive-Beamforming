# RL-for-Adaptive-Beamforming
Master thesis about using reinforcement learning for adaptive beamforming

This project uses [Quadriga](https://quadriga-channel-model.de/) to simulate the environment. To run Quadriga either Octave or MATLAB is needed and as this project will mainly be done in Python an API is also needed. See the following installation guides for the different API's.
<br/><br/><br/>

## Installation guide Octave:
This project will be mainly done in Python 3.8.5, therefore is has been chosen to use **[oct2py](https://pypi.org/project/oct2py/)** version 5.2.0.
<br/>
### 1. Install oct2py
**oct2py** can be installed using pip
> pip install oct2py

or

> conda install -c conda-forge oct2py

<br/>

### 2. Install octave
Is has then been chosen to use Octave-6.4.0. 

Octave 6.4.0 can be install from:
https://www.gnu.org/software/octave/download
<br/><br/>
### 3. Add **octave-cli** to the variable environment
To use **oct2py** it is needed to add **octave-cli** to the variable environment. This can be done by following **oct2py**'s installation guide: <br/>
https://oct2py.readthedocs.io/en/latest/source/installation.html

<br/>

## Installation guide MATLAB:
As this project uses Python 3.8.5 MATLAB 2020b or a newer version is needed. 

See https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf. 
<br/><br/>
### 1. Install MATLAB
MATLAB can downloaded from: https://se.mathworks.com/products/matlab.html.
<br/><br/>
### 2. Install MATLAB Engine
To install the API follow MathWorks' setup guide:<br/>
https://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
