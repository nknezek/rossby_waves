# -*- coding: utf-8 -*-
"""
Created on May 5, 2015

@author: nknezek
"""

import numpy as np
import rossbymodel as rossby
import rossby_plotting as rplt
import rossbyloglib as rlog
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as LA
from numpy import sin
from numpy import cos
from datetime import datetime
import sys
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import matplotlib.pyplot as plt


# =============================================================================
# %% Define Parameters
# =============================================================================

# modes to simulate
m = 1
m_min = m
m_max = m
m_values = range(m_min, m_max + 1)

# Size of grid
Nk = 1
Nl = 251

model_variables = ('ur', 'uth', 'uph', 'p')
boundary_variables = ('ur', 'uth', 'uph', 'p')

# Physical Constants
R = 3480.  # Outer core radius in km
Omega = np.pi/(24.0*3600.0)  # rotation rate in rad/s
rho = 1.e4  # density in kg/m^3
h = 100.  # layer thickness in km
nu = 1e-1  # kinematic viscosity in m^2/s (5e4 max for Rossby)

physical_constants = {'R': R, 'Omega': Omega, 'rho': rho, 'h': h, 'nu': nu}
model_parameters = {'Nk': Nk, 'Nl': Nl, 'm_values': m_values}


# =============================================================================
# %% Create Model
# =============================================================================
model = rossby.Model(model_variables, boundary_variables, physical_constants,
                     model_parameters)
model.make_A()
model.make_M()

# =============================================================================
# %% Solve Using scipy Sparse Solver
# =============================================================================
vecs = {}
vals = {}
starting_sigma = 0+1.1j
which = 'LI'
num_eigs_to_find = 12
tol = 0.
vals, vecs = LA.eigs(model.A, k=num_eigs_to_find, M=model.M,
                     sigma=starting_sigma, return_eigenvectors=True, tol=tol,
                     which=which)
#%%
l=1
n=0
rplt.plot_1D_rossby(model, vecs[:,n], vals[n], m, l)

# =============================================================================
# %% Set up SLEPc
# =============================================================================

# Convert Matrices to SLEPc
model.make_A_SLEPc()
model.make_M_SLEPc(epsilon=1e-7)

Aeq = np.allclose(model.A.todense(),
                  model.A_SLEPc.getValues(range(model.SizeM),
                                          range(model.SizeM)))
Meq = np.allclose(model.M.todense(),
                  model.M_SLEPc.getValues(range(model.SizeM),
                                          range(model.SizeM)))
print 'Is A close? '+str(Aeq)
print 'Is M close? '+str(Meq)

# %% Solve with SLEPc Solver
EPS = SLEPc.EPS().create()
EPS.setDimensions(10, PETSc.DECIDE)
EPS.setOperators(model.A_SLEPc, model.M_SLEPc)
EPS.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
EPS.setTarget(0.67j)
EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE)
EPS.setFromOptions()
ST = EPS.getST()
ST.setType(SLEPc.ST.Type.SINVERT)

EPS.solve()

conv = EPS.getConverged()
vs, ws = PETSc.Mat.getVecs(model.A_SLEPc)
vals_SLEPc = np.empty((10), dtype='complex')
vecs_SLEPc = np.empty((model.SizeM, 10), dtype='complex')

for ind in range(conv):
    vals_SLEPc[ind] = EPS.getEigenpair(ind, ws)
    vecs_SLEPc[:, ind] = ws.getArray()
    print 'v_SLEPc_{1} = {0:.3e}'.format(vals_SLEPc[ind], ind)

f_vals_SLEPc, f_vecs_SLEPc = model.filter_eigenvalues(vals_SLEPc,
                                                      vecs_SLEPc.T.tolist(),
                                                      rtol=1e-2, atol=1e-2)
print vals[0] - vals_SLEPc[0]
print vals_SLEPc[0] - 2.j/3.
# =============================================================================
# %% Plot 1D Plot of Waves
# =============================================================================
for l in f_vals_SLEPc.iterkeys():
    rplt.plot_1D_rossby(model, f_vecs_SLEPc[l], f_vals_SLEPc[l], m, l)
