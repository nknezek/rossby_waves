'''
Created on Apr 23, 2015

@author: nknezek
'''
import logging
import os
from __builtin__ import enumerate


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)    
        
def setup_custom_logger(dir_name='./',filename='rossby.log'):
    ensure_dir(dir_name)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    fileHandler = logging.FileHandler(dir_name+filename)

    handler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(fileHandler)
    return logger

def log_model(logger,model):
    m = model.m_values[0]
    Nk = model.Nk
    Nl = model.Nl
    R = model.R
    Omega = model.Omega
    h = model.h
    rho = model.rho
    nu = model.nu
    dr = model.dr
    dth = model.dth
    t_star = model.t_star
    r_star = model.r_star
    u_star = model.u_star
    P_star = model.P_star
    E = model.E
    
    logger.info(
    '\nRossby model, m={0}, Nk={1}, Nl={2}\n'.format(m,Nk,Nl)
    +'\nPhysical Parameters\n'
    +'Omega = {0:.2e} rad/s\n'.format(Omega)
    +'R = {0} km\n'.format(R*1e-3)
    +'h = {0} km\n'.format(h*1e-3)
    +'rho = {0:.2e} kg/m^3\n'.format(rho)
    +'nu = {0:.2e} m/s^2\n'.format(nu)
    +'\nNon-Dimensional Parameters\n'
    +'t_star = {0:.2e} s\n'.format(t_star)
    +'r_star = {0:.2e} m\n'.format(r_star)
    +'u_star = {0:.2e} m/s\n'.format(u_star)
    +'P_star = {0:.2e} Pa = {1:.2f} GPa\n'.format(P_star,P_star*1e-9)
    +'E = {0:.2e}\n'.format(E)
    +'\nGrid Spacing Evaluation\n'
    +'dr^2/dth^2 = {0:.3e}\n'.format(dr**2/dth**2)
    +'E/dr^2 = {0:.1e}\n'.format(E/dr**2)
    +'E/dth^2 = {0:.1e}\n'.format(E/dth**2)
    )

def log_eigenvalues(logger,model,vals):
    found = '\nEigenvalues found :\n'
    for ind,val in enumerate(vals):
        found = found+'{0}: {1:.3e}\n'.format(ind,val)
    logger.info(found)
