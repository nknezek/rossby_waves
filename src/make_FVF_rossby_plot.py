#! /usr/bin/env python
"""
Created on Tue Feb  9 18:07:32 2016

@author: nknezek
"""
import rossby_plotting as rplt
import pickle as pkl
#%%
data = pkl.load(open('FVF_rossby_plot_data.p','rb'))
rplt.plot2(data['model'], data['vecm1l1'], 1,1,data['vecm2l3'],2,3)
