

##### Display and Save 1D Waves with comparison Spherical Harmonics for found Eigenvalues and vector field map ######
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from numpy import sin
from numpy import cos
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import matplotlib as mpl
import rossbymodel as rf

colors = ['b','g','r','c']

def plot_1D_rossby(model,vec,val,m,l,dir_name=None):
	if dir_name is None:
		dir_name='../output/'
	E = model.E
	Nk = model.Nk
	Nl = model.Nl
	th = model.th

	## Calculate Form of Analytical Solution (exact up to a complex constant):
	uth_a_withBC = 1j*m/sin(th)*sp.special.sph_harm(m,l,0.*np.ones_like(th),th)
	uth_a = uth_a_withBC[1:-1]
	uph_a = -np.diff(uth_a_withBC[0:-1]*sin(th[0:-1])/(m*1j))/np.diff(th[0:-1])

	## Fit analytical solution to numerical solution with constant complex term C
	(uth_n, _, _) = model.getVariable(vec,'uth')
	ind = np.where(max(abs(uth_a)) == abs(uth_a))[0]
	C = uth_n[0,ind]/uth_a[ind]

	## Plot Figures
	fig, axes = plt.subplots(3,1,figsize=(15,8), sharex=True, sharey=True)
	titles = ['Absolute Value','Real','Imaginary']
	for (ax,title,ind) in zip(axes,titles,range(len(axes))):
		ax.set_title(title)
		line = []
		for (var,color) in zip(model.model_variables,colors):
			 (out,bbound,tbound)=model.getVariable(vec,var)
			 if ind == 0:
				  line.append(ax.plot(th[1:-1]*180./np.pi,abs(out.T),color=color))
			 elif ind ==1:
				  ax.plot(th[1:-1]*180./np.pi,out.T.real,color=color)
			 elif ind==2:
				  ax.plot(th[1:-1]*180./np.pi,out.T.imag,color=color)
		if ind ==0:
			 line.append(ax.plot(th[1:-1]*180./np.pi,abs(C*uth_a),color='g',ls='--',marker='+',markevery=10))
			 line.append(ax.plot(th[1:-1]*180./np.pi,abs(C*uph_a),color='r',ls='--',marker='+',markevery=10))
			 labels = ['ur','uth','uph','p','uth analytical','uph analytical']
			 ax.legend([x[0] for x in line],labels,loc=0,ncol=3)
			 ax.grid()
		elif ind ==1:
			 ax.plot(th[1:-1]*180./np.pi,(C*uth_a).real,color='g',ls='--',marker='+',markevery=10)
			 ax.plot(th[1:-1]*180./np.pi,C*uph_a.real,color='r',ls='--',marker='+',markevery=10)
			 ax.grid()
		elif ind==2:
			 ax.plot(th[1:-1]*180./np.pi,(C*uth_a).imag,color='g',ls='--',marker='+',markevery=10)
			 ax.plot(th[1:-1]*180./np.pi,(C*uph_a).imag,color='r',ls='--',marker='+',markevery=10)
			 ax.grid()
	plt.suptitle('Eigenvalue = {0:.5f}, m={1}, l={2}\n Nk={3}, Nl={4}, E={5:.2e}, C=({6:.2e})'.format(val,m,l,Nk,Nl,E,C), size=14)
	plt.savefig(dir_name+'m={1}_l={2}_Eig{0:.2f}j_Nk={3}_Nl={4}_E={5:.2e}_C={6:.2e}.png'.format(val.imag,m,l,Nk,Nl,E,C))

def plot_mollyweide_rossby(model,vec,val,m,l):
	E = model.E
	Nk = model.Nk
	Nl = model.Nl
	th = model.th

	## Calculate vector field and contour field for plotting with basemap
	## Create full vector grid in theta and phi
	u_1D = model.getVariable(vec,'uph')[0][0]
	v_1D = model.getVariable(vec,'uth')[0][0]
	Nph = 2*Nl
	ph = np.linspace(-180.,180.-360./Nph,Nph)
	lon_grid, lat_grid = np.meshgrid(ph,th[1:-1]*180./np.pi-90.,)
	v = (np.exp(1j*m*lon_grid*np.pi/180.).T*v_1D).T
	u = (np.exp(1j*m*lon_grid*np.pi/180.).T*u_1D).T
	absu=u**2 + v**2
	Nvec = np.floor(Nl/20.)

	### Plot Mollweide Projection
	plt.figure(figsize=(10,10))
	## Set up map
	bmap = Basemap(projection='moll',lon_0=0.)
	bmap.drawparallels(np.arange(-90.,90.,15.))
	bmap.drawmeridians(np.arange(0.,360.,15.))
	## Convert Coordinates to those used by basemap to plot
	lon,lat = bmap(lon_grid,lat_grid)
	bmap.contourf(lon,lat,absu,15,cmap=plt.cm.Reds,alpha=0.5)
	bmap.quiver(lon[::Nvec,::Nvec],lat[::Nvec,::Nvec],u[::Nvec,::Nvec].real,v[::Nvec,::Nvec].real)
	plt.title('Mollweide Projection Vector Field for m={0}, l={1}'.format(m,l))
	plt.savefig('./output/m={1}/MollweideVectorField_m={1}_l={2}.png'.format(val.imag,m,l))

def plot_robinson_rossby(model,vec,val,m,l):
	E = model.physical_constants['E']
	Nk = model.Nk
	Nl = model.Nl
	th = model.th

	## Plot Robinson vector field
	#### Display waves on a Spherical Map Projection
	projtype = 'robin'

	## Create full vector grid in theta and phi
	u_1D = model.getVariable(vec,'uph')[0][0]
	v_1D = model.getVariable(vec,'uth')[0][0]
	Nph = 2*Nl
	ph = np.linspace(-180.,180.-360./Nph,Nph)
	lon_grid, lat_grid = np.meshgrid(ph,th[1:-1]*180./np.pi-90.,)
	v = (np.exp(1j*m*lon_grid*np.pi/180.).T*v_1D).T
	u = (np.exp(1j*m*lon_grid*np.pi/180.).T*u_1D).T
	absu=u**2 + v**2
	Nvec = np.floor(Nl/20.)

	### Plot Robinson Projection
	plt.figure(figsize=(10,10))
	## Set up map
	bmap = Basemap(projection=projtype,lon_0=0.)
	bmap.drawparallels(np.arange(-90.,90.,15.))
	bmap.drawmeridians(np.arange(0.,360.,15.))
	## Convert Coordinates to those used by basemap to plot
	lon,lat = bmap(lon_grid,lat_grid)
	bmap.contourf(lon,lat,absu,15,cmap=plt.cm.Reds,alpha=0.5)
	bmap.quiver(lon[::Nvec,::Nvec],lat[::Nvec,::Nvec],u[::Nvec,::Nvec].real,v[::Nvec,::Nvec].real)
	plt.title('{0} Projection Vector Field for m={1}, l={2}'.format('Robinson',m,l))
	plt.savefig('./output/m={1}/{0}VectorField_m={1}_l={2}.png'.format('Robinson',m,l))

def plot_matrix(Mat,dir_name=None,title=None):
	"""Plots and saves a dense matrix"""
	if dir_name is None:
		dir_name='./'
	if title is None:
		title = 'Matrix'
	plt.figure(figsize=(10,10))
	plt.spy(np.abs(Mat))
	plt.grid()
	plt.title(title)
	plt.savefig(dir_name+title+'.png')

def animate_robinson_rossby(model,vec,val,m,l):
	E = model.E
	Nk = model.Nk
	Nl = model.Nl
	th = model.th

	##  Robinson vector field Animation
	#### Display waves on a Spherical Map Projection
	import matplotlib.animation as animation
	mpl.use("Agg")

	# Set up formatting for the movie files
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

	projtype = 'robin'

	## Create full vector grid in theta and phi
	u_1D = model.getVariable(vec,'uph')[0][0]
	v_1D = model.getVariable(vec,'uth')[0][0]
	Nph = 2*Nl

	ph = np.linspace(-180.,180.-360./Nph,Nph)
	lon_grid, lat_grid = np.meshgrid(ph,th[1:-1]*180./np.pi-90.,)
	v = (np.exp(1j*m*lon_grid*np.pi/180.).T*v_1D).T
	u = (np.exp(1j*m*lon_grid*np.pi/180.).T*u_1D).T
	absu=u**2 + v**2
	Nvec = np.floor(Nl/20.)

	### Plot Robinson Projection
	## Set up map
	fig = plt.figure(figsize=(10,10))
	bmap = Basemap(projection=projtype,lon_0=0.)


	## Convert Coordinates to those used by basemap to plot
	lon,lat = bmap(lon_grid,lat_grid)

	bmap.drawparallels(np.arange(-90.,90.,15.))
	bmap.drawmeridians(np.arange(0.,360.,15.))
	bmap.contourf(lon,lat,absu,15,cmap=plt.cm.Reds,alpha=0.5)
	bmap.quiver(lon[::Nvec,::Nvec],lat[::Nvec,::Nvec],u[::Nvec,::Nvec].real,v[::Nvec,::Nvec].real)
	plt.title('{0} Projection Vector Field for m={1}, l={2}'.format('Robinson',m,l))

	## Animation Functions
	Nt = 15
	dtime = 2*m*np.pi/Nt
	time=0

	def updatefig(u,v,time,dtime,val,lon,lat):
		## Update values
		u = u*np.exp(val*dtime)
		v = v*np.exp(val*dtime)
		absu = u**2 + v**2
		time = time+dtime

		## Clear figure and re-plot
		fig.clf()
		bmap.drawparallels(np.arange(-90.,90.,15.))
		bmap.drawmeridians(np.arange(0.,360.,15.))
		bmap.contourf(lon,lat,absu,15,cmap=plt.cm.Reds,alpha=0.5)
		bmap.quiver(lon[::Nvec,::Nvec],lat[::Nvec,::Nvec],u[::Nvec,::Nvec].real,v[::Nvec,::Nvec].real)
		plt.title('{0} Projection Vector Field for m={1}, l={2}, t={3}'.format('Robinson',m,l,time))

	im_ani = animation.FuncAnimation(fig, updatefig(u,v,time,dtime,val,lon,lat), interval=15, blit=True)

	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, metadata=dict(artist='Nick Knezek'), bitrate=1800)
	im_ani.save('./output/m={1}/{0}VectorFieldAnimation_m={1}_l={2}.mp4'.format('Robinson',m,l), writer=writer)

def plot2(model, vec1, m1, l1, vec2, m2, l2):
    title1 = 'm={0}, l={1}'.format(m1, l1)
    title2 = 'm={0}, l={1}'.format(m2, l2)
    dir_name='../output/'
    E = model.E
    Nk = model.Nk
    Nl = model.Nl
    th = model.th
    thdeg = th[1:-1]*180./np.pi

    ## Solution 1
    uth1_a_withBC = 1j*m1/sin(th)*sp.special.sph_harm(m1,l1,0.*np.ones_like(th),th)
    uth1_a = uth1_a_withBC[1:-1]
    uph1_a = -np.diff(uth1_a_withBC[0:-1]*sin(th[0:-1])/(m1*1j))/np.diff(th[0:-1])
    uth1, _, _ = model.getVariable(vec1,'uth')
    uph1,_,_ = model.getVariable(vec1, 'uph')
    ind1 = np.where(max(abs(uth1_a)) == abs(uth1_a))[0][0]
    C1 = uth1[0,ind1]/uth1_a[ind1]

    ##Solution 2
    uth2_a_withBC = 1j*m2/sin(th)*sp.special.sph_harm(m2,l2,0.*np.ones_like(th),th)
    uth2_a = uth2_a_withBC[1:-1]
    uph2_a = -np.diff(uth2_a_withBC[0:-1]*sin(th[0:-1])/(m2*1j))/np.diff(th[0:-1])
    uth2, _, _ = model.getVariable(vec2,'uth')
    uph2,_,_ = model.getVariable(vec2, 'uph')
    ind2 = np.where(max(abs(uth2_a)) == abs(uth2_a))[0][0]
    C2 = uth2[0,ind2]/uth2_a[ind2]

    ## Plot Figures
    fig, axes = plt.subplots(1,2,figsize=(7,5), sharex=True, sharey=True)
#    plt.subplot(121)
    axes[0].plot(uth1[0,:].real, thdeg, 'r-')
    axes[0].plot(uph1[0,:].real,thdeg, 'b--')
#    axes[0].grid()
    axes[0].plot(((C1*uth1_a).real)[::3], thdeg[::3], 'rx',markevery=10, markersize=10)
    axes[0].plot(((C1*uph1_a).real)[::3], thdeg[::3], 'b+',markevery=10, markersize=10)
    axes[0].set_title(title1, fontsize=14)
    axes[0].set_xticklabels([str(x) for x in [-1.0, -0.66, -0.33, 0, 0.33, 0.66, 1.0]], fontsize=10)
    axes[0].set_yticklabels([180, 160, 140, 120, 100, 80, 60, 40, 20, 0], fontsize=10)
    axes[0].set_xlabel('amplitude', fontsize=14)
    axes[0].set_ylabel('colatitude (degrees)', fontsize=14)
    axes[0].plot([0, 0], [0,180],'k:')
    xticks = axes[0].get_xticks()
    axes[0].plot([xticks[0],-xticks[0]], [90,90],'k:')
    axes[0].legend(['$u_{\\theta}^{n}$','$u_{\phi}^{n}$', '$u_{\\theta}^{a}$','$u_{\phi}^{a}$'], loc='lower right', fontsize=14)

#    plt.subplot(122)
    axes[1].plot(uth2[0,:].real, thdeg, 'r-')
    axes[1].plot(uph2[0,:].real,thdeg, 'b--')
#    axes[1].grid()
    axes[1].plot((C2*uth2_a).real, thdeg, 'rx',markevery=10, markersize=10)
    axes[1].plot((C2*uph2_a).real, thdeg, 'b+',markevery=10, markersize=10)
    axes[1].set_title(title2, fontsize=12)
    axes[1].set_xticklabels([str(x) for x in [-1.0, -0.66, -0.33, 0, 0.33, 0.66, 1.0]], fontsize=10)
    axes[1].set_yticklabels([180, 160, 140, 120, 100, 80, 60, 40, 20, 0], fontsize=10)
    axes[1].set_xlabel('amplitude', fontsize=14)
    axes[1].plot([0, 0], [0,180],'k:')
    xticks = axes[1].get_xticks()
    axes[1].plot([xticks[0],xticks[-1]], [90,90],'k:')
    plt.savefig('FVFrossbyfig2.pdf')

