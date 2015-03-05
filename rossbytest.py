import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import matplotlib as mpl
from numpy import sin
from numpy import cos
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Size of grid
K = 10
L = 40
m=1

# Physical Parameters

R = 2890
Omega = np.pi/(24.0*3600.0) # rotation rate in rad/s
rho = 1e5  # density in kg/m^3

# Non-Dimensionalized Parameters

rmin = 0.99
rmax = 1.0


#ModelVariables = {'uk','ul','um','bk','bl','bm','p','c'};
model_variables = ('uk','ul','um','p')

# Size of Matricies
Sizeuk = (K+2)*L
Sizeul = K*L
Sizeum = K*L
Sizep = (K+2)*L
SizeM = Sizeuk+Sizeul+Sizeum+Sizep

# Create model parameter vectors
dr = (rmax-rmin)/(K+1)
r = np.linspace(rmin-dr/2,rmax+dr/2,num=K+2)
rm = np.linspace(rmin-dr,rmax,num=K+2)
rp = np.linspace(rmin,rmax+dr,num=K+2)
dt = np.pi/(L+1)
t = np.linspace(-dt/2,np.pi+dt/2,num=L+2)
tm = np.linspace(dt,np.pi+dt,num=L+2)
tp = np.linspace(0,np.pi,num=L+2)


## Functions

def getIndex(k,l,m,var):
    if (var in model_variables):
        if var == 'uk':
            if k>=0 and k<=K+1 and l>=1 and l<=L:
                return k + (l-1)*(K+2)
            else:
                return 'uk index out of bounds'
        elif var == 'ul':
            if k>=1 and k<=K and l>=1 and l<=L:
                return Sizeuk + k-1 + (l-1)*K 
            else:
                return 'ul index out of bounds'
        elif var == 'um':
            if k>=1 and k<=K and l>=1 and l<=L:
                return Sizeuk + Sizeul + k-1 + (l-1)*K
            else:
                return 'um index out of bounds'
        elif var == 'p':
            if k>=0 and k<=K+1 and l>=1 and l<=L:
                return Sizeuk + Sizeul + Sizeum + k + (l-1)*(K+2)
            else:
                return 'p index out of bounds'
    else:
        return 'Not a valid variable name in this model'


class GovEquation():
    def __init__(self,variable):
        self.rows = []
        self.cols = []
        self.vals = []
        self.variable = variable

    def addTerm(self,k,l,m,var,value,kdiff=0,ldiff=0,mdiff=0):
        self.rows.append(getIndex(k,l,m,self.variable))
        self.cols.append(getIndex(k+kdiff,l+ldiff,m+mdiff,var))
        self.vals.append(value)

    def getCooMatrix(self):
        return sparse.coo_matrix((self.vals,(self.rows,self.cols)),shape=(SizeM,SizeM))

    def todense(self):
        return self.getCooMatrix().todense()

    def getCsrMatrix(self):
        return self.getCooMatrix().tocsr()


## Right Hand Side Equations (A Matrix in Eigenvalue Problem)
k_momentum = GovEquation('uk')
l_momentum = GovEquation('ul')
m_momentum = GovEquation('um')
mass_conservation = GovEquation('p')

# Add terms for all non-edge cells
for k in range(1,K+1):
    for l in range(2,L):
        k_momentum.addTerm(k,l,0,'p', (rm[k]/r[k])**2/(2*dr), kdiff=-1)
        k_momentum.addTerm(k,l,0,'p', (rp[k]/r[k])**2/(2*dr), kdiff=+1)
        k_momentum.addTerm(k,l,0,'p', (sin(tm[l])/sin(t[l]))/(4*r[k]), ldiff=-1)
        k_momentum.addTerm(k,l,0,'p', (sin(tp[l])/sin(t[l]))/(4*r[k]), ldiff=+1)
        k_momentum.addTerm(k,l,0,'p', (sin(tm[l])+sin(tp[l]))/(2*r[k]*sin(t[l]) )-1/r[k])

        l_momentum.addTerm(k,l,0,'p', (sin(tm[l])/sin(t[l]))/(2*r[k]*dt) , ldiff=-1)
        l_momentum.addTerm(k,l,0,'p', -(sin(tp[l])/sin(t[l]))/(2*r[k]*dt) , ldiff=+1)
        l_momentum.addTerm(k,l,0,'p', (sin(tm[l])-sin(tp[l]))/(2*dt) + 1 )
        l_momentum.addTerm(k,l,0,'um', 2*cos(t[l]) )

        m_momentum.addTerm(k,l,0,'ul', -2*cos(t[l]) )
#         m_momentum.addTerm(k,l,0,'uk', -2*sin(t[l]) )
        m_momentum.addTerm(k,l,0,'p', -1j*m/(r[k]*sin(t[l])) )
        
        mass_conservation.addTerm(k,l,0,'uk', +(rp[k]/r[k])**2/(2*dr) , kdiff = +1)
        mass_conservation.addTerm(k,l,0,'uk', -(rm[k]/r[k])**2/(2*dr) , kdiff = -1)
        mass_conservation.addTerm(k,l,0,'uk', (rp[k]**2-rm[k]**2)/r[k]**2/(2*dr) )
        
        mass_conservation.addTerm(k,l,0,'ul', +sin(tp[l])/sin(t[l])/(2*r[k]*dt) , ldiff = +1)
        mass_conservation.addTerm(k,l,0,'ul', -sin(tm[l])/sin(t[l])/(2*r[k]*dt) , ldiff = -1)
        mass_conservation.addTerm(k,l,0,'ul', (sin(tp[l]) - sin(t[l]))/(sin(t[l])*2*r[k]*dt) )
        
        mass_conservation.addTerm(k,l,0,'um', 1j*m/(r[k]*sin(t[l]) )  )
        
        
# Along top and bottom edges
for l in range(1,L+1):
    k=0
    k_momentum.addTerm(k,l,0,'p',-1,kdiff=+1)
    k_momentum.addTerm(k,l,0,'p',1)
    
    mass_conservation.addTerm(k,l,0,'uk',-1,kdiff=+1)
    mass_conservation.addTerm(k,l,0,'uk',1)
    
    k=K+1
    k_momentum.addTerm(k,l,0,'p',-1,kdiff=-1)
    k_momentum.addTerm(k,l,0,'p',1)
    
    mass_conservation.addTerm(k,l,0,'uk',-1,kdiff=-1)
    mass_conservation.addTerm(k,l,0,'uk',1)

# At poles
for k in range(1,K+1):
    l=1
    k_momentum.addTerm(k,l,0,'p', (rm[k]/r[k])**2/(2*dr), kdiff=-1)
    k_momentum.addTerm(k,l,0,'p', (rp[k]/r[k])**2/(2*dr), kdiff=+1)
    k_momentum.addTerm(k,l,0,'p', (sin(tp[l])/sin(t[l]))/(4*r[k]), ldiff=+1)
    k_momentum.addTerm(k,l,0,'p', (sin(tm[l])+sin(tp[l]))/(2*r[k]*sin(t[l]) )-1/r[k])

    l_momentum.addTerm(k,l,0,'p', -(sin(tp[l])/sin(t[l]))/(2*r[k]*dt) , ldiff=+1)
    l_momentum.addTerm(k,l,0,'p', (sin(tm[l])-sin(tp[l]))/(2*dt) + 1 )
    l_momentum.addTerm(k,l,0,'um', 2*cos(t[l]) )
    
    m_momentum.addTerm(k,l,0,'ul', -2*cos(t[l]) )
#     m_momentum.addTerm(k,l,0,'uk', -2*sin(t[l]) )
    m_momentum.addTerm(k,l,0,'p', -1j*m/(r[k]*sin(t[l])) )

    mass_conservation.addTerm(k,l,0,'uk', +(rp[k]/r[k])**2/(2*dr) , kdiff = +1)
    mass_conservation.addTerm(k,l,0,'uk', (rp[k]**2-rm[k]**2)/r[k]**2/(2*dr) )
    
    mass_conservation.addTerm(k,l,0,'ul', +sin(tp[l])/sin(t[l])/(2*r[k]*dt) , ldiff = +1)
    mass_conservation.addTerm(k,l,0,'ul', (sin(tp[l]) - sin(t[l]))/(sin(t[l])*2*r[k]*dt) )
    
    mass_conservation.addTerm(k,l,0,'um', 1j*m/(r[k]*sin(t[l]) )  )
    
    l=L
    k_momentum.addTerm(k,l,0,'p', (rm[k]/r[k])**2/(2*dr), kdiff=-1)
    k_momentum.addTerm(k,l,0,'p', (rp[k]/r[k])**2/(2*dr), kdiff=+1)
    k_momentum.addTerm(k,l,0,'p', (sin(tm[l])/sin(t[l]))/(4*r[k]), ldiff=-1)
    k_momentum.addTerm(k,l,0,'p', (sin(tm[l])+sin(tp[l]))/(2*r[k]*sin(t[l]) )-1/r[k])

    l_momentum.addTerm(k,l,0,'p', (sin(tm[l])/sin(t[l]))/(2*r[k]*dt) , ldiff=-1)
    l_momentum.addTerm(k,l,0,'p', (sin(tm[l])-sin(tp[l]))/(2*dt) + 1 )
    l_momentum.addTerm(k,l,0,'um', 2*cos(t[l]) )

    m_momentum.addTerm(k,l,0,'ul', -2*cos(t[l]) )
#     m_momentum.addTerm(k,l,0,'uk', -2*sin(t[l]) )
    m_momentum.addTerm(k,l,0,'p', -1j*m/(r[k]*sin(t[l])) )

    mass_conservation.addTerm(k,l,0,'uk', +(rp[k]/r[k])**2/(2*dr) , kdiff = +1)
    mass_conservation.addTerm(k,l,0,'uk', -(rm[k]/r[k])**2/(2*dr) , kdiff = -1)
    mass_conservation.addTerm(k,l,0,'uk', (rp[k]**2-rm[k]**2)/r[k]**2/(2*dr) )
    
    mass_conservation.addTerm(k,l,0,'ul', -sin(tm[l])/sin(t[l])/(2*r[k]*dt) , ldiff = -1)
    mass_conservation.addTerm(k,l,0,'ul', (sin(tp[l]) - sin(t[l]))/(sin(t[l])*2*r[k]*dt) )
    
    mass_conservation.addTerm(k,l,0,'um', 1j*m/(r[k]*sin(t[l]) )  )
    
RHS = k_momentum.getCooMatrix() + l_momentum.getCooMatrix() + m_momentum.getCooMatrix() + mass_conservation.getCooMatrix()

##Left Hand Side (M matrix in eigenvalue problem)
k_LHS = GovEquation('uk')
l_LHS = GovEquation('ul')
m_LHS = GovEquation('um')
for k in range(1,K+1):
    for l in range(1,L+1):
        k_LHS.addTerm(k,l,0,'uk',1j)
        l_LHS.addTerm(k,l,0,'ul',1j)
        m_LHS.addTerm(k,l,0,'um',1j)

LHS = k_LHS.getCooMatrix() + l_LHS.getCooMatrix() + m_LHS.getCooMatrix()

plt.figure(1)
plt.spy(RHS.todense())
plt.grid()
plt.savefig('RHS.png')
plt.figure(2)
plt.spy(LHS.todense())
plt.grid()
plt.savefig('LHS.png')

## Solve Eigenvalue Equation
tol = 1e-4
vals,vecs = LA.eigs(RHS,k=7,M=LHS,sigma=2,return_eigenvectors=True,tol=tol)
np.save('eigenvals',vals)
np.save('eigenvecs',vecs)

## Display Waves 

i=0
value = vals[i]
vector = vecs[:,i]

uk = np.reshape(vector[0:Sizeuk],(K+2,L))
ul = np.reshape(vector[Sizeuk:(Sizeuk+Sizeul)],(K,L))
um = np.reshape(vector[(Sizeuk+Sizeul):(Sizeuk+Sizeul+Sizeum)],(K,L))
p = np.reshape(vector[(Sizeuk+Sizeul+Sizeum):SizeM],(K+2,L))


fig = plt.figure(3)
ax = fig.add_subplot(1,1,1, projection='3d')
X,Y = np.meshgrid(r[1:-1],t[1:-1])
p = ax.plot_surface(X,Y,ul.T)
plt.show()
plt.savefig('testplot.png')