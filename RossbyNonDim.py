import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import matplotlib as mpl
from numpy import sin
from numpy import cos


# Size of grid
Nk = 20
Nl = 20
Nm = 1

# Physical Parameters

R = 2890
Omega = np.pi/(24.0*3600.0) # rotation rate in rad/s
rho = 1e5  # density in kg/m^3

# Non-Dimensionalized Parameters

rmin = 0.99
rmax = 1.0


#ModelVariables = {'uk','ul','um','bk','bl','bm','p','c'};
ModelVariables = ('uk','ul','um','p')

# Size of Matricies
Sizeuk = (Nk+2)*Nl
Sizeul = Nk*Nl
Sizeum = Nk*Nl
Sizep = Nk*(Nl+2)
SizeM = Sizeuk+Sizeul+Sizeum+Sizep

# Create model parameter vectors
dr = (rmax-rmin)/(Nk+1)
r = np.linspace(rmin-dr/2,rmax+dr/2,num=Nk+2)
rm = np.linspace(rmin-dr,rmax,num=Nk+2)
rp = np.linspace(rmin,rmax+dr,num=Nk+2)
dt = np.pi/(Nl+1)
t = np.linspace(-dt/2,np.pi+dt/2,num=Nl)
tm = np.linspace(-dt,np.pi,num=Nl)
tp = np.linspace(0,np.pi+dt,num=Nl)


# print r
# print t

def getIndex(k,l,m,var):
    if (var in ModelVariables):
        if var == 'uk':
            if k>=0 and k<=Nk+1 and l>=1 and l<=Nl and m>=1 and m<=Nm:
                return k + (l-1)*(Nk+2) + (m-1)*SizeM
            else:
                return 'index out of bounds'
        elif var == 'ul':
            if k>=1 and k<=Nk and l>=0 and l<=Nl+1 and m>=1 and m<=Nm:
                return Sizeuk + k-1 + l*Nk + (m-1)*SizeM
            else:
                return 'index out of bounds'
        elif var == 'um':
            if k>=1 and k<=Nk and l>=1 and l<=Nl and m>=1 and m<=Nm:
                return Sizeuk + Sizeul + k-1 + (l-1)*Nk + (m-1)*SizeM
            else:
                return 'index out of bounds'
        elif var == 'p':
            if k>=1 and k<=Nk and l>=1 and l<=Nl and m>=1 and m<=Nm:
                return Sizeuk + Sizeul + Sizeum + k-1 + (l-1)*Nk + (m-1)*SizeM
            #if ((k>=0 and k<=Nk+1 and l>=1 and l<=Nl) or (k>=1 and k<=Nk and l>=0 and l<=Nl+1)) and (m>=1 and m<=Nm):
            #    if l==0:
            #        return Sizeuk + Sizeul + Sizeum + k-1 + (m-1)*SizeM
            #    elif l==Nl+1:
            #        return Sizeuk + Sizeul + Sizeum + Nk + (Nk+2)*Nl + (k-1) + (m-1)*SizeM
            #    else:
            #        return Sizeuk + Sizeul + Sizeum + Nk + (Nk+2)*(l-1) + k + (m-1)*SizeM
            else:
                return 'index out of bounds'
    else:
        return 'Not a valid variable name in this model'

class GovEquation():
    def __init__(self,variable):
        self.rows = []
        self.cols = []
        self.vals = []
        self.variable = variable

    def addTerm(self,var,value,kdiff=0,ldiff=0,mdiff=0):
        for m in range(1,Nm+1):
            for l in range(1,Nl+1):
                for k in range(1,Nk+1):
                    if var =='uk':
                        if (k+kdiff >= 0 and k+kdiff <=Nk+1) and (l+ldiff >= 1 and l+ldiff <=Nl):
                            self.rows.append(getIndex(k,l,m,self.variable))
                            self.cols.append(getIndex(k+kdiff,l+ldiff,m+mdiff,var))
                            self.vals.append(eval(value,globals(),locals()))
                        else:
                            return 'index out of bounds'
                    elif var =='ul':
                        if (k+kdiff >= 1 and k+kdiff <=Nk) and (l+ldiff >= 0 and l+ldiff <=Nl+1):
                            self.rows.append(getIndex(k,l,m,self.variable))
                            self.cols.append(getIndex(k+kdiff,l+ldiff,m+mdiff,var))
                            self.vals.append(eval(value,globals(),locals()))
                        else:
                            return 'index out of bounds'
                    elif var =='um':
                        if (k+kdiff >= 1 and k+kdiff <=Nk) and (l+ldiff >= 1 and l+ldiff <=Nl):
                            self.rows.append(getIndex(k,l,m,self.variable))
                            self.cols.append(getIndex(k+kdiff,l+ldiff,m+mdiff,var))
                            self.vals.append(eval(value,globals(),locals()))
                        else:
                            return 'index out of bounds'
                    elif var =='p':
                        if (k+kdiff >= 1 and k+kdiff <=Nk) and (l+ldiff >= 1 and l+ldiff <=Nl):
                            self.rows.append(getIndex(k,l,m,self.variable))
                            self.cols.append(getIndex(k+kdiff,l+ldiff,m+mdiff,var))
                            self.vals.append(eval(value,globals(),locals()))
                        elif k+kdiff == 0 and (l+ldiff >= 1 and l+ldiff <=Nl):
                            self.rows.append(getIndex(k,l,m,self.variable))
                            self.cols.append(getIndex(1,l+ldiff,m+mdiff,var))
                            self.vals.append(eval(value,globals(),locals()))
                        elif k+kdiff == Nk+1 and (l+ldiff >= 1 and l+ldiff <=Nl):
                            self.rows.append(getIndex(k,l,m,self.variable))
                            self.cols.append(getIndex(Nk,l+ldiff,m+mdiff,var))
                            self.vals.append(eval(value,globals(),locals()))
                        elif (k+kdiff >= 1 and k+kdiff <=Nk) and l+ldiff ==0:
                            self.rows.append(getIndex(k,l,m,self.variable))
                            self.cols.append(getIndex(k+kdiff,1,m+mdiff,var))
                            self.vals.append(eval(value,globals(),locals()))
                        elif (k+kdiff >= 1 and k+kdiff <=Nk) and l+ldiff ==Nl+1:
                            self.rows.append(getIndex(k,l,m,self.variable))
                            self.cols.append(getIndex(k+kdiff,Nl,m+mdiff,var))
                            self.vals.append(eval(value,globals(),locals()))
                        else:
                            return 'index out of bounds'

    def addBC(self,var,value,index):
        for m in range(1,Nm+1):
            if var == 'uk':
                for l in range(1,Nl+1):
                    self.rows.append(getIndex(0,l,m,self.variable))
                    self.cols.append(getIndex(0,l,m,var))
                    self.vals.append(eval(value,globals(),locals()))

                    self.rows.append(getIndex(0,l,m,self.variable))
                    self.cols.append(getIndex(index,l,m,var))
                    self.vals.append(eval(value,globals(),locals()))

                    self.rows.append(getIndex(0,l,m,self.variable))
                    self.cols.append(getIndex(Nk+1,l,m,var))
                    self.vals.append(eval(value,globals(),locals()))

                    self.rows.append(getIndex(0,l,m,self.variable))
                    self.cols.append(getIndex(Nk+1-index,l,m,var))
                    self.vals.append(eval(value,globals(),locals()))
            elif var == 'ul':
                for k in range(1,Nk+1):
                    self.rows.append(getIndex(k,0,m,self.variable))
                    self.cols.append(getIndex(k,0,m,var))
                    self.vals.append(eval(value,globals(),locals()))
                    self.rows.append(getIndex(k,0,m,self.variable))
                    self.cols.append(getIndex(k,index,m,var))
                    self.vals.append(eval(value,globals(),locals()))
                    self.rows.append(getIndex(k,Nl+1,m,self.variable))
                    self.cols.append(getIndex(k,Nl+1,m,var))
                    self.vals.append(eval(value,globals(),locals()))
                    self.rows.append(getIndex(k,Nl+1,m,self.variable))
                    self.cols.append(getIndex(k,Nl+1-index,m,var))
                    self.vals.append(eval(value,globals(),locals()))
            #elif var == 'p':
            #    for k in range(1,Nk+1):
            #        self.rows.append(getIndex(k,0,m,self.variable))
            #        self.cols.append(getIndex(k,0,m,var))
            #        self.vals.append(eval(value,globals(),locals()))
            #        self.rows.append(getIndex(k,0,m,self.variable))
            #        self.cols.append(getIndex(k,index,m,var))
            #        self.vals.append(eval(value,globals(),locals()))
            #        self.rows.append(getIndex(k,Nl+1,m,self.variable))
            #        self.cols.append(getIndex(k,Nl+1,m,var))
            #        self.vals.append(eval(value,globals(),locals()))
            #        self.rows.append(getIndex(k,Nl+1,m,self.variable))
            #        self.cols.append(getIndex(k,Nl+1-index,m,var))
            #        self.vals.append(eval(value,globals(),locals()))
            #    for l in range(1,Nl+1):
            #        self.rows.append(getIndex(0,l,m,self.variable))
            #        self.cols.append(getIndex(0,l,m,var))
            #        self.vals.append(eval(value,globals(),locals()))
            #        self.rows.append(getIndex(0,l,m,self.variable))
            #        self.cols.append(getIndex(index,l,m,var))
            #        self.vals.append(eval(value,globals(),locals()))
            #        self.rows.append(getIndex(0,l,m,self.variable))
            #        self.cols.append(getIndex(Nk+1,l,m,var))
            #        self.vals.append(eval(value,globals(),locals()))
            #        self.rows.append(getIndex(0,l,m,self.variable))
            #        self.cols.append(getIndex(Nk+1-index,l,m,var))
            #        self.vals.append(eval(value,globals(),locals()))

    def getCooMatrix(self):
        return sparse.coo_matrix((self.vals,(self.rows,self.cols)),shape=(SizeM*Nm,SizeM*Nm))

    def todense(self):
        return self.getCooMatrix().todense()

    def getCsrMatrix(self):
        return self.getCooMatrix().tocsr()


###### Right Hand Matrix ########
rmom = GovEquation('uk')
tmom = GovEquation('ul')
pmom = GovEquation('um')
div =GovEquation('p')

rmom.addTerm('p','(rt[k]**2.0-rb[k]**2.0)/2.0/P_star')
rmom.addTerm('p','rt[k]**2.0/2.0/P_star',kdiff=1)
rmom.addTerm('p','-rb[k]**2.0/2.0/P_star',kdiff=-1)
rmom.addTerm('um','2*r[k]**2.0*dr*sin(t[l])')
rmom.addBC('uk','1',1)

tmom.addTerm('p','-sin(tt[l])/2.0/P_star',kdiff=1)
tmom.addTerm('p','(sin(tb[l])-sin(tt[l]))/2.0/P_star',kdiff=0)
tmom.addTerm('p','sin(tb[l])/2.0/P_star',kdiff=-1)
tmom.addTerm('um','2*r[k]*sin(t[l])*dt*cos(t[l])')
tmom.addBC('ul','1',1)

pmom.addTerm('uk','-2*sin(t[l])')
pmom.addTerm('ul','2*cos(t[l])')
pmom.addTerm('p','-1j*m/(r[k]*sin(t[l]))/P_star')

div.addTerm('uk','rt[k]**2*sin(t[l])*dt/2',kdiff=1)
div.addTerm('uk','(rt[k]**2-rb[k]**2)*sin(t[l])*dt/2')
div.addTerm('uk','-rb[k]**2*sin(t[l])*dt/2',kdiff=-1)
div.addTerm('ul','r[k]*dr*sin(tt[l])/2',ldiff=1)
div.addTerm('ul','r[k]*dr*(sin(tt[l])-sin(tb[l]))/2')
div.addTerm('ul','-r[k]*dr*sin(tb[l])/2',ldiff=-1)
div.addTerm('um','1j*m*r[k]*dr*dt')

A = rmom.getCooMatrix() + tmom.getCooMatrix() + pmom.getCooMatrix() + div.getCooMatrix()


####### Left Hand Matrix ########
lrmom = GovEquation('uk')
ltmom = GovEquation('ul')
lpmom = GovEquation('um')
lrmom.addTerm('uk','1.0')
ltmom.addTerm('ul','1.')
lpmom.addTerm('um','1.')
M = lrmom.getCooMatrix() + ltmom.getCooMatrix() + lpmom.getCooMatrix()
# M = np.eye(M.shape[0])

### Save Matrix Pictures #####
plt.figure(1)
plt.spy(np.abs(A.todense()))
plt.grid()
plt.savefig('A_matrix.png')
plt.figure(2)
plt.spy(np.abs(M.todense()))
plt.grid()
plt.savefig('M_matrix.png')

##### Solve Eigenvalue Equation ######
# print A
# print M
np.savetxt('A_matrix.txt',A.todense())
np.savetxt('M_matrix.txt',M.todense())
tol = 1e-2
vals,vecs = LA.eigs(A,k=2,M=M,sigma=1.1,return_eigenvectors=True,tol=tol)
np.save('eigenvals',vals)
np.save('eigenvecs',vecs)
# vals,vecs = LA.eigs(A,k=2,return_eigenvectors=True)


##### Display Waves ######
plt.figure(3)
plt.grid()
plt.plot(vecs[0][0:Nk])
plt.savefig('testplot.png')



