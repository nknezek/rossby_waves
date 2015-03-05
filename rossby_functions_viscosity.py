import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as LA
from numpy import sin
from numpy import cos

class Model():
	def __init__(self,model_variables,physical_constants,model_parameters):
		self.model_variables = model_variables
		self.physical_constants = physical_constants
		self.Nk = model_parameters['Nk']
		self.Nl = model_parameters['Nl']
		self.SizeM = model_parameters['SizeM']
		self.dr = model_parameters['dr']
		self.r = model_parameters['r']
		self.rp = model_parameters['rp']
		self.rm = model_parameters['rm']
		self.dt = model_parameters['dt']
		self.t = model_parameters['t']
		self.tm = model_parameters['tm']
		self.tp = model_parameters['tp']
		self.Sizeuk = self.Nk*self.Nl
		self.Sizeul = self.Nk*self.Nl
		self.Sizeum = self.Nk*self.Nl
		self.Sizep = self.Nk*self.Nl
		self.SizeukBC = 2*self.Nl
		self.SizeulBC = 2*self.Nl
		self.SizeumBC = 2*self.Nl
		self.SizepBC = 2*self.Nl
		self.SizeMnoBC = self.Sizeuk+self.Sizeul+self.Sizeum+self.Sizep
		self.SizeMwithBC = self.SizeMnoBC + self.SizeukBC + self.SizeulBC + self.SizeumBC + self.SizepBC
		self.m_values = model_parameters['m_values']
		self.l_values = model_parameters['l_values']

	def getIndex(self,k,l,var):
		'''
		Takes coordinates for a point, gives back index in matrix.
		:param k: k grid value from 1 to K (or 0 to K+1 for uk or p)
		:param l: l grid value from 1 to L
		:param var: 'uk', 'ul', 'um', or 'p'
		:return: index of location in matrix
		'''
		Nk = self.Nk
		Nl = self.Nl
		SizeM = self.SizeM
		SizeMnoBC = self.SizeMnoBC
		Sizeuk = self.Sizeuk
		Sizeul = self.Sizeul
		Sizeum = self.Sizeum
		SizeulBC = self.SizeulBC
		SizeumBC = self.SizeumBC
		SizeukBC = self.SizeukBC

		if (var in self.model_variables):
			if var == 'uk':
				if l>=1 and l<=self.Nl:
					# Check for boundary point request, assign at end of matrix
					if k==0:
						return	SizeMnoBC + l-1
					elif k==Nk+1:
						return SizeMnoBC + Nl + l-1
					elif k>=1 and k<=Nk:
						return (k-1) + (l-1)*Nk
					else:
						return 'k index out of bounds'
				else:
					return 'l index out of bounds'

			elif var == 'ul':
				if l>=1 and l<=Nl:
					if k == 0:
						return SizeMnoBC + SizeukBC + l-1
					elif k == Nk+1:
						return SizeMnoBC + SizeukBC + Nl + l-1
					elif k>=1 and k<=Nk:
						return Sizeuk + k-1 + (l-1)*Nk
					else:
						return 'k index out of bounds'
				else:
					return 'l index out of bounds'

			elif var == 'um':
				if l>=1 and l<=Nl:
					if k == 0:
						return SizeMnoBC + SizeukBC + SizeulBC + l-1
					elif k == Nk+1:
						return SizeMnoBC + SizeukBC + SizeulBC + Nl + l-1
					elif k>=1 and k<=Nk:
						return Sizeuk + Sizeul + k-1 + (l-1)*Nk
					else:
						return 'k index out of bounds'
				else:
					return 'l index out of bounds'

			elif var == 'p':
				if l>=1 and l<=Nl:
					# Check for boundary point request, assign at end of matrix
					if k==0:
						return	SizeMnoBC + SizeukBC + SizeulBC + SizeumBC + l-1
					elif k==Nk+1:
						return SizeMnoBC + SizeukBC + SizeulBC + SizeumBC + Nl + l-1
					elif k>=1 and k<=Nk:
						return	Sizeuk + Sizeul + Sizeum + k-1 + (l-1)*Nk
					else:
						return 'k index out of bounds'
				else:
					return 'l or m index out of bounds'
		else:
			return 'Not a valid variable name in this model'

	def getVariable(self,vector,var):
		Nk = self.Nk
		Nl = self.Nl
		SizeMwithBC = self.SizeM
		SizeMnoBC = self.SizeMnoBC
		Sizeuk = self.Sizeuk
		Sizeul = self.Sizeul
		Sizeum = self.Sizeum
		SizeukBC = self.SizeukBC
		SizeulBC = self.SizeulBC
		SizeumBC = self.SizeumBC

		if (var in self.model_variables):
			if var == 'uk':
				return np.reshape(vector[0:Sizeuk],(Nk,Nl),'F')
			elif var == 'ul':
				return np.reshape(vector[Sizeuk:Sizeuk+Sizeul],(Nk,Nl),'F')
			elif var == 'um':
				return np.reshape(vector[Sizeuk+Sizeul:Sizeuk+Sizeul+Sizeum],(Nk,Nl),'F')
			elif var == 'p':
				return np.reshape(vector[Sizeuk+Sizeul+Sizeum:SizeMnoBC],(Nk,Nl),'F')
		elif var == 'ukBC':
			return np.reshape(vector[SizeMnoBC:SizeMnoBC+Nl*2],(2,Nl),'C')
		elif var == 'ulBC':
			return np.reshape(vector[SizeMnoBC+SizeukBC:SizeMnoBC+SizeukBC+SizeulBC],(2,Nl),'C')
		elif var == 'umBC':
			return np.reshape(vector[SizeMnoBC+SizeukBC+SizeulBC:SizeMnoBC+SizeukBC+SizeulBC+SizeumBC],(2,Nl),'C')
		elif var == 'pBC':
			return np.reshape(vector[SizeMnoBC+SizeukBC+SizeulBC+SizeumBC:SizeMwithBC],(2,Nl),'C')
		else:
			return 'Not a valid variable name in this model'

	def create_vector(self,uk,ul,um,p,ukBC,ulBC,umBC,pBC):
		Nk = self.Nk
		Nl = self.Nl
		return np.array((np.concatenate((np.reshape(uk,(Nk*Nl,1)),np.reshape(ul,(Nk*Nl,1)),np.reshape(um,(Nk*Nl,1)),np.reshape(p,(Nk*Nl,1)),
								np.reshape(ukBC,(2*Nl,1)), np.reshape(ulBC,(2*Nl,1)), np.reshape(umBC,(2*Nl,1)), np.reshape(pBC,(2*Nl,1))), axis=0)).T)

	def add_gov_equation(self,name,variable):
		setattr(self,name,GovEquation(self,variable))

class GovEquation():
	def __init__(self,model,variable):
		self.rows = []
		self.cols = []
		self.vals = []
		self.variable = variable
		self.model = model

	def addTerm(self,var,value,m,kdiff=0,ldiff=0,mdiff=0,first_l=1,last_l=None,first_k=1,last_k=None):
		dr = self.model.dr
		r = self.model.r
		rp = self.model.rp
		rm = self.model.rm
		dt = self.model.dt
		t = self.model.t
		tm = self.model.tm
		tp = self.model.tp
		Nk = self.model.Nk
		Nl = self.model.Nl
		E = self.model.physical_constants['E']

		if last_l == None:
			last_l = Nl
		if last_k == None:
			last_k = Nk

		for l in range(first_l,last_l+1):
			for k in range(first_k,last_k+1):
				value_computed = eval(value,globals(),locals())
				if not np.isclose(value_computed,0.0,atol=1E-12,rtol=1E-12):
					if var =='uk':
						if (k >= 1 and k <=Nk) and (l >= 1 and l <=Nl):
							self.rows.append(self.model.getIndex(k,l,self.variable))
							self.cols.append(self.model.getIndex(k+kdiff,l+ldiff,var))
							self.vals.append(value_computed)
						else:
							return 'index out of bounds'
					elif var =='ul':
						if (k >= 1 and k <=Nk) and (l >= 1 and l <=Nl):
							self.rows.append(self.model.getIndex(k,l,self.variable))
							self.cols.append(self.model.getIndex(k+kdiff,l+ldiff,var))
							self.vals.append(value_computed)
						else:
							return 'index out of bounds'
					elif var =='um':
						if (k >= 1 and k <=Nk) and (l >= 1 and l <=Nl):
							self.rows.append(self.model.getIndex(k,l,self.variable))
							self.cols.append(self.model.getIndex(k+kdiff,l+ldiff,var))
							self.vals.append(value_computed)
						else:
							return 'index out of bounds'
					elif var =='p':
						if (k >= 1 and k <=Nk) and (l >= 1 and l <=Nl):
							self.rows.append(self.model.getIndex(k,l,self.variable))
							self.cols.append(self.model.getIndex(k+kdiff,l+ldiff,var))
							self.vals.append(value_computed)
						else:
							return 'index out of bounds'

	def addBC(self,var,value,k,m,kdiff=0,first_l=1,last_l=None):
		Nl = self.model.Nl
		if last_l ==None:
			last_l = Nl
		if var in self.model.model_variables:
			for l in range(first_l,last_l+1):
				self.rows.append(self.model.getIndex(k,l,var))
				self.cols.append(self.model.getIndex(k+kdiff,l,var))
				self.vals.append(eval(value,globals(),locals()))
		else:
			return 'not a valid variable name in this model'

	def addValue(self,value,row,col):
		'''
		Adds a term to a specific index.
		value: term to add
		row: dictionary containing 'k','l',and 'var'
		col: dictionary containing 'k','l',and 'var'
		'''
		self.rows.append(self.model.getIndex(row['k'],row['l'],row['var']))
		self.cols.append(self.model.getIndex(col['k'],col['l'],col['var']))
		self.vals.append(eval(value,globals(),locals()))

	def getCooMatrix(self):
		return sparse.coo_matrix((self.vals,(self.rows,self.cols)),shape=(self.model.SizeM,self.model.SizeM))

	def todense(self):
		return self.getCooMatrix().todense()

	def getCsrMatrix(self):
		return self.getCooMatrix().tocsr()