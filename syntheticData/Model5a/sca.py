# Implementation of various effective medium estimates
# This is a collection of Python functions to calculate the 
# effective properties of multi-phase elastic media
# The functions have been tested in Python 2.7 and may need some
# modifications to work with other Python implementations
# 
# Michele Paulatto - Imperial College London - January 2018
# Code first written during a PhD studentship at the University of Southampton
# in 2010. Ported to Python and modified at Imperial College London and during
# a visit to the University of Oregon, Eugene, in 2018
#
# Licenced under Creative Commons Attribution 4.0 International (CC BY 4.0)
# You are free to copy, use, modify and redistribute this work provided that you 
# provide appropriate credit to the original author, provide a link to the licence 
# and indicate if changes where made.
# Full terms at: https://creativecommons.org/licenses/by/4.0/
#
# List of functions:
# Primary: gassman(), gassman_2(), qimax(), mod_b(), mode_multi(),mod_a(), dem()
# Bounds: Voight(), Reuss(), hs_bounds()
# Polarization factors: pe1(), qu1(), pe4(), qu4()
# Auxiliary functions: theta_oblate(), theta(prolate), ff(), ccsi(),

#-------------------------------------
import numpy as np
import math
import physics

zero=1e-12       # almost zero


#-------------------------------------
def fitmelt(vp0,vp1,vpi,rom,roi,am,ai):
# ai	Aspect ratio of inclusions
# am	Aspect ratio of solid grains
# Vs from Brocher
	vs0 = physics.vp2vs(vp0*1e-3)*1e3
# Moduli of matrix
	Km = rom*(vp0**2.-4./3.*vs0**2.)
	Gm = rom*vs0**2.
# moduli of melt
	Ki = roi*vpi**2  #1.27e1
	Gi = zero
	
	dmelt=0.002
	melt=np.ones(Gm.shape)*dmelt	# Initialize melt fraction or porosity

	for x in np.ndindex(Gm.shape):
		if vp1[x] < vp0[x]:
			vpdiff=-2.0
			while ( vpdiff <= 0 and melt[x] < 1.):
				me=melt[x]
				so = 1.-me
				roe = roi*me+so*rom[x]
# Calculate moduli
# Relaxed, low frequency case - dry + Gassman
				Kd,Gd = mod_b_scalar(Km[x],zero,Gm[x],zero,am,ai,so,me)
				Ks,Gs = gassman_2(Kd,Gd,Km[x],Gm[x],Ki,Gi,me)
#				print(Ks,Gs,roe)			
				Vpr=math.sqrt((Ks+4.0/3.0*Gs)/roe)
				vpdiff=vp1[x]-Vpr
#		print(me,Ks,Gs,Vpr,vpdiff)		
				melt[x] = melt[x]+dmelt
		else:
			melt[x]=0.0
	return melt

#-------------------------------------
def fitmelttemp(vp0,vp1,vpi,rom,roi,am,ai,t0,t35,Q):
# ai	Aspect ratio of inclusions
# am	Aspect ratio of solid grains
# moduli of melt
	Ki = roi*vpi**2  #1.27e1
	Gi = zero
	
	dmelt = 0.002									  # melt step for iteration
	dt = dmelt *(t35-t0)/0.35				# corresponding temperature step
	melt = np.ones(vp1.shape)*dmelt	# Initialize melt fraction or porosity
	temp = t0*1.0+dt										# intialize temp array
	for x in np.ndindex(vp1.shape):
		if vp1[x] < vp0[x]:
			vpdiff=-2.0    # initialize vpdiff			
			while ( vpdiff <= 0 and melt[x] < 1.):
# Correct for temp
				vpref=physics.karato_a(vp0[x],t0[x],temp[x]-t0[x],Q)
# Vs from Brocher
				vsref = physics.vp2vs(vpref*1e-3)*1e3
# Moduli of matrix
				Km = rom[x]*(vpref**2.-4./3.*vsref**2.)
				Gm = rom[x]*vsref**2.
#
				me=melt[x]
				so = 1.-me
				roe = roi*me+so*rom[x]
# Calculate moduli
# Relaxed, low frequency case - dry + Gassman
				Kd,Gd = mod_b_scalar(Km,zero,Gm,zero,am,ai,so,me)
				Ks,Gs = gassman_2(Kd,Gd,Km,Gm,Ki,Gi,me)
#				print(Ks,Gs,roe)			
				Vpr=math.sqrt((Ks+4.0/3.0*Gs)/roe)
				vpdiff=vp1[x]-Vpr
#		print(me,Ks,Gs,Vpr,vpdiff)		
				melt[x] = melt[x]+dmelt
				temp[x] = temp[x]+dt[x]
			print(vp0[x],vpref,vp1[x],dt[x],t0[x],temp[x],melt[x])
		else:
			melt[x]=0.0
	return melt-dmelt,temp-dt



#-------------------------------------
def fitmelt_v(vp0,vp1,vpi,rom,roi,am,ai):
# Same as above but vpi and roi are vectors
# ai	Aspect ratio of inclusions
# am	Aspect ratio of solid grains
# Vs from Brocher
	vs0 = physics.vp2vs(vp0*1e-3)*1e3
# Moduli of matrix
	Km = rom*(vp0**2.-4./3.*vs0**2.)
	Gm = rom*vs0**2.
# moduli of melt
	Ki = roi*vpi**2  #1.27e1
	Gi = zero
	
	dmelt=0.002
	melt=np.ones(Gm.shape)*dmelt	# Initialize melt fraction or porosity

	for x in np.ndindex(Gm.shape):
		if vp1[x] < vp0[x]:
			vpdiff=-2.0
			while ( vpdiff <= 0 and melt[x] < 1.):
				me=melt[x]
				so = 1.-me
				roe = roi[x]*me+so*rom[x]
# Calculate moduli
# Relaxed, low frequency case - dry + Gassman
				Kd,Gd = mod_b_scalar(Km[x],zero,Gm[x],zero,am,ai,so,me)
				Ks,Gs = gassman_2(Kd,Gd,Km[x],Gm[x],Ki[x],Gi,me)
#				print(Ks,Gs,roe)			
				Vpr=np.sqrt((Ks+4.0/3.0*Gs)/roe)
				vpdiff=vp1[x]-Vpr
#		print(me,Ks,Gs,Vpr,vpdiff)		
				melt[x] = melt[x]+dmelt
		else:
			melt[x]=0.0
	return melt

#-------------------------------------
def dvpdm(vp0,dens,me,al,am,fluid):
# Adjust vp to km/s if it is in m/s
  vp0=vp0*1.e-3
# Some constants
# Elastic moduli are in GPa, density is in g/cm^2
  if fluid == 'magma':
    vpvs = 1.73
    Ki = 1.27e1
    Gi = 0.000
    roi = 2.4
  elif fluid == 'water':
    vpvs = 1.9
    Ki = 2.2
    Gi = 0.000  
    roi = 1.05    
  vs0 = physics.vp2vs(vp0)
  Km = dens*(vp0**2.-4./3.*vs0**2.)
  Gm = dens*vs0**2.
  vp=me*0.0
  f=me
  if f > 0:
      Kd,Gd = mod_b_scalar(Km,zero,Gm,zero,am,al,1-f,f)
      Ks,Gs = gassman_2(Kd,Gd,Km,Gm,Ki,Gi,f)
      roe = dens*(1-f)+roi*f
      vp=np.sqrt((Ks+4.0/3.0*Gs)/roe)
  else:
      vp=vp0
  return vp*1.e3

#-------------------------------------
# Gassman's equation for fluid substitution
def gassman(Kd,Gd,Km,Gm,Kl,Gl,p):
# I follow here the equation published by Korringa 1979
# Km, Gm = moduli of solid matrix
# Kl, Gl = moduli of fluid
# Kd, Gd = moduli of dry composite
# p = porosity
# Output Kw, Gw = fluid saturated moduli
# Apply Gassman's equation
#  Kw = Kd
  Kw = Kd+(Km-Kd)**2./(Km-Kd+p*(Km/Kl)*(Km-Kl))
  Gw = Gd
  return Kw, Gw

#-------------------------------------
def gassman_2(Kd,Gd,Km,Gm,Kl,Gl,p):
# I follow here the equation published by Marion 1990 PhD Thesis
# This should be equivalent to the one in gassman()
# Km, Gm = moduli of solid matrix
# Kl, Gl = moduli of fluid
# Kd, Gd = moduli of dry composite
# p = porosity
# Output Kw, Gw = fluid saturated moduli
# Apply Gassman's equation
#  Kw = Kd
  Kw = Kd+(1.-Kd/Km)**2./(p/Kl+(1.-p)/Km-Kd/Km**2.)
  Gw = Gd
  return Kw, Gw

#-------------------------------------  
def qimax(Mh,Ml):
# Estimate of the maximum P-wave attenuation from the relaxed and unrelaxed moduli
# Mh = high-frequency modulus (unrelaxed)
# Ml = low-frequency modulus (relaxed)
# a should be = 2 in the assumption of a standard linear solid with a single
# relaxation frequency. In general a <= 2
  a = 2.0
  attenuation = (Mh - Ml)/(a*np.sqrt(Mh*Ml))
  return attenuation
#-------------------------------------
def mod_b_scalar(K1,K2,G1,G2,a1,a2,c1,c2):
# Apply Self-consistent approximation. This routine should be vectorized
# Here I follow Berryman 1980ii. This gives the same results as Schmeling
# 1985 only for the case of spherical pockets. We need to define the geometry
# of both the solid and the liquid phase. Could be modified to deal with
# an arbitrary number of phases with different properties. This is the 
# symmetric "true" sca, which should be preferred (according to Berryman).
# There is no material corresponding to the "matrix", instead both phases 
# solid, liquid or gas, need to be specified with their moduli and pore or grain 
# shape
# K1, K2: bulk moduli of phase 1 and phase 2 (scalar)
# G1, G2: shear moduli (scalar)
# a1, a2: aspect ratios (scalar)
# c1, c2: volume fractions (these should be scalars)
# 
  nmax=200                # maximum number of iterations
  convlim=1e-6             # convergence limit
  
# Initialize arrays as simple average 
#  Kd=(K1+K2)/2.0
#  Gd=(G1+G2)/2.0
  Kd = (Voight(K1,K2,c2)+Reuss(K1,K2,c2))/2.0
  Gd = (Voight(G1,G2,c2)+Reuss(G1,G2,c2))/2.0
# Alternative start with properties of the previous step
#  Kd=K1
#  Gd=G1
  nu = (3.*Kd-2.*Gd)/2.0/(3.*Kd+Gd)
# Initialize and start effective medium calculation
  convergence=50.
  nn=0
# Solve by iteration
  while convergence >= convlim and nn <= nmax:
      K0=Kd
      G0=Gd
# Calculate the polarization factors
# Phase 1 (usually the solid phase)
      P1=pe4(Kd,K1,Gd,G1,nu,a1)
      Q1=qu4(Kd,K1,Gd,G1,nu,a1)
# Phase 2 (usually the liquid phase)
      P2=pe4(Kd,K2,Gd,G2,nu,a2)
      Q2=qu4(Kd,K2,Gd,G2,nu,a2)
# Berryman's SC "mixing" expression
      Kd = (c1*K1*P1+c2*K2*P2)/(c1*P1+c2*P2)
      Gd = (c1*G1*Q1+c2*G2*Q2)/(c1*Q1+c2*Q2)
      nu = 0.5*(3.0*Kd-2.0*Gd)/(3.0*Kd+Gd)
# Check convergence
      convergence = math.sqrt((K0-Kd)**2.0+(G0-Gd)**2.0)
      nn=nn+1
# Output number of iterations for debug
#    print c2,nn
  Ksca=Kd
  Gsca=Gd
  return Ksca,Gsca
  
    
#-------------------------------------
def mod_b(K1,K2,G1,G2,a1,a2,c1a,c2a):
# Apply Self-consistent approximation. This routine should be vectorized
# Here I follow Berryman 1980ii. This gives the same results as Schmeling
# 1985 only for the case of spherical pockets. We need to define the geometry
# of both the solid and the liquid phase. Could be modified to deal with
# an arbitrary number of phases with different properties. This is the 
# symmetric "true" sca, which should be preferred (according to Berryman).
# There is no material corresponding to the "matrix", instead both phases 
# solid, liquid or gas, need to be specified with their moduli and pore or grain 
# shape
# K1, K2: bulk moduli of phase 1 and phase 2 (scalar)
# G1, G2: shear moduli (scalar)
# a1, a2: aspect ratios (scalar)
# c1a, c2a: volume fractions (these should be numpy arrays)
# 
  nmax=1000                # maximum number of iterations
  convlim=1e-7             # convergence limit
  nm=np.shape(c1a)[0]      # size of volume fractions array
# Initialize arrays as simple average 
  Ksca=np.linspace(K1,K2,num=nm)
  Gsca=np.linspace(G1,G2,num=nm)
  for j in range(0,nm):
    c1=c1a[j]
    c2=c2a[j]
# Start with Voight-Reuss-Hill average
#    Kd = ( Voight(K1,K2,c2) + Reuss(K1,K2,c2) ) / 2.0
#    Gd = ( Voight(G1,G2,c2) + Reuss(G1,G2,c2) ) / 2.0    
# Alternative start with properties of the previous step
    if j > 1:
      Kd=Ksca[j-1]
      Kg=Gsca[j-1]
    else:
      Kd=K1
      Gd=G1
    nu = (3.*Kd-2.*Gd)/2.0/(3.*Kd+Gd)
# Initialize and start effective medium calculation
    convergence=50.
    nn=0
# Solve by iteration
    while convergence >= convlim and nn <= nmax:
      K0=Kd
      G0=Gd
# Calculate the polarization factors
# Phase 1 (usually the solid phase)
      P1=pe4(Kd,K1,Gd,G1,nu,a1)
      Q1=qu4(Kd,K1,Gd,G1,nu,a1)
# Phase 2 (usually the liquid phase)
      P2=pe4(Kd,K2,Gd,G2,nu,a2)
      Q2=qu4(Kd,K2,Gd,G2,nu,a2)
# Berryman's SC "mixing" expression
      Kd = (c1*K1*P1+c2*K2*P2)/(c1*P1+c2*P2)
      Gd = (c1*G1*Q1+c2*G2*Q2)/(c1*Q1+c2*Q2)
      nu = 0.5*(3.0*Kd-2.0*Gd)/(3.0*Kd+Gd)
# Check convergence
      convergence = math.sqrt((K0-Kd)**2.0+(G0-Gd)**2.0)
      nn=nn+1
# Output number of iterations for debug
#    print c2,nn
    Ksca[j]=Kd
    Gsca[j]=Gd
  return Ksca,Gsca



#-------------------------------------
def mod_multi(K,G,a,c):
# Apply Self-consistent approximation.
# Here I follow Berryman 1980ii. This gives the same results as Schmeling
# 1985 only for the case of spherical pockets. This is the 
# symmetric "true" sca, which should be preferred (according to Berryman)
# This function is equivalent to mod_b, but it works for any number of phases
# not just 2
# K = bulk moduli of components (1D numpy array)
# G = shear moduli of components (1D numpy array)
# a = aspect ratios of components (1D numpy array)
# c = volume fractions of components (2D numpy array)
# The sum of of each row of c should be 1.0 ((np.sum(c,axis=0)==1)==true)
# 
  nmax=100                 # maximum number of iterations (10-100 should work) 
  convlim=1e-7
  n1,n2=np.shape(c)
# Initialize arrays as simple average 
  Ksca=np.linspace(K[0],K[-1],num=n2)
  Gsca=np.linspace(G[0],G[-1],num=n2)

  for j in range(n2):
# Start with Voight-Reuss-Hill average (now changed as this was inaccurate)
#    Kd = ( Voight(K1,K2,c2) + Reuss(K1,K2,c2) ) / 2.0
#    Gd = ( Voight(G1,G2,c2) + Reuss(G1,G2,c2) ) / 2.0
# Start from the properties of the previous step
    if j > 1:
      Kd=Ksca[j-1]
      Kg=Gsca[j-1]
    else:
      Kd=K[0]
      Gd=G[0]
    nu = (3*Kd-2*Gd)/2.0/(3*Kd+Gd)
# Initialize and start effective medium calculation
    convergence=50.0
    nn=0
    while convergence >= convlim and nn <= nmax:
      K0=Kd
      G0=Gd
# Polarization factors
      P=np.zeros(n1)
      Q=np.zeros(n1)
# Loop over components
      for i in range(n1):
        P[i]=pe4(Kd,K[i],Gd,G[i],nu,a[i])
        Q[i]=qu4(Kd,K[i],Gd,G[i],nu,a[i])
# Berryman's SC expression
      top=0.0
      bot=0.0
      for i in range(n1):
        top=top+c[i,j]*K[i]*P[i]
        bot=bot+c[i,j]*P[i]
      Kd = top/bot
      top=0.0
      bot=0.0
      for i in range(n1):
        top=top+c[i,j]*G[i]*Q[i]
        bot=bot+c[i,j]*Q[i]
      Gd = top/bot
      nu = 0.5*(3.0*Kd-2.0*Gd)/(3.0*Kd+Gd)
# Check convergence
      convergence = math.sqrt((K0-Kd)**2.0+(G0-Gd)**2.0)
      nn=nn+1
# Output number of iterations for debug
#    print c2,nn
    Ksca[j]=Kd
    Gsca[j]=Gd
  return Ksca,Gsca
  
  
#-------------------------------------
def mod_a(K1,K2,G1,G2,a2,c2a):
# Apply Self-consistent approximation. This routine should be vectorized
# Here I follow Berryman 1980ii, modified to give the same results as
# Schmeling 1985, i.e. the "asymmetric" sca with the solid as a special
# component that acts as a matrix. I calculate only the polarization factors for
# the liquid phase and I have modified the "mixing" step accordingly
# K1, K2: bulk moduli of solid (phase 1) and liquid (phase 2) - scalars
# G1, G2: shear moduli (scalar)
# a2: aspect ratio of inclusions (scalar)
# c2a: porosity - volume fractions of inclusions (these should be numpy arrays)
#
  nmax=1000
  convlim=1e-6
  nm=np.shape(c2a)[0]
# Initialize arrays as simple average 
  Ksca=np.linspace(K1,K2,num=nm)
  Gsca=np.linspace(G1,G2,num=nm)
  for j in range(0,nm):
    c2=c2a[j]
# Start with Voight-Reuss-Hill average
    Kd = ( Voight(K1,K2,c2) + Reuss(K1,K2,c2) ) / 2.0
    Gd = ( Voight(G1,G2,c2) + Reuss(G1,G2,c2) ) / 2.0    
    nu = (3.*Kd-2.*Gd)/2.0/(3.*Kd+Gd)
# Initialize and start effective medium calculation
    convergence=50
    nn=0
    while convergence >= convlim and nn <= nmax:
      K0=Kd
      G0=Gd
# Coefficients for inclusions
      P2=pe4(Kd,K2,Gd,G2,nu,a2)
      Q2=qu4(Kd,K2,Gd,G2,nu,a2)
# Berryman's SC expression modified
      Kd = K1 + c2*(K2-K1)*P2
      Gd = G1 + c2*(G2-G1)*Q2
      nu = 0.5*(3.0*Kd-2.0*Gd)/(3.0*Kd+Gd)
# Check convergence
      convergence = math.sqrt((K0-Kd)**2.0+(G0-Gd)**2.0)
      nn=nn+1
# Output number of iterations for debug
#    print c2,nn
    Ksca[j]=Kd
    Gsca[j]=Gd
  return Ksca,Gsca


#-------------------------------------
def dem(K1,K2,G1,G2,a2,c2a):
# Modified routine to apply the differential effective medium theory (DEM)
# Here I use the same polarization factors as for the SCA, but only for 
# the liquid phase.
# I have modified the "mixing" step to implement the DEM, i.e. I integrate 
# the porosity from 0 to the desired value.
# K1, K2: bulk moduli of solid (phase 1) and liquid (phase 2) - scalars
# G1, G2: shear moduli (scalar)
# a2: aspect ratio of inclusions (scalar)
# c2a: porosity - volume fractions of inclusions (these should be numpy arrays)
#
  nm=np.shape(c2a)[0]
  ns=10
# Initialize arrays as simple average 
  Kdem=np.linspace(K1,K2,num=nm)
  Gdem=np.linspace(G1,G2,num=nm)
  for j in range(nm):
    c2=c2a[j]
    dc=(c2a[1]-c2a[0])/ns
# Start with Voight-Reuss-Hill average
    if j==0:
      K0 = K1
      G0 = G1
    else:
      K0 = Kdem[j-1]
      G0 = Gdem[j-1]
    for i in range(ns):
      nu = (3.0*K0-2.0*G0)/2.0/(3.*K0+G0)
# Coefficients for inclusions
      P2=pe4(K0,K2,G0,G2,nu,a2)
      Q2=qu4(K0,K2,G0,G2,nu,a2)
# Berryman's SC expression modified
      dK = 1.0/(1.0-c2)*dc*(K2-K0)*P2
      dG = 1.0/(1.0-c2)*dc*(G2-G0)*Q2
      K0 = K0 + dK
      G0 = G0 + dG
    Kdem[j]=K0
    Gdem[j]=G0
  return Kdem,Gdem


############################  
# Some rigorous bounds which may be handy

#-------------------------------------
# The Voight bound, i.e. the simple weighted average
def Voight(a,b,m):
  x = a*(1.0-m)+b*m
  return x

#-------------------------------------
# The Reuss bound, i.e. the harmonic mean
def Reuss(a,b,m):
  if b > 0:
    x = 1.0/((1.0-m)/a+m/b)
  else:
    x = np.zeros_like(m) 
  return x

#-------------------------------------
# The Voight bound for n components
def Voight_n(x,c):
	y = np.average(x,weights=c)
	return y

#-------------------------------------
# The Reuss bound for n components
def Reuss_n(x,c):
	a = np.average(1./x,weights=c)
	y=1./a
	return y

#-------------------------------------
# Voight-Reuss-Hill average for n components
def VRH_n(x,c):
	y1=Voight_n(x,c)
	y2=Reuss_n(x,c)
	y=(y1+y2)/2.
	return y

#-------------------------------------
def hs_bounds(K1,K2,G1,G2,c1,c2):
# Hashin-Shtrikman bounds. We get the upper and lower bound  by swapping
# the first and second component
  Khs=K1+c2/(1./(K2-K1)+c1/(K1+4./3.*G1))
  Ghs=G1+c2/(1./(G2-G1)+(2.*c1*(K1+2.*G1))/(5.*G1*(K1+4./3.*G1)))
  return Khs, Ghs
  

############################  
# Some functions used above

#-------------------------------------
# bulk modulus polarization factor for spheres
def pe1(KK,Ki,G):
  P=(KK+4.0/3.0*G)/(Ki+4.0/3.0*G)  
  return P
  
#-------------------------------------
# shear modulus polarization factor for spheres
def qu1(KK,Gi,G):
  csi=ccsi(KK,G)
  Q=(G+csi)/(Gi+csi)
  return Q

#-------------------------------------
def theta_oblate(a):
  th=a/(1.0-a**2.0)**(1.5)*(math.acos(a)-a*(1.0-a**2.0)**(0.5))
  return th

#-------------------------------------
def theta_prolate(a):
  th=a/(a**2.0-1.0)**(1.5)*(a*(a**2.0-1.0)**(0.5)-math.acosh(a))
  return th
  
#-------------------------------------
def ff(a,t):
  f=a**2.0/(1.0-a**2.0)*(3.0*t-2.0)
  return f

#-------------------------------------
def ccsi(K,G):
  c=G/6.0*(9.0*K+8.0*G)/(K+2.0*G)
  return c

#-------------------------------------
def pe4(KK,Ki,G,Gi,nu,al):
# bulk modulus polarization factor for spheroids
  A=Gi/G-1.0
  B=(Ki/KK-Gi/G)/3.0  
  R=(1.0-2.0*nu)/2.0/(1.0-nu)
#  R=3.0*G/(3.0*KK+4.0*G)  ! why this?  
# Spheres
  if al == 1: 
    P = (KK+4.0/3.0*G)/(Ki+4.0/3.0*G)  
  else:
# Oblate spheroids
    if al < 1:
      th=theta_oblate(al)
# Prolate spheroids      
    elif al > 1:
      th=theta_prolate(al)
    f=ff(al,th)

    F1 = 1.0+A*( 1.5*(f+th) - R*(1.5*f+2.5*th-4.0/3.0) )
    F2 = 1.0+(A*(1.0+((3.0/2.0)*(f+th))-((1.0/2.0)*R*((3.0*f)+(5.0*th)))))+(B*(3.0-(4.0*R))) \
            +((1.0/2.0)*A*(A+(3.0*B))*(3.0-(4.0*R))*(f+th-(R*(f-th+(2.0*(th**2.0))))))
    T=3.0*F1/F2
    P=T/3.0
  return P

#-------------------------------------
def qu4(KK,Ki,G,Gi,nu,al):
# shear modulus polarization factor for spheroids
  A=Gi/G-1.0
  B=(Ki/KK-Gi/G)/3.0
  R=(1.0-2.0*nu)/2.0/(1.0-nu)
#  R=3.0*G/(3.0*KK+4.0*G)  ! why this?
# Spheres
  if al == 1:
    csi=ccsi(KK,G)
    Q=(G+csi)/(Gi+csi)
  else:
# Oblate spheroids
    if al < 1:
      th=theta_oblate(al)
# Prolate spheroids      
    elif al > 1:
      th=theta_prolate(al)
    f=ff(al,th)
# Wu put a - here, but the rock physics handbook puts a +
# Who is right? Wu is wrong?
    F1=1.0+A*( 1.5*(f+th) - R*(1.5*f+2.5*th-4.0/3.0) )
    F2=1.0+(A*(1.0+((3.0/2.0)*(f+th))-((1.0/2.0)*R*((3.0*f)+(5.0*th)))))+(B*(3.0-(4.0*R))) \
          +((1.0/2.0)*A*(A+(3.0*B))*(3.0-(4.0*R))*(f+th-(R*(f-th+(2.0*(th**2.0))))))
    F3=1.0+A*(1.0-(f+1.5*th)+R*(f+th))
    F4=1.0+0.25*A*(f+3.0*th-R*(f-th))
    F5=A*(-f+R*(f+th-4.0/3.0)) + B*th*(3.0-4.0*R)
    F6=1.0+A*(1+f-R*(f+th)) + B*(1.0-th)*(3.0-4.0*R)
    F7=2.0+0.25*A*(3.0*f+9.0*th-R*(3.0*f+5*th)) + B*th*(3.0-4.0*R)
    F8=A*(1.0-2.0*R +0.5*f*(R-1.0) + th*0.5*(5.0*R-3.0)) +B*(1.0-th)*(3.0-4.0*R)
    F9=A*((R-1.0)*f-R*th)+B*th*(3.0-4.0*R)
    Q=(2.0/F3+1.0/F4+(F4*F5+F6*F7-F8*F9)/(F2*F4))/5.0
  return Q
