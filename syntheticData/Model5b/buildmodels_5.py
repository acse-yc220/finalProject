#!/anaconda3/bin/python
# This script reads a png image of lithology and builds models of physical properties
# Michele Paulatto - Imperial College London - May 2019
#
# Licenced under Creative Commons Attribution 4.0 International (CC BY 4.0)
# You are free to copy, use, modify and redistribute this work provided that you 
# give appropriate credit to the original author, provide a link to the licence 
# and indicate if changes were made.
# Full terms at: https://creativecommons.org/licenses/by/4.0/


# Import own libraries
import physics
import sca
# Import common libraries
import numpy as np
import scipy.io as scipio
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from math import pi,e,log,sqrt
import imageio
import iapws
from FyeldGenerator import generate_field
from astropy.convolution import convolve, Gaussian2DKernel

# Helper that generates power-law power spectrum
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk

# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b

#===============================

# Some general constants
kelvin=273.15	# zero degrees celsius in Kelvin
zero=1e-10		# Use instead of actual zero in some cases to avoid instabilities
idx = 5.		# input x grid spacing
idz = 5.		# input z grid spacing
dx=200.			# output x grid spacing
dz=200.			# output z grid spacing

# Import image of lithology. This can be made with a vector graphics software e.g. Inkscape
im = imageio.imread("sketch_5_5.png")

# Convert colorscale to integers from 0 to 9, representing the lithological classes
A = im[:,:,0]/40.
A = 6-A.astype(int)

# Median filter before subsample to avoid smoothing
A = ndimage.median_filter(A,size=(10,10))
A = ndimage.median_filter(A,size=(10,10))

# Resample to dx and dz
zoomfactor_x = idx/dx
zoomfactor_z = idz/dz
A = ndimage.zoom(A,(zoomfactor_z,zoomfactor_x),order=0)
print(A.shape)

# X and Z arrays
xedge = np.arange(0,24000.,dx)
zedge = np.arange(0,10000.,dz)
# Meshgrid for plotting
X,Z = np.meshgrid(xedge,zedge)

# Names of lithological classes
litos = np.asarray([
	'Water',
	'Andesite clastic sediments',
	'Andesite lavas',
	'Metasediments',
	'Granitic crust',
	'Diorite Intrusions'])

	
# Solid matrix Vp at standard temperature and pressure in m/s from Christensen (1979)
vp0 = np.asarray([
	1500, 	5533,	5533,	5829,	6246,	6497])

# Solid matrix Vs at standard temperature and pressure in m/s from Christensen (1979)
vs0 = np.asarray([
	zero,	3034,	3034,	3406,	3669,	3693])

# Solid matrix Qp at standard temperature and pressure (adimensional)
qp0 = np.asarray([
	5e3,	5e3,	5e3,	5e3,	5e3,	5e3])
# Solid matrix density at standard temperature and pressure in Kg/m^3 from Christensen (1979)
dn0 = np.asarray([
	1000,	2627,	2627,	2682,	2652,	2810])

# Background fractional porosity for each class
por0 = np.asarray([
	0.000,	0.08,	0.08,	0.03,	0.01,	0.001])

# Pore aspect ratio
aw = np.asarray([
	0.0,	0.2,	0.5,	0.1,	0.05,	0.1])

# Anharmonic derivative of Vp with respect to T from Christensen (1979) in m/s/K
dvpdtah = np.asarray([
	-0.0,	-3.9e-4,	-3.9e-4,	-3.9e-4,	-3.9e-4,	-3.9e-4])
# Anharmonic derivative of Vs with respect to T from Christensen and Stanley 2003 in m/s/K
dvsdtah = np.asarray([
	-0.0,	-2.1e-4,	-2.1e-4,	-2.1e-4,	-2.1e-4,	-2.1e-4])
# Derivative of Vp with respect to P from Christensen and Stanley 2003 in m/s/MPa
dvpdp = np.asarray([
	0.0,	0.36,	0.36,	0.36,	0.36,	0.36])
dvsdp = dvpdp/1.73
# Activation enthalpy/energy for seismic attenuation Burgman and Dresen (2008) in J/mol
hh = np.asarray([
	0.0,	2.2e5,	2.2e5,	2.2e5,	2.2e5,	2.2e5])


# Build starting arrays of vp, vs and density
# Initialize arrays
Avp0 = np.empty_like(A)*1.0		# Vp
Avs0 = np.empty_like(Avp0)		# Vs
Adn0 = np.empty_like(Avp0)		# density
Aqp0 = np.empty_like(Avp0)		# Qp
Aqs0 = np.empty_like(Avp0)		# Qs
Aaw = np.empty_like(Avp0)		# aspect ratio for porosity
Advpdp = np.empty_like(Avp0)	# dVp/dP 
Advsdp = np.empty_like(Avp0)	# dVs/dP
Advpdtah = np.empty_like(Avp0)	# dVp/dT anharmonic term
Advsdtah = np.empty_like(Avp0)	# dVs/dT anharmonic term
AH = np.empty_like(Avp0)		# Activation enthalpy
# Assign values based omn lithology
for x in np.ndindex(A.shape):
	Avp0[x] = vp0[A[x]]
	Avs0[x] = vs0[A[x]]
	Adn0[x] = dn0[A[x]]
	Aqp0[x] = qp0[A[x]]
	Aqs0[x] = qp0[A[x]]/2.35
	Aaw[x] = aw[A[x]]
	Advpdp[x] = dvpdp[A[x]]
	Advsdp[x] = dvsdp[A[x]]
	Advpdtah[x] = dvpdtah[A[x]]
	Advsdtah[x] = dvsdtah[A[x]]
	AH[x] = hh[A[x]]



# Find seabed
seabed = np.empty_like(xedge)
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size-1):
		if Avp0[k,i] <= 1500 and Avp0[k+1,i] > 1500:
			seabed[i] = zedge[k]+dz
			break
# Find basement
basement = np.empty_like(xedge)
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size-1):
		if A[k,i] <= 2 and A[k+1,i] > 2:
			basement[i] = zedge[k]+dz
			break

#-------------------------------------			
# Set up porosity field
por = np.empty_like(Avp0)
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size):
		if A[k,i] in [1,2]:
			por[k,i] = por0[A[k,i]] + 0.25 * np.exp(-(zedge[k]-seabed[i])/2000)
		elif A[k,i] in [3]:
			por[k,i] = por0[A[k,i]] + 0.15 * np.exp(-(zedge[k]-seabed[i])/2000)
		elif A[k,i] in [4]:
			por[k,i] = por0[A[k,i]] + 0.06 * np.exp(-(zedge[k]-basement[i])/2000)
		elif A[k,i] in [5]:
			por[k,i] = por0[A[k,i]] + 0.05 * np.exp(-(zedge[k]-basement[i])/2000)
		else:
			por[k,i] = 0.0
			
# Add random field
shape = A.shape
field = generate_field(distrib, Pkgen(2), shape)
por = por * (1.0 + field)
por[A==0] = 0.0
			
#-------------------------------------			
# Set up temperature field
dtdzc=0.050
dtdzh=0.040
dtdzd=0.010
temp = np.empty_like(Avp0)
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size):
		if A[k,i] == 0 :
			temp[k,i] = kelvin +10.	
		else:	
			temp[k,i] = kelvin + 10. + (zedge[k] - seabed[i])*dtdzc
# Add temp anomaly
temph = temp * 1.0
solidusdepth = np.empty_like(xedge)
for i in np.arange(xedge.size):
	if 11000 <= xedge[i] <= 17000 :
		for k in np.arange(zedge.size):
			temph[k,i] = temph[k,i] + 20. + (zedge[k] - seabed[i])*dtdzh
			if temph[k,i] >= 670. + kelvin:
				solidusdepth[i] = zedge[k]-dz
				break
		for k in np.arange(zedge.size):
			if zedge[k] > solidusdepth[i]:
				temph[k,i] = 670. + kelvin + (zedge[k] - solidusdepth[i])*dtdzd
tempano = ndimage.filters.gaussian_filter(temph-temp,sigma=[4,4],mode='nearest')
temp = temp + tempano

# Add magma lenses
#for i in np.arange(xedge.size):
#	for k in np.arange(zedge.size):
#		if 12500 <= xedge[i] <= 15500 and 5000 <= zedge[k] <= 5500:
#			temp[k,i] = 850 + kelvin
#		if 12500 <= xedge[i] <= 15500 and 8500 <= zedge[k] <= 9000:
#			temp[k,i] = 900 +kelvin					

temp = ndimage.filters.gaussian_filter(temp,sigma=[1,2],mode='nearest')
# Reset temp in water
temp[A==0] = 10. + kelvin

#-------------------------------------			
# Adjust density for temperature
Adn0 = physics.ddensdt_a(Adn0,kelvin+10,temp)

# Calculate pressure in MPa
pres0 = np.empty_like(Avp0)
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size):
		if k == 0:
			pres0[k,i] = 0.1
		else:
			pres0[k,i] = pres0[k-1,i]+(Adn0[k,i]*9.81*dz)*1e-6

# Calculate water properties given pressure and temp using IAPWS formulas
# https://iapws.readthedocs.io/en/latest/iapws.html
Awvp = np.empty_like(Avp0)
Awdn = np.empty_like(Avp0)
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size):
		if pres0[k,i] > 100:
			p = 100.
		else: 
			p = pres0[k,i]
		if temp[k,i] > 1000:
			t = 1000.
		else:
			t = temp[k,i]

#		water = iapws.iapws97.IAPWS97(T=t,P=p)
#		print(pres0[k,i],temp[k,i],vars(water))
#		Awvp[k,i] = water.w
#		Awdn[k,i] = water.rho
		Awvp[k,i] = 1500.
		Awdn[k,i] = 1000.
		
# Adjust density for porosity
Adn = Adn0*(1-por)+Awdn*por

# Recalculate pressure in MPa
pres = np.empty_like(Avp0)
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size):
		if k == 0:
			pres[k,i] = 0.1
		else:
			pres[k,i] = pres[k-1,i]+(Adn[k,i]*9.81*dz)*1e-6

# Adjust matrix Qp for temperature and pressure
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size):
		if A[k,i] != 0:
			Aqp0[k,i] = Aqp0[k,i]*np.exp(-temp[k,i]/250)
# Calculate matrix Qs
Aqs0 = Aqp0 / 2.25

# Adjust Vp and Vs for pressure in MPa
Avp = Avp0+Advpdp*pres   #+dvpdt*(temp0-kelvin)
Avs = Avs0+Advsdp*pres   #+dvpdt*(temp0-kelvin)

# Adjust Vp and Vs for temperature
Avp, dump =  physics.karato_q(Avp,kelvin+10,temp-kelvin-10,A=1.0,Ea=3e5,al=0.15,qmax=2e3)
Avs, dump =  physics.karato_q(Avs,kelvin+10,temp-kelvin-10,A=1.0/2.25,Ea=3e5,al=0.15,qmax=2e3)


#-------------------------------------
# Adjust Vp and Vs for porosity
# Calculate bulk and shear moduli of matrix
AKs = Adn0*(Avp**2.-4./3.*Avs**2.)
AGs = Adn0*Avs**2.

# Initialize arrays
AKd = np.empty_like(Avp0)
AGd = np.empty_like(Avp0)
AKr = np.empty_like(Avp0)
AGr = np.empty_like(Avp0)
AKur = np.empty_like(Avp0)
AGur = np.empty_like(Avp0)
AKw = Awdn*Awvp**2
AGw = zero

# Do effective medium theory calculation based on Berryman (1980)
# to add water-filled porosity
for x in np.ndindex(AKd.shape):
# Relaxed moduli (low frequency limit)
	AKd[x],AGd[x] = sca.mod_b_scalar(AKs[x],zero,AGs[x],zero,1.0,Aaw[x],1-por[x],por[x])
	AKr[x],AGr[x] = sca.gassman(AKd[x],AGd[x],AKs[x],AGs[x],AKw[x],AGw,por[x])
# Unrelaxed moduli (high frequency limit)
	AKur[x],AGur[x] = sca.mod_b_scalar(AKs[x],AKw[x],AGs[x],AGw,1.0,Aaw[x],1-por[x],por[x])

# Relaxed Vp and Vs 
Avp = np.sqrt((AKr+4.0/3.0*AGr)/Adn)
Avs = np.sqrt(AGr/Adn)

# Adjust Qp and Qs for porosity
deltap = (AKur-AKr)/AKur	# Bulk modulus defect
Aqppor = 8./deltap
Aqp = 1./(1./Aqp0+1./Aqppor)
deltas = (AGur-AGr)/AGur	# Shear modulus defect
Aqspor = 8./deltas
Aqs = 1./(1./Aqs0+1./Aqspor)


#-------------------------------------
# Adjust Vp for melt
solidustemp = 670. + kelvin	# Solidus temperature =  melting temperature
temp35 = 850. + kelvin		# Temperature at which we reach 35% melt content
# Melt distribution
melt = np.empty_like(Avp0)
melta = np.empty_like(Avp0)
for i in np.arange(xedge.size):
	for k in np.arange(zedge.size):
		if temp[k,i] > solidustemp:
			melt[k,i] = (temp[k,i] - solidustemp)/(temp35-solidustemp)*0.35
			melta[k,i] = 0.1	# melt inclusions aspect ratio
		else:
			melt[k,i] = 0.0
			melta[k,i] = 1.0
melt = np.clip(melt,0.0,1.0)			
#print(melt)

# Melt properties
meltdn = 2400.								# melt density
meltvp = 2300.								# melt Vp
meltK = np.empty_like(Avp)+meltdn*meltvp**2	# melt bulk modulus
meltG = zero								# melt shear modulus

# Recalculate bulk and shear moduli
AKs = Adn*(Avp**2.-4./3.*Avs**2.)
AGs = Adn*Avs**2.

# Do effective medium calculation to add melt 
for x in np.ndindex(AKd.shape):
	if melt[x] > 0:
# Relaxed moduli
		AKd[x],AGd[x] = sca.mod_b_scalar(AKs[x],zero,AGs[x],zero,1.0,melta[x],1-melt[x],melt[x])
		AKr[x],AGr[x] = sca.gassman(AKd[x],AGd[x],AKs[x],AGs[x],meltK[x],meltG,melt[x])
# Unrelaxed moduli
		AKur[x],AGur[x] = sca.mod_b_scalar(AKs[x],meltK[x],AGs[x],meltG,1.0,melta[x],1-melt[x],melt[x])
# Density
Adn = Adn*(1.-melt)+meltdn*melt	
		
# Relaxed Vp and Vs
Avp = np.sqrt((AKr+4.0/3.0*AGr)/Adn)
Avs = np.sqrt(AGr/Adn)

# Adjust Qp and Qs for porosity and melt
deltap = (AKur-AKr)/AKur
Aqppor = 8./deltap
Aqp = 1./(1./Aqp+1./Aqppor)
deltas = (AGur-AGr)/AGur
Aqspor = 8./deltas
Aqs = 1./(1./Aqs+1./Aqspor)


# ---------------------------
# Reset velocities and Q in water
Avp[A==0] = 1500.0
Avs[A==0] = 0.0
Aqp[A==0] = np.nan
Aqs[A==0] = np.nan
Adn[A==0] = 1000

# ---------------------------
# Apply smoothing due to limited resolution in geophysical inversion
#Avps = ndimage.filters.gaussian_filter(Avp,sigma=[2,2],mode='nearest')
#Avss = ndimage.filters.gaussian_filter(Avs,sigma=[2,2],mode='nearest')
#Aqps = ndimage.filters.gaussian_filter(Aqp,sigma=[2,2],mode='nearest')
#Aqss = ndimage.filters.gaussian_filter(Aqs,sigma=[2,2],mode='nearest')
#Adns = ndimage.filters.gaussian_filter(Adn,sigma=[2,2],mode='nearest')

kernel = Gaussian2DKernel(x_stddev=1.5)
Avps = convolve(Avp,kernel,'extend')
Avss = convolve(Avs,kernel,'extend')
Aqps = convolve(Aqp,kernel,'extend')
Aqss = convolve(Aqs,kernel,'extend')
Adns = convolve(Adn,kernel,'extend')


# ---------------------------
# Reset velocities and Q in water
Avps[A==0] = 1500.0
Avss[A==0] = 0.0
Aqps[A==0] = np.nan
Aqss[A==0] = np.nan
Adns[A==0] = 1000


#=====================================
# Save arrays to disk
file="input_fields.npz"
np.savez(file, classes=A, vp0=Avp0, vs0=Avs0, dn0=Adn0, por=por, temp=temp, melt=melt, 
		aspect=Aaw, x=xedge, z=zedge)

file="extra_params.npz"
np.savez(file, aenthalpy=AH, dvpdp=Advpdp, dvsdp=Advsdp, dvpdt=Advpdtah, dvsdt=Advsdtah,
		x=xedge, z=zedge)

# These are the "data" for clustering
file="output_fields.npz"
np.savez(file, vp=Avp, vs=Avs, dn=Adn, vpvs=Avp/Avs, qp=Aqp, qs=Aqs, x=xedge, z=zedge)

# These are the "data" for clustering
file="output_fields_smooth.npz"
np.savez(file, vp=Avps, vs=Avss, dn=Adns, vpvs=Avps/Avss, qp=Aqps, qs=Aqss, x=xedge, z=zedge)

#-------------------------------------
# Plot
# Figure 1: input fields
fig1 = plt.figure(figsize=(12,8))
ax1 = fig1.add_subplot(321)
ax1.set_aspect('equal')
ax1.invert_yaxis()
plt.pcolormesh(X,Z,A,cmap='tab10',vmin=-0.5, vmax=9.5)
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Classes")

ax2 = fig1.add_subplot(322)
ax2.set_aspect('equal')
ax2.invert_yaxis()
plt.pcolormesh(X,Z,Avp0,cmap='viridis_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Vp (m/s)")

ax3 = fig1.add_subplot(323)
ax3.set_aspect('equal')
ax3.invert_yaxis()
plt.pcolormesh(X,Z,Avs0,cmap='viridis_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Vs (m/s)")

ax4 = fig1.add_subplot(324)
ax4.set_aspect('equal')
ax4.invert_yaxis()
plt.pcolormesh(X,Z,Adn0,cmap='magma_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Density (kg/m$^3$)")

ax5 = fig1.add_subplot(325)
ax5.set_aspect('equal')
ax5.invert_yaxis()
plt.pcolormesh(X,Z,por,cmap='magma')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Porosity")

ax6 = fig1.add_subplot(326)
ax6.set_aspect('equal')
ax6.invert_yaxis()
plt.pcolormesh(X,Z,temp-kelvin,cmap='seismic')
cbar=plt.colorbar()
cbar.set_label("Temperature ($^{\circ}$C)")
plt.contour(X,Z,temp-kelvin, [200,400,670])
plt.plot(xedge,seabed,color='black')

fileplot='fig1_xsections_inputs.pdf'
plt.tight_layout()
plt.savefig(fileplot,dpi=300)

#---------------------------
# Figure 2: output fields
fig2 = plt.figure(figsize=(12,8))
ax1 = fig2.add_subplot(321)
ax1.set_aspect('equal')
ax1.invert_yaxis()
plt.pcolormesh(X,Z,Avp,cmap='viridis_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Vp (m/s)")

ax2 = fig2.add_subplot(322)
ax2.set_aspect('equal')
ax2.invert_yaxis()
plt.pcolormesh(X,Z,Avs,cmap='viridis_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Vs (m/s)")

ax3 = fig2.add_subplot(323)
ax3.set_aspect('equal')
ax3.invert_yaxis()
plt.pcolormesh(X,Z,Adn,cmap='magma_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Density (kg/m$^3$)")

ax4 = fig2.add_subplot(324)
ax4.set_aspect('equal')
ax4.invert_yaxis()
plt.pcolormesh(X,Z,Avp/Avs,cmap='magma_r',vmin=1.5,vmax=3)
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Vp/Vs")

ax5 = fig2.add_subplot(325)
ax5.set_aspect('equal')
ax5.invert_yaxis()
plt.pcolormesh(X,Z,1./Aqp,cmap='plasma')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("${Q_P}^{-1}$)")

ax6 = fig2.add_subplot(326)
ax6.set_aspect('equal')
ax6.invert_yaxis()
plt.pcolormesh(X,Z,1./Aqs,cmap='plasma')
cbar=plt.colorbar()
cbar.set_label("${Q_S}^{-1}$)")
plt.plot(xedge,seabed,color='black')

fileplot='fig2_xsections_outputs.pdf'
plt.tight_layout()
plt.savefig(fileplot,dpi=300)


#---------------------------
# Figure 4: Smooth output fields
fig4 = plt.figure(figsize=(12,8))
ax1 = fig4.add_subplot(321)
ax1.set_aspect('equal')
ax1.invert_yaxis()
plt.pcolormesh(X,Z,Avps,cmap='viridis_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Vp (m/s)")

ax2 = fig4.add_subplot(322)
ax2.set_aspect('equal')
ax2.invert_yaxis()
plt.pcolormesh(X,Z,Avss,cmap='viridis_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Vs (m/s)")

ax3 = fig4.add_subplot(323)
ax3.set_aspect('equal')
ax3.invert_yaxis()
plt.pcolormesh(X,Z,Adns,cmap='magma_r')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Density (kg/m$^3$)")

ax4 = fig4.add_subplot(324)
ax4.set_aspect('equal')
ax4.invert_yaxis()
plt.pcolormesh(X,Z,Avps/Avss,cmap='magma_r',vmin=1.5,vmax=3)
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("Vp/Vs")

ax5 = fig4.add_subplot(325)
ax5.set_aspect('equal')
ax5.invert_yaxis()
plt.pcolormesh(X,Z,1./Aqps,cmap='plasma')
plt.plot(xedge,seabed,color='black')
cbar=plt.colorbar()
cbar.set_label("${Q_P}^{-1}$)")

ax6 = fig4.add_subplot(326)
ax6.set_aspect('equal')
ax6.invert_yaxis()
plt.pcolormesh(X,Z,1./Aqss,cmap='plasma')
cbar=plt.colorbar()
cbar.set_label("${Q_S}^{-1}$)")
plt.plot(xedge,seabed,color='black')

fileplot='fig4_xsections_outputs_smooth.pdf'
plt.tight_layout()
plt.savefig(fileplot,dpi=300)

#--------- ------------------
# Figure 3: vertical cross-sections
fig3 = plt.figure(figsize=(12,5.5))
ax1 = fig3.add_subplot(151)

x1=40
x2=75

plt.plot(Avp[:,x1],zedge,color='blue')
plt.plot(Avp[:,x2],zedge,color='red')
plt.plot(Avp0[:,x1],zedge,color='blue',ls="--")
plt.plot(Avp0[:,x2],zedge,color='red',ls="--")
plt.plot(Avs[:,x1],zedge,color='blue')
plt.plot(Avs[:,x2],zedge,color='red')
plt.plot(Avs0[:,x1],zedge,color='blue',ls="--")
plt.plot(Avs0[:,x2],zedge,color='red',ls="--")
plt.ylabel("Z(m)")
plt.xlabel("Seismic velocity (m/s)")
ax1.invert_yaxis()

ax2 = fig3.add_subplot(152)
plt.plot(Adn[:,x1],zedge,color='blue')
plt.plot(Adn[:,x2],zedge,color='red')
plt.plot(Adn0[:,x1],zedge,color='blue',ls="--")
plt.plot(Adn0[:,x2],zedge,color='red',ls="--")
plt.xlabel("Density (kg/$m^3$)")
ax2.invert_yaxis()

ax3 = fig3.add_subplot(153)
plt.plot(por[:,x1],zedge,color='blue')
plt.plot(por[:,x2],zedge,color='red')
plt.plot(melt[:,x2],zedge,color='orange')
plt.xlabel("Volume fraction")
ax3.invert_yaxis()

ax4 = fig3.add_subplot(154)
plt.plot(temp[:,x1]-kelvin,zedge,color='blue')
plt.plot(temp[:,x2]-kelvin,zedge,color='red')
plt.xlabel("Temperature ($^{\circ}$C)")
ax4.invert_yaxis()

ax5 = fig3.add_subplot(155)
plt.plot(Aqp[:,x1],zedge,color='blue')
plt.plot(Aqp[:,x2],zedge,color='red')
plt.plot(Aqs[:,x1],zedge,color='blue')
plt.plot(Aqs[:,x2],zedge,color='red')
plt.xlabel("Attenuation Q")
plt.xlim(0,500)
ax5.invert_yaxis()

fileplot='fig3_vsections.pdf'
plt.tight_layout()
plt.savefig(fileplot,dpi=300)

plt.show()

