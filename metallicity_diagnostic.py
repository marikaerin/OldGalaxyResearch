#import no_repo_illustris_python as illustris_python
import numpy as np
#from util import hdf5lib
import h5py
import sys
from scipy.io import readsav
import simread.readsubfHDF5 as read_subf
import matplotlib.pyplot as plt

base='/orange/paul.torrey/IllustrisTNG/Runs/'
#run='L35n2160TNG'
#run='L205n2500TNG'
run='L75n1820TNG'
snapnr = 99

path=base+run

print path

# load the data
cat = read_subf.subfind_catalog( path+'/', snapnr, keysel = ['SubhaloMass', 'SubhaloMassType', 'SubhaloSFR', 'SubhaloGasMetallicitySfr'])
#, 'GroupFirstSub', 'SubhaloGrNr', 'GroupMass', 'GroupMass' ] )
SFRRem = readsav('data/LineFiles/SFRemissionlines_snap0'+ str(snapnr) +'.sav')
AGNem  = readsav('data/LineFiles/AGNemissionlines_snap0'+ str(snapnr) +'.sav')
PAGBem = readsav('data/LineFiles/PAGBemissionlines_snap0'+ str(snapnr) +'.sav')
Shckem = readsav('data/LineFiles/Shockemissionlines_snap0'+ str(snapnr) +'_refined.sav')

#nitrogen metallicty components
nitroSFR  = 10**(SFRRem['NII_6584_line_sfr_sim'])
nitroAGN  = 10**(AGNem['NII_6584_line_agn_sim'])
nitroPAGB = 10**(PAGBem['NII_6584_line_LIN_sim'])
nitroShk  = 10**(Shckem['NII_6584_shock'])

#sum of nitrogenII line strength
nitro  = nitroSFR + nitroAGN + nitroPAGB + nitroShk

#OII line strength components
oIISFR  = 10**(SFRRem['OII_3727_line_sfr_sim'])
oIIAGN  = 10**(AGNem['OII_3727_line_agn_sim'])
oIIPAGB = 10**(PAGBem['OII_3727_line_LIN_sim'])
oIIShk  = 10**(Shckem['OII_3727_shock'])

#sum of OII line strength
oII = oIISFR + oIIAGN + oIIPAGB + +oIIShk

#halpha line strength components
halphaSFR  = 10**(SFRRem['Halpha_line_sfr_sim'])
halphaAGN  = 10**(AGNem['Halpha_line_agn_sim'])
halphaPAGB = 10**(PAGBem['Halpha_line_LIN_sim'])
halphaShk  = 10**(Shckem['Ha_6563_shock'])

#sum of halpha line strength
halpha = halphaSFR + halphaAGN + halphaPAGB + halphaShk

#OIII doublet
oIIISFR  = 10**(SFRRem['OIII_5007_line_sfr_sim'])
oIIIAGN  = 10**(AGNem['OIII_5007_line_agn_sim'])
oIIIPAGB = 10**(PAGBem['OIII_5007_line_LIN_sim'])
oIIIShk  = 10**(Shckem['OIII_5007_shock'])

#sum of OIII line strength
oIII = oIIISFR + oIIIAGN + oIIIPAGB + oIIIShk

hbetaSFR  = 10**(SFRRem['Hbeta_line_sfr_sim'])
hbetaAGN  = 10**(AGNem['Hbeta_line_agn_sim'])
hbetaPAGB = 10**(PAGBem['Hbeta_line_LIN_sim'])
hbetaShk  = 10**(Shckem['Hb_4861_shock'])

hbeta = hbetaSFR + hbetaAGN + hbetaPAGB + hbetaShk

#simulation stellar masses, sfr and metallicity
stellar_mass = np.array( cat.SubhaloMassType[:,4] ) * 1e10 / 0.6774     # in units of M_solar
sfr          = np.array( cat.SubhaloSFR )                               # in units of M_solar / yr 
subgas_metal = np.array( cat.SubhaloGasMetallicitySfr )

#metallicity from line profiles
nii_ha = 8.9 + 0.57 * np.log10(nitro/halpha)
nii_oii = 8.9 + 0.57 * np.log10(nitro/oII)
R = np.log(nii_oii)
nii_diagnostic = np.log(1.54020 + 1.26602*R + 0.167977*R**R) + 6 #+8.93 
R23 = (oII + oIII)/hbeta
R1 = (oIII / oII)

#metallicity from line profile components
metal_SFR  = 8.9 + 0.57 * np.log10(nitroSFR/halphaSFR)
metal_AGN  = 8.9 + 0.57 * np.log10(nitroAGN/halphaAGN)
metal_PAGB = 8.9 + 0.57 * np.log10(nitroPAGB/halphaPAGB)
metal_Shk  = 8.9 + 0.57 * np.log10(nitroShk/halphaShk)

#Kewley R23 Method

#This method only uses OII, OII and HBeta.

#R23 method 1st iteration

def getZ94(r23):
        return 9.265-0.33*r23-0.202*r23**2-0.207*r23**3-0.333*r23**4

def getM91(r23,y):
        return 12 - 4.944 + .767*r23 + .602*r23**2 - y*(.29 + .332*r23 - .331*r23**2)

def getQ(Z94, r1):
        k = getIonCoeff(Z94)
        q = 10**(k[0] + k[1]*r1+k[2]*r1**2)
        return q

def getIonCoeff(metal):
        Z05 = [7.39167, 0.667891, 0.0680367] # [k0, k1, k2]
        Z1 = [7.46128, 0.685835, 0.0866086]
        Z2 = [7.57817, 0.739315, 0.0843640]
        Z5 = [7.73013, 0.843125, 0.118166]

        if metal in range(8.629,8.93):
                return Z5
        elif metal in range(8.24,8.629):
                return Z2
        elif metal in range(7.93,8.24):
                return Z1
        elif metal < 7.629:
                return Z05
        else:
                return 0

# ks for R23 method for 
def getKs(q):
        if q in range(5e6,1e7):
                return [-27.0004, 6.03910, -0.327006] #[k0, k1, k2]
        elif q in range( 1e7,2e7):
                return [-31.2133, 7.15810, -0.399343]
        elif q in range( 2e7,4e7):
                return [-36.0239, 8.44804, -0.483762]
        elif q in range( 4e7,8e7):
                return [-40.9994, 9.78396, -0.571551]
        elif q in range( 8e7,1.5e8):
                return [-44.7026, 10.8052, -0.640113]
        elif q in range( 1.5e8,3e8):
                return [-46.1589, 11.2557, -0.672731]
        elif q > 3e8:
                return [-45.6075, 11.2074, -0.674460]


def testConvergence(q1,q2):
        k1 = getKs(q1)
        k2 = getKs(q2)
        if k1 == k2:
                return True
        else:
                return False
def getR23(k, r23):

        R23 = (-k[1] + (k[1]**2 - 4*k[2]*(k[0] -r23))**(1/2))/(2*k[2])

# Takes no parameters. Returns list of metallicities.
def R23method():
        R23_metals = []

        R1 = oIII/oII
        R23 = (oII+oIII)/hbeta
        y = np.log(oIII/oII)

        #need to rework. it needs to keep looping until it converges. tempR23 should be run in getQ to compare qs.
        for idx in np.arange(0, len(R23)):
                #calculate z94
                z94 = getZ94(R23[idx])
                if z94 > 9.0:
                        #calculate q1
                        q = getQ(z94, R1[idx])
                        k = getKs(q)
                        tempR23 = getR23(k,R23[idx])
                        q2 = getQ(tempR23, R1[idx])

                        convergenge = False
                        while convergence==False:
                                k = getKs(q2)
                                tempR23 = getR23(k, R23[idx])
                                q = getQ(tempR23, R1[idx])
                                convergence = testConvergence(q, q2)
                        if convergence == True:
                                R23_metals.append(tempR23)

                elif z94 in range(8.5,9):
                        m91 = getM91(R23[idx],y[idx])
                        tempR23 = (m91+z94)/2
                        R23_metals.append(tempR23)
                elif z94 < 8.6:
                        q = getQ(z94, R1[idx])
                        k = getKs(q)
                        tempR23 = getR23(k,R23)

                        if tempR23 in range(z94-1, z94+1):
                                R23_metals.append(tempR23)
        return R23_metals

#run getR23
print "getR23"
R23_metals = getR23()
print "done"


#overall mass cut
mass_cut = stellar_mass > 3000153000
select_stellar_mass = stellar_mass[ mass_cut ] #from sim
sfr = sfr[ mass_cut ] #from sim
subgas_metal = subgas_metal[ mass_cut ] #from cat

#PLOTS

#R23 Diagnostic
x = np.log10(select_stellar_mass) #np.log(stellar_mass)
plt.xlabel("Log[M_Stellar]")
plt.ylabel("R23 Diagnostic")
plt.title("Stellar Mass v Metallicity (R23 Diagnostic)")
colors = ['blue','green','orange','yellow','red','purple','black']
print len(x), len(R23_metals)
for i in np.arange(0,len(R23_metals),1):
        plt.scatter(x, R23_metals, marker='.', s=1)
#plt.legend()
plt.show()



#R23 Diagnostic
x = np.log10(select_stellar_mass) #np.log(stellar_mass)
plt.xlabel("Log[M_Stellar]")
plt.ylabel("R23 Diagnostic")
plt.title("Stellar Mass v Metallicity (R23 Diagnostic)")
colors = ['blue','green','orange','yellow','red','purple','black']
for i in np.arange(0,len(R23_metals),1):
        plt.scatter(x, R23_metals[i][0], label = R23_metals[i][1] , marker='.', s=1, c=colors[i])
plt.legend()
#plt.show()

#R23 Diagnostic 2
x = np.log10(select_stellar_mass) #np.log(stellar_mass)
plt.xlabel("Log[M_Stellar]")
plt.ylabel("Metallicity R")
plt.title("Stellar Mass v Metallicity (R23 Diagnostic)")
colors = ['blue','green','orange','yellow','red','purple','black']
for i in np.arange(0,len(R23_metals),1):
        plt.scatter(x, R23_metals[i][0], label = R23_metals[i][1] , marker='.', s=1, c=colors[i])
plt.scatter(x, R+5.5, marker='.', s=1)
plt.legend()
#plt.show()

#R23 diagnostic with NII diagnostic
x = np.log10(select_stellar_mass) #np.log(stellar_mass)
plt.xlabel("Log[M_Stellar]")
plt.ylabel("Metallicity log[(OII+OII)/Hb]")
plt.title("Stellar Mass v Metallicity (R23 Diagnostic with NII Diagnostic)")
colors = ['blue','green','orange','yellow','red','purple','black']
for i in np.arange(0,len(R23_metals),1):
        plt.scatter(x, R23_metals[i][0], label = R23_metals[i][1] , marker='.', s=1, c=colors[i])
plt.scatter(x, nii_diagnostic, marker='.', s=1)
plt.legend()
#plt.show()

#NII diagnnostic
x = np.log10(select_stellar_mass) #np.log(stellar_mass)
y = nii_ha
plt.xlabel("Log[M_Stellar]")
plt.ylabel("(NII/Ha)")
plt.title("Stellar Mass v Metallicity (NII/Ha)")
plt.scatter(x, y, marker='.', s=1)
#plt.show()

#NII diagnostic using R
x = np.log10(select_stellar_mass) #np.log(stellar_mass)
y = nii_diagnostic
plt.xlabel("Log[M_Stellar]")
plt.ylabel("NII diagnostic (R)")
plt.title("Stellar Mass v Metallicity (NII Diagnostic using R)")
plt.scatter(x, y, marker='.', s=1)
#plt.show()


#Metallicity by SFR, AGN, PAGB, and Shock contributions only
x = np.log10(select_stellar_mass) #np.log(stellar_mass)
y = metal_SFR
plt.xlabel("Log(stellar mass > 1e8)")
plt.ylabel("metal_SFR")
plt.title("Stellar Mass v Metallicity(SFR Contribution)")
plt.scatter(x, y, marker='.', s=1)
#plt.show()


x = np.log10(select_stellar_mass) #np.log(stellar_mass)
y = metal_AGN
plt.xlabel("Log(stellar mass > 1e8)")
plt.ylabel("metal_AGN")
plt.title("Stellar Mass v Metallicity(AGN Contribution)")
plt.scatter(x, y, marker='.', s=1)
#plt.show()


x = np.log10(select_stellar_mass) #np.log(stellar_mass)
y = metal_PAGB
plt.xlabel("Log(stellar mass > 1e8)")
plt.ylabel("metal_PAGB")
plt.title("Stellar Mass v Metallicity(PAGB Contribution)")
plt.scatter(x, y, marker='.', s=1)
#plt.show()


x = np.log10(select_stellar_mass) #np.log(stellar_mass)
y = metal_Shk
plt.xlabel("Log(stellar mass > 1e8)")
plt.ylabel("metal_Shk")
plt.title("Stellar Mass v Metallicity(Shock Contribution)")
plt.scatter(x, y, marker='.', s=1)
#plt.show()
