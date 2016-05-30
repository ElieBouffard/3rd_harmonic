from scipy.fftpack import fft
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sys
import scipy.signal as diag
from astropy import constants as const
from gatspy.periodic import LombScargle
from gatspy.periodic import LombScargleFast
#This program generates a Flux signal form a star + planet system. We add to this the
#periodic noise of the telescope sensibility and a gaussian noise. Then, it is possible to 
#phase fold the global signal according to the planet's orbital frequency frequency or not. Then,
#the Fourier transform is taken and plotted with the global signal. 

#ver1.4: now using a lomb-scargle diagram to have a much better frequency resolution in
#low frequency. The FFT writes every frequency in multiples of the lowest frequency,
#but the lowest frequency after the phase-folding is f_real, so the resolution at 
#low frequency is terrible. Also changed the * between the noise and the signal for a +
#ver2.0: Fixed an important mistake, signals (real and noise) are of the form
#1 + Asin(2*pi*f*t) not Asin(2*pi*f*t). Also now works even if sample results in an
#uneven split
#ver3.0: Can now make holes in the data (using masks). Also uses the L-S periodogram of gatspy
#ver3.1: Holes in the data are randomly spaced. There is a random number of them and their size is 
#also random. Transits have been added. They have a fixed lenght (transit_len), but their phase is 
#random. Transits are 180 degrees (in the planets orbit) apart from each other. 

#-----------------------------------------------------------
#Inputs
Method = 2                                          #0: We keep all the signal. 1:phase-folding 2:Both
hole_index = 1                                    #0: we don't add holes. 1: we do
T_tot = 3.*365.25
delta_t = 30./(24.*3600.)
N = int(T_tot/delta_t)                                            #Number of sample points  
t = np.linspace(0.0, T_tot, N)                  #time
real_period = np.pi
f_real = 1./real_period                            #Real signal's frequency 
transit_depth = 1.e-3
transit_len = real_period/10.
A_real = transit_depth                              #Real signal's amplitude

N_freq = 1./(2.**(0.5)*10.)                                     #Noise frequency
A_noise = 1.e-2                                    #Noise amplitude
Noise = 1.0 + A_noise*np.sin(N_freq * 2.0*np.pi*t)  #Noise
mean=0.0                                            #Random noise mean
sdev=1.e-3                                        #Random noise standard deviation
F_real = 1.0 + A_real*np.sin(f_real * 2.0*np.pi*t)  #Real signal                                               
period_number = int(N*delta_t*f_real)                  #Total real signal periods in a duration t (needed to phase-fold)

#We fill the random numbers array after a gaussian distribution
N_ran=np.random.normal(mean,sdev,N)
N_ran = np.asarray(N_ran)

#The global percieved signal is computed
signal = F_real+Noise + N_ran

print N
#-----------------------------------------------------------------------------
#This function is useful for finding the closest value in an array.
#Inputs: an array and the value you want to find
#Output: The closest value of the wanted value in the given array
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
#-----------------------------------------------------------------------------

#**************Transits and eclipses parameters***********************
#The function is only used if we phase-fold. 
if Method == 1 or Method == 2:
	indexes = []
	t0 =  real_period*np.random.rand()/2.				#The beginning of the transit. Must be in the first half of the planet's period
	one_period_N = int((N - N % period_number)/period_number)       #The number of points in a single planet period. Points may be removed in order to have an equal division.
	new_t = np.linspace(0.,real_period, one_period_N)               #The time array of a planet period
	transit_pts = int(transit_len/(real_period/one_period_N))       #The number of points in a planet period
	nearest_value = find_nearest(new_t, t0)                         #The nearest value of t0 in the time array
	indexi = np.where(new_t == nearest_value)[0][0]                 #The index of that value
	half_period_pts = int(one_period_N/2.)                          #Number of points in half a planet period
	for i in range(2):                                              #We create a list of the indexes of the time falling the transit and the eclipse
		indexes = indexes + (np.asarray(range(transit_pts))+ indexi + i*half_period_pts).tolist()
	to_mask_transit = np.zeros(one_period_N)                        #used in order to mask the data points in the transit and eclipse
	indexes = np.asarray(indexes)	
	to_mask_transit[indexes[indexes < one_period_N]] = 1                   #We don't consider to points outside the range         
#------------------------------------------------------------------------------

if hole_index == 1:
	#We remove parts of the signal to simulate real data                 #size of each hole
	hole_number = int(period_number*np.random.rand()/2.)                 #random hole number
	hole_size = np.random.rand(hole_number)                              #random size for each hole
	hole_pos = N*delta_t*np.random.rand(hole_number)                     #random position for each hole
	num_pts = hole_size/delta_t                                          #number of points in each hole
	num_pts = num_pts.astype(int)                                        #this number is an integer
	to_mask = np.zeros(N)                                                #mask initialization
	temp =[]
	#Create a list of all the indexes to be masked
	for i in range(hole_number):
		nearest_value = find_nearest(t, hole_pos[i])                 #find the nearest value of the hole position
		indexi = np.where(t == nearest_value)[0][0]                  #find its index
		temp = temp + (np.asarray(range(num_pts[i]))+indexi).tolist()#add the indexes of all the points in the hole in a list
	to_mask[temp] = 1                                                    #all the hole points are to be masked
if hole_index != 0 and hole_index != 1:                                      #failsafe
	sys.exit("hole_index must be either 0 (no holes considered) or 1 (holes considered).")
#------------------------------------------------------------------------

if Method == 0:                                #If method is 0, we don't phase fold             
	#if there are holes, the arrays now have masks
	if hole_index == 1:	
		t=np.ma.array(t,mask=to_mask)
		signal=np.ma.array(signal,mask=to_mask)
	#We choose the frequency range we want to check. An angular frequency is required
	#for the L-S diagram.	
	fmin=0.1*f_real
	fmax=10.*f_real
	Nf=N
	df = (fmax - fmin) / Nf		
	f = 2.0*np.pi*np.linspace(fmin, fmax, Nf)
	
	#We take the L-S of the signal.	
	if hole_index == 0:
		pgram = LombScargleFast().fit(t, signal,sdev)
	elif hole_index == 1:
	#if we consider holes, then the holes mustn't be taken into account for the L-S
		pgram = LombScargleFast().fit(t[~t.mask], signal[~signal.mask],sdev)
	power = pgram.score_frequency_grid(fmin,df,Nf)
	
	#We plot the signal and the L-S
	fig, ax = plt.subplots(2, 1)
	ax[0].plot(t,signal, 'o', ms = 1.5)                           
	ax[0].set_xlabel('Time (days)')
	ax[0].set_ylabel('Flux')
	ax[0].grid()
	ax[1].set_xlim([0.5,6])
	
	#We want the frequency in Hz.
	ax[1].plot(f/2.0/np.pi/f_real, power, 'o')
	ax[1].set_xlabel('Freq (1/orbital period)')
	ax[1].set_ylabel('L-S power')
	ax[1].grid()
	plt.show()
                
#-------------------------------------------------------------------------------

elif Method == 1:     #If Method is 1, we phase fold.                     
	#if there are holes, the arrays now have masks
	if hole_index == 1:		
		t=np.ma.array(t,mask=to_mask)
		signal=np.ma.array(signal,mask=to_mask)
		F_real=np.ma.array(F_real,mask=to_mask) 
		Noise=np.ma.array(Noise,mask=to_mask)
		N_ran=np.ma.array(N_ran,mask=to_mask)	
	
	remove = N % period_number	
	#We split the signal in equal parts the lenght of the real signal's period
	#We discard the last period elements, in case of eneven array division	
	if remove == 0:
		new_signal =np.array_split(signal,period_number)                        
	else:
		new_signal =np.array_split(signal[:-remove],period_number)	
	new_signal_avrg = []	
	
	#The number of points in a single period
	new_N = int((N-remove)/period_number)	
	
	#Because of phase-folding, there are many points along the same x. So, the average is computed.
	if hole_index == 0:
		new_signal_avrg = np.mean(new_signal, axis = 0)
	elif hole_index == 1:
		new_signal_avrg = np.ma.mean(new_signal, axis = 0)
	#Since only one period is considered, we only need the time of one period.                            
	new_t = np.linspace(0.,1./f_real,len(new_signal_avrg))	
	
	#We choose the frequency range we want to check. An angular frequency is required
	#for the L-S diagram.
	fmin=0.1*f_real
	fmax=10.*f_real
	Nf=N
	df = (fmax - fmin) / Nf	
	f = 2.0*np.pi*np.linspace(fmin, fmax, Nf)
	
	#We take the L-S of the signal
	if hole_index == 1:
		mask_tot = new_signal_avrg.mask + to_mask_transit
		mask_tot[mask_tot == 2] = 1	
		new_signal_avrg=np.ma.array(new_signal_avrg,mask=mask_tot)
		pgram_pf = LombScargleFast().fit(new_t[~new_signal_avrg.mask], new_signal_avrg[~new_signal_avrg.mask],sdev)		
	elif hole_index == 0:
		new_signal_avrg=np.ma.array(new_signal_avrg,mask=to_mask_transit)
		pgram_pf = LombScargleFast().fit(new_t[~new_signal_avrg.mask], new_signal_avrg[~new_signal_avrg.mask],sdev)	
	
	power_pf = pgram_pf.score_frequency_grid(fmin,df,Nf)		
	
	#We plot the relevant graphs
	fig, ax = plt.subplots(2, 1)
	ax[0].set_xlim([0,1./f_real])
	ax[0].plot(new_t,new_signal_avrg)
	for i in range(len(new_signal)):
		ax[0].plot(new_t,new_signal[i],'o', ms = 1)                      
	ax[0].set_xlabel('Time (days)')
	ax[0].set_ylabel('Flux')
	ax[0].grid()
	ax[1].set_xlim([0.5,6])
	
	#We want the frequency in Hz and we normalise.	
	ax[1].plot(f/2.0/np.pi/f_real, power_pf, 'o')
	ax[1].set_xlabel('Freq (1/orbital period)')
	ax[1].set_ylabel('L-S power')
	ax[1].set_xlim([fmin/2.0/np.pi/f_real,6])
	ax[1].grid()
	plt.show()


#-----------------------------------------------------------------------------------

elif Method == 2:  #If Method is 2, we do both.       
	if hole_index == 1:
		t=np.ma.array(t,mask=to_mask)
		signal=np.ma.array(signal,mask=to_mask)
		F_real=np.ma.array(F_real,mask=to_mask) 
		Noise=np.ma.array(Noise,mask=to_mask)
		N_ran=np.ma.array(N_ran,mask=to_mask)	
	#We choose the frequency range we want to check. An angular frequency is required
	#for the L-S diagram.	
	fmin=0.1*f_real
	fmax=10.*f_real
	Nf=N
	df = (fmax - fmin) / Nf	
	f = 2.0*np.pi*np.linspace(fmin, fmax, Nf)
	
	#We take the L-S of the signal.	
	if hole_index == 1:
		pgram = LombScargleFast().fit(t[~t.mask], signal[~signal.mask],sdev)
	if hole_index == 0:
		pgram = LombScargleFast().fit(t, signal,sdev)
	power = pgram.score_frequency_grid(fmin,df,Nf)	
	
	#We plot the signal and the L-S
	fig = plt.figure()
	
	ax1 = fig.add_subplot(321)
	ax1.plot(t,F_real, 'o')                           
	ax1.set_xlabel('Time (days)')
	ax1.set_ylabel('Real signal')
	ax1.set_xlim([0,50.])
	ax1.grid()
	ax1.get_yaxis().get_major_formatter().set_useOffset(False)

	ax2 = fig.add_subplot(323)
	ax2.plot(t,Noise, 'o')                           
	ax2.set_xlabel('Time (days)')
	ax2.set_ylabel('Starspots')
	ax2.set_xlim([0,50.])
	ax2.grid()
	ax2.get_yaxis().get_major_formatter().set_useOffset(False)

	ax3 = fig.add_subplot(325)
	ax3.plot(t,N_ran)                           
	ax3.set_xlabel('Time (days)')
	ax3.set_ylabel('Gaussian noise')
	ax3.set_xlim([0,T_tot])
	ax3.grid()
	ax3.get_yaxis().get_major_formatter().set_useOffset(False)

	ax4 = fig.add_subplot(4, 2, 2)
	ax4.plot(t,signal)                           
	ax4.set_xlabel('Time (days)')
	ax4.set_ylabel('Global signal')
	ax4.set_xlim([0,50.])
	ax4.grid()
	ax4.get_yaxis().get_major_formatter().set_useOffset(False)
	
	#We want the frequency in Hz and we normalise.	
	ax5 = fig.add_subplot(4, 2, 4)
	ax5.set_xlim([fmin/2.0/np.pi/f_real,6])
	ax5.plot(f/2.0/np.pi/f_real, power, 'o')
	ax5.set_xlabel('Freq (1/orbital period)')
	ax5.set_ylabel('Power')
	ax5.grid()	
	ax5.get_yaxis().get_major_formatter().set_useOffset(False)

	#****Phase-folding part****
	remove = N % period_number	
	#We split the signal in equal parts the lenght of the real signal's period	
	if remove == 0:
		new_signal =np.array_split(signal,period_number)                     
	else:
		new_signal =np.array_split(signal[:-remove],period_number)	
	new_signal_avrg = []	
	
	
	#The number of points in a single period
	new_N = int((N-remove)/period_number)	
	
	#Because of phase-folding, there are many points along the same x. So, the average is computed.
	if hole_index == 1:
		new_signal_avrg = np.ma.mean(new_signal, axis = 0)
	elif hole_index == 0:
		new_signal_avrg = np.mean(new_signal, axis = 0)
	new_t = np.linspace(0.,1./f_real,len(new_signal_avrg))
	#We choose the frequency range we want to check. An angular frequency is required
	#for the L-S diagram.
	fmin_pf=0.1*f_real
	fmax_pf=10.*f_real
	Nf_pf=N
	df_pf = (fmax_pf - fmin_pf) / Nf_pf	
	f_pf = 2.0*np.pi*np.linspace(fmin_pf, fmax_pf, Nf_pf)
	
	#We take the L-S of the signal.
	if hole_index == 1:
		#we add the masks of the holes and the transits and the eclipse.
		mask_tot = new_signal_avrg.mask + to_mask_transit
		mask_tot[mask_tot == 2] = 1	
		new_signal_avrg=np.ma.array(new_signal_avrg,mask=mask_tot)	
		pgram_pf = LombScargleFast().fit(new_t[~new_signal_avrg.mask], new_signal_avrg[~new_signal_avrg.mask],sdev)		
	elif hole_index == 0:
		new_signal_avrg=np.ma.array(new_signal_avrg,mask=to_mask_transit)
		pgram_pf = LombScargleFast().fit(new_t[~new_signal_avrg.mask], new_signal_avrg[~new_signal_avrg.mask],sdev)
	power_pf = pgram_pf.score_frequency_grid(fmin_pf,df_pf,Nf_pf)
	ax6 = fig.add_subplot(4, 2, 6)	
	ax6.set_xlim([0,1./f_real])
	ax6.plot(new_t[~new_signal_avrg.mask],new_signal_avrg[~new_signal_avrg.mask], linewidth = 5)
	for i in range(len(new_signal)):
		ax6.plot(new_t[~new_signal_avrg.mask],new_signal[i][~new_signal_avrg.mask], 'o', ms = 1)                      
	ax6.set_xlabel('Time (days)')
	ax6.set_ylabel('Global phase-folded signal')
	ax6.grid()
	ax6.get_yaxis().get_major_formatter().set_useOffset(False)
	
	#We want the frequency in Hz and we normalise.
	ax7 = fig.add_subplot(4, 2, 8)	
	ax7.set_xlim([fmin_pf/2.0/np.pi/f_real,6])	
	ax7.plot(f_pf/2.0/np.pi/f_real, power_pf, 'o')
	ax7.set_xlabel('Freq (1/orbital period)')
	ax7.set_ylabel('Power of the phase-folded signal')
	ax7.grid()
	ax7.get_yaxis().get_major_formatter().set_useOffset(False)
	plt.show()
else:
	print "Invalid Method value. Must be 1 if the user doesn't want to phase fold, 0 if he wants to or 2 for both."

	
