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
Method = 2                                                  #0: We keep all the signal. 1:phase-folding 2:Both
hole_index = 1                                              #0: we don't add holes. 1: we do
bin_index = 1						    #0: We don't bin the data for ploting. 1#: We do 
bin_number = 10000                                          #number of bins if binning is done
T_tot = 3.*365.25                              		    #Total time considered (days)
delta_t = 30./(24.*3600.)			            #time spacing
N = int(T_tot/delta_t)                                      #Number of sample points  
t = np.linspace(0.0, T_tot, N)                              #time
real_period = np.pi                                         #planet's signal period
f_real = 1./real_period                                     #planet's signal frequency 
transit_len = real_period/10.                               #transit lenght
A_real = 1.e-3                                              #planet's signal amplitude
N_freq = 1./(2.**(0.5)*10.)                                 #Starspots frequency
A_noise = 1.e-2                                             #Starspots amplitude
Noise = 1.0 + A_noise*np.sin(N_freq * 2.0*np.pi*t)          #starspots
mean=0.0                                                    #Random noise mean
sdev=1.e-3                                                  #Random noise standard deviation
F_real = 1.0 + A_real*np.sin(f_real * 2.0*np.pi*t)          #planet signal                                               
period_number = int(T_tot*f_real)                           #Total planet periods in a duration t (needed to phase-fold)
periods = 7                                     #Number of planet periods we phase-fold on
#We fill the random numbers array after a gaussian distribution
N_ran=np.random.normal(mean,sdev,N)
N_ran = np.asarray(N_ran)

#The global percieved signal is computed
signal = F_real+Noise + N_ran

print "N: ", N, "Total time: ", T_tot, " days ", "Period_number: ", period_number

#-----------------------------------------------------------------------------

#**************Transits and eclipses parameters***********************
#The function is only used if we phase-fold. 
#Inputs: new_t: time array
#periods: number of periods of the real signal we want to put eclipses/transits on
def transit(new_t,periods): 
	indexes = []
	one_period_N = len(new_t)
	t0 =  real_period*np.random.rand()/2.				#The beginning of the transit. Must be in the first half of the planet's period
	index_it0 = (np.abs(new_t-t0)).argmin()                            #Its index
	index_transit_end = (np.abs(new_t- t0 - transit_len)).argmin()     #The index of the end of the transit
	index_half_period_later = (np.abs(new_t- t0 - real_period/2.)).argmin()    #The index half a period later
	half_period_pts = index_half_period_later - index_it0                      #number of points in half a period
	transit_pts = index_transit_end - index_it0                                 #number of points in a transit
	for i in range(2*periods):                                              #We create a list of the indexes of the time falling the transit and the eclipse
		indexes = indexes + (np.asarray(range(transit_pts))+ index_it0 + i*half_period_pts).tolist()
	to_mask_transit = np.zeros(one_period_N)                        #used in order to mask the data points in the transit and eclipse
	indexes = np.asarray(indexes)		
	to_mask_transit[indexes[indexes < one_period_N]] = 1                  #We don't consider to points outside the range     
	return to_mask_transit      
#------------------------------------------------------------------------------

if hole_index == 1:
	#We remove parts of the signal to simulate real data                 #size of each hole
	hole_number = int(period_number*np.random.rand()/2.)                 #random hole number
	hole_size = np.random.rand(hole_number)                              #random size for each hole
	hole_pos = T_tot*np.random.rand(hole_number)                     #random position for each hole
	num_pts = hole_size/delta_t                                          #number of points in each hole
	num_pts = num_pts.astype(int)                                        #this number is an integer
	to_mask = np.zeros(N)                                                #mask initialization
	temp =[]
	#Create a list of all the indexes to be masked
	for i in range(hole_number):
		indexi = (np.abs(t-hole_pos[i])).argmin()                  #find its index
		temp = temp + (np.asarray(range(num_pts[i]))+indexi).tolist()#add the indexes of all the points in the hole in a list
	to_mask[temp[temp < N]] = 1                                          #all the hole points are to be masked
if hole_index != 0 and hole_index != 1:                                      #failsafe
	sys.exit("hole_index must be either 0 (no holes considered) or 1 (holes considered).")
#------------------------------------------------------------------------
if bin_index != 0 and bin_index != 1:                                      #failsafe
	sys.exit("bin_index must be either 0 (no binning for the plot considered) or 1 (binning considered).")
#------------------------------------------------------------------------
if Method == 0:                                #If method is 0, we don't phase fold             
	#if there are holes, the arrays now have masks
	if hole_index == 1:	
		t=np.ma.array(t,mask=to_mask)
		signal=np.ma.array(signal,mask=to_mask)
	#We choose the frequency range we want to check. An angular frequency is required
	#for the L-S diagram.	
	fmin=0.1*min(f_real,N_freq)
	fmax=10.*f_real
	Nf= 20000
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
	
	
	#We want the frequency in Hz.
	ax[1].set_xlim([fmin/2.0/np.pi/f_real,6])
	ax[1].plot(f/2.0/np.pi/f_real, power, 'o')
	ax[1].set_xlabel('Freq (1/orbital period)')
	ax[1].set_ylabel('L-S power')
	ax[1].grid()
	plt.show()
                
#-------------------------------------------------------------------------------

elif Method == 1:     #If Method is 1, we phase fold.                     
	#if there are holes, the arrays now have masks
	if hole_index == 1:		
		t = np.ma.array(t,mask=to_mask)
		signal= np.ma.array(signal,mask=to_mask)
		F_real = np.ma.array(F_real,mask=to_mask) 
		Noise = np.ma.array(Noise,mask=to_mask)
		N_ran = np.ma.array(N_ran,mask=to_mask)
	
	t = np.ma.mod(t,periods*real_period)                                     #We phase-fold
	data = np.ma.column_stack((t,signal))                                    #We make a 2D array of the time/signal
	data = data[np.lexsort((data[:,1],data[:,0]))]	                         #We sort the points
	to_mask_transit = transit(data[:,0],int(periods))                        #We find the transits/eclipses
	
	#We choose the frequency range we want to check. An angular frequency is required
	#for the L-S diagram.
	fmin_pf = 0.1*min(f_real,N_freq)
	fmax_pf = 10.*f_real
	Nf_pf = 20000
	df_pf = (fmax_pf - fmin_pf) / Nf_pf	
	f_pf = 2.0*np.pi*np.linspace(fmin_pf, fmax_pf, Nf_pf)
	
	#We take the L-S of the signal
	if hole_index == 1:
		mask_tot = data[:,0].mask + to_mask_transit                      #We compute the total mask
		mask_tot[mask_tot == 2] = 1	                                 #Two holes are still one hole!
		data[:,0] = np.ma.array(data[:,0], mask = mask_tot)              
		t_eff = data[:,0][~data[:,0].mask]
		signal_eff = data[:,1][~data[:,0].mask]                          
		pgram_pf = LombScargleFast().fit(t_eff, signal_eff,sdev)		
	elif hole_index == 0:
		data = np.ma.array(data, mask = to_mask_transit)
		t_eff = data[:,0][~data[:,0].mask]
		signal_eff = data[:,1][~data[:,0].mask]
		pgram_pf = LombScargleFast().fit(t_eff, signal_eff,sdev)	
	power_pf = pgram_pf.score_frequency_grid(fmin_pf,df_pf,Nf_pf)			


	#We plot the relevant graphs
	fig, ax = plt.subplots(2, 1)
	ax[0].set_xlim([0,periods*real_period])
	if bin_index == 1:
		bins = np.linspace(0,periods*real_period,bin_number)
		digitized = np.digitize(t_eff,bins)
		bin_means_t = []
		bin_means_sign = []
		for i in range(1, len(bins)):
			bin_means_t.append(t_eff[digitized == i].mean())
			bin_means_sign.append(signal_eff[digitized == i].mean())
		bin_means_t = np.asarray(bin_means_t) 
		bin_means_sign = np.asarray(bin_means_sign)
		ax[0].plot(bin_means_t,bin_means_sign)                    
	elif bin_index == 0:
		ax[0].plot(t_eff, signal_eff)                    
	ax[0].set_xlabel('Time')
	ax[0].set_ylabel('Flux')
	ax[0].grid()
	
	
	#We want the frequency in Hz and we normalise.
	ax[1].set_xlim([fmin_pf/2.0/np.pi/f_real,6])	
	ax[1].plot(f/2.0/np.pi/f_real, power_pf, 'o')
	ax[1].set_xlabel('Freq (1/orbital period)')
	ax[1].set_ylabel('L-S power')
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
	fmin=0.1*min(f_real,N_freq)
	fmax=10.*f_real
	Nf=20000
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

	#***************************Phase-folding part**************************
	t = np.ma.mod(t,periods*real_period)                                            #Phase-folding
	data = np.ma.column_stack((t,signal))                                           #2D array of t,signal
	data = data[np.lexsort((data[:,1],data[:,0]))]	                                #we sort it
	to_mask_transit = transit(data[:,0],int(periods))                               #We find the transits/eclipses
	#We choose the frequency range we want to check. An angular frequency is required
	#for the L-S diagram.
	fmin_pf=0.1*min(f_real,N_freq)
	fmax_pf=10.*f_real
	Nf_pf=20000
	df_pf = (fmax_pf - fmin_pf) / Nf_pf	
	f_pf = 2.0*np.pi*np.linspace(fmin_pf, fmax_pf, Nf_pf)
	
	#We take the L-S of the signal
	if hole_index == 1:
		mask_tot = data[:,0].mask + to_mask_transit
		mask_tot[mask_tot == 2] = 1	
		data[:,0] = np.ma.array(data[:,0], mask = mask_tot)
		t_eff = data[:,0][~data[:,0].mask]
		signal_eff = data[:,1][~data[:,0].mask]
		pgram_pf = LombScargleFast().fit(t_eff, signal_eff,sdev)		
	elif hole_index == 0:
		data = np.ma.array(data, mask = to_mask_transit)
		t_eff = data[:,0][~data[:,0].mask]
		signal_eff = data[:,1][~data[:,0].mask]
		pgram_pf = LombScargleFast().fit(t_eff, signal_eff,sdev)	
	
	power_pf = pgram_pf.score_frequency_grid(fmin_pf,df_pf,Nf_pf)		
	
	#We plot the relevant graphs
	ax6 = fig.add_subplot(4, 2, 6)	
	ax6.set_xlim([0,periods*real_period])
	if bin_index == 1:
		bins = np.linspace(0,periods*real_period, bin_number)
		digitized = np.digitize(t_eff,bins)
		bin_means_t = []
		bin_means_sign = []
		for i in range(1, len(bins)):
			bin_means_t.append(t_eff[digitized == i].mean())
			bin_means_sign.append(signal_eff[digitized == i].mean())
		bin_means_t = np.asarray(bin_means_t) 
		bin_means_sign = np.asarray(bin_means_sign)
		ax6.plot(bin_means_t,bin_means_sign)                    
	elif bin_index == 0:
		ax6.plot(t_eff, signal_eff)
	ax6.set_xlabel('Time (days)')
	ax6.set_ylabel('Global phase-folded signal')
	ax6.grid()
	ax6.get_yaxis().get_major_formatter().set_useOffset(False)
	
	#We want the frequency in Hz
	print len(f_pf),len(power_pf)
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

