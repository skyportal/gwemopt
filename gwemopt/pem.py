
# coding: utf-8

# In[81]:


import numpy as np
import numbers
import sys
import scipy.stats
import time

import matplotlib.pyplot as plot
import scipy.spatial as spatial
from scipy import integrate, interpolate, optimize
import pickle
np.core.arrayprint._line_width = 200  

# In[117]:


def Main(T_tot, readout_time, slew_time, PGW, tau, prob, method):
    #Main(T_tot,readout_time,slew_time,PGW,tau,prob)
    #Main is the main body of the optimisation code
    
    #T_tot: total available telescope time, in hrs
    #readout_time : the time needed to perform readout, scalar.
    
    #slew time: the time needed to slew telescope , scalar.
    #           If the time needed to perform both slew and readout is just the larger
    #           of the two, just set one of these parameters equal to the larger one
    #           and the other equal to zero.
    #
    #area_index: A matrix representing the fields and containing the GW
    #            probabilities. an output of function 'fields'. 
     
    #tau: A vector containing the observation tau from 0.1 seconds to 1e5
    #    seconds. An output of Pem.
    
    #prob: A vector containing the values of Pem at each of the values of
    #     tau. An output of Pem.       
    
    #T: The total Observation Time available, in second.
    
    if isinstance(T_tot, numbers.Number) == False:
        sys.exit('Please enter the total observation times, in hours')
        
    nof = len(PGW)
    message1=['The number of fields is ', str(nof), '\n']
    message1 = ''.join(message1)
    print(message1)
    
    numberoffields = np.linspace(1, nof, nof)
    
    tprob = np.zeros(np.count_nonzero(PGW))
    prob_distri = [[] for i in range(np.count_nonzero(PGW))]
    time_allocation = [[] for i in range(np.count_nonzero(PGW))]
    
    print('Start generating the optimal strategy...\n')
    
    
    
    for nos in range(nof):
        T = T_tot-(readout_time+slew_time)*(nos+1);   #This computes the total time available to observation only.
        text = ['The number of fields being considered now is ', str(nos + 1), '.']
        #print(''.join(text))
        if np.count_nonzero(PGW) >= nos + 1:
            time_allocated, individual_prob, probability = Solver(PGW[0 : nos + 1], nos + 1, tau, prob, T, method)
        else:
            text = text1=['Because the sum of the posterior probabilities within the fields from the ',num2str(nos), ' th field onwards are zero, this analysis aborts.'];
            #print(''.join(text))
            break
        tprob[nos] = probability
        prob_distri[nos] = individual_prob
        time_allocation[nos] = time_allocated
        
        text = ['Provided observing ', str(nos), ' fields, the detection probability of the target EM counterpart is ', str(tprob[nos]), '\n'];
        #print(''.join(text)) 
        
        text='The time assigned to each of these fields is as follows:';
        #print(''.join(text)) 
        #print(time_allocation[nos])
        #print('\n')
    
    maxprob = np.amax(tprob)
    maxindex = np.argwhere(tprob == maxprob).squeeze()
    #print('Therefore, based on the analyses above \n')
    text = ['The highest possibility could be obtained by observing ', str(maxindex + 1), ' fields, and the detection probability is ', str(maxprob), '. \n'];
    #print(''.join(text))
    text = ['The time should be allocated as follows: ', str(time_allocation[maxindex][0:-1])]
    #print(''.join(text))
    
    text = ['The exit condition for optimize.root is:', ]
    
    
    plot.plot(numberoffields, tprob)
    plot.xlabel('The total observed fields k')
    plot.ylabel('$P(D_{EM}|k)$')
    plot.show()
    
    return maxprob, time_allocation[maxindex]


# In[96]:


def Pem(lim_mag, lim_time, N_ref = 9.7847e9, L_min = 4.9370e31, L_max = 4.9370e33, model = '', sample_length = None, pMdM = None, tau = None, Loftau = 61, D_mu = 200.0e6, D_sig = 60.0e6, R = None, p_R = None):
    #tau,prob = lim_mag, lim_time, N_ref = 9.7847e9, L_min = 4.9370e31, L_max = 4.9370e33, model = '', sample_length = None, pMdM = None, tau = None, Loftau = 61, D_mu = 200.0e6, D_sig = 60.0e6, R = None, p_R = None)
    #Pem calculates the values of the P_EM integral 
    
    #lim_mag: the limiting magnitude of the telescope. For kilonovae, the
    #         magnitude should be in R-band.

    #lim_time: the limiting time at which the lim_mag is achieved, in
    #          seconds. For example, if a telescope can achieve apparent an R
    #          band magnitude 21 in 3 minutes, then lim_time is 180 seconds,
    #          and lim_mag is 21.

    #N_ref: the number of photons expected at apparent magnitude equal to
    #       zero. The default value is 9.7847*10^9.  

    #L_min: the minimum peak Luminosity of kilonovae, in w.
    #       The default value is 4.970e31w (M = -8) 

    #L_max: the maximum peak Luminosity of kilonovae, in w. 
    #       The default value is 4.970e33w (M = -13) from (Barnes &
    #       Kasen, 2013).
    
    # model: the model that determines whether telescopes will see a target given
    #        a specific observation time. 
    #        'Poisson': Poisson distribution. The function will use a Poisson distribution
    #             to determines how many photons will be seen at a given observation time
    #             lambda is the expected number of photons per second per unit area given 
    #             a peak luminosity
    #        '': The function will use a scaling equation to compute the observation time 
    #           needed to achieve a detection given a magnitude. 
    
    # sample_length: the number of points in distance and magnitudes(luminosity) spaces for 
    #               calculating the integral over magnitude and distance (Eq. 5b in 
    #               DOI:10.3847/1538-4357/834/1/84.
    #               The larger the value, the more accurate the intergral is.
    #               the default value is 100. 
    
    # pMdM: p(M) dM where p(M) is the probability of M. 
    #        A vector the same length as sample_length.
    #        The default is that derived from the prior on peak luminosity ( ~1/sqrt(L) )
    #        If this is a user input value, L_min and L_max will not be used.
    
    # tau: The points in observation time space. 
    
    # Loftau: the number of points is observation time space. The default is 61.
    #         tau will then be defined as tau = 10 ^ (-1 + i * 0.1) with i ranging 
    #         from 0 to Loftau - 1.
    
    # D_mu: This function assumes a gaussian distribution to approximate the distribution on 
    #       distance. D_mu is the mean of the distribution, in par sec.
    
    # D_sig: the variance of the gaussian distribution on distance, in par sec.
    
    # R: A vector in distance space. The interval between the elements should be equal
    #    so that dR = R[1] - R[0]. The default is derived from D_min, D_max equal to 1e7
    #    1e9 par sec.
    
    # p_R: the probability distribution density of R. default is gaussian distribution.
    
    start_time = time.time()
        
    if sample_length == None:
        sample_length = 100
        
    L = np.linspace(L_min, L_max, sample_length)
    logL = np.log10(L)
    dL = L[1] - L[0]
    L_sun = 3.85e26
    logL_sun = np.log10(L_sun)
    M = 4.77 - 2.5 * logL + 2.5 * logL_sun
    
    if pMdM == None:
        k = 1.0 / (2.0 * (np.sqrt(L_max) - np.sqrt(L_min)))
        L_prior = k / np.sqrt(L)
        pMdM = L_prior * dL
    elif len(pMdM) != sample_length:
        sys.exit('pMdm has to have length equal to sample_length.') 
    
    
    if p_R is None:
        D_min = 1.0e7
        D_max = 1.0e9
        Rstepsize = (D_max - D_min) / sample_length
        Rsteps = (D_max - D_min) / Rstepsize
        R = np.linspace(D_min, D_max, int(Rsteps))
        
        p_R = scipy.stats.norm(D_mu, D_sig).pdf(R)   
    else:
        Rstepsize = R[1] - R[0]
        Rsteps = len(R)
    
    if tau is None:    
        tau = np.zeros(Loftau)
        for i in range(Loftau):
            tau[i] = 10 ** (-1 + (i) * 0.1)
    else:
        Loftau = len(tau)
    prob = np.zeros(Loftau)
    N_exp = np.zeros((len(R), sample_length))
    
    
    scaling_num = 10 ** (np.log10(N_ref) - lim_mag / 2.5);
    scaling_flux = lim_time * scaling_num 

    message1='Computing the values of P_EM at tau equal to ';
    message2='This calculation will be repeated ';
    message3=' times.';
    message4='There are still ';
    message5=' times to go. \n';
    
    if model == 'Poisson':
        for i in range(Loftau):
            message = [message1, str(tau[i]), 's.']
            print(''.join(message))
            message = [message2, str(Loftau), message3]
            print(''.join(message))
            message = [message4, str(Loftau - i - 1), message5]
            print(''.join(message))

            for dist in range(len(R)):
                for mag in range(sample_length):
                    N_exp[dist][mag] = 10 ** (np.log10(N_ref) - (M[mag] + 5.0 * np.log10(R[dist]) - 5)/2.5)
                    int_value = (1 - scipy.stats.poisson.cdf(scaling_flux, N_exp[dist][mag] * tau[i])) * p_R[dist] * pMdM[mag] * Rstepsize
                    prob[i] += int_value
    else:
        for i in range(Loftau):
            message = [message1, str(tau[i]), 's.']
            print(''.join(message))
            message = [message2, str(Loftau), message3]
            print(''.join(message))
            message = [message4, str(Loftau - i - 1), message5]
            print(''.join(message))

            for dist in range(len(R)):
                for mag in range(sample_length):
                    N_exp[dist][mag] = 10 ** (np.log10(N_ref) - (M[mag] + 5.0 * np.log10(R[dist]) - 5)/2.5)
                    if N_exp[dist][mag] * tau[i] >= scaling_flux:
                        int_value = p_R[dist] * pMdM[mag] * Rstepsize
                        prob[i] += int_value            

    finish_time = time.time() - start_time
    print(''.join(['The calculation has taken ', str(round(finish_time,2)), ' seconds.']))
 
    return tau, prob

def rotx(angle):
    rotR = np.array([[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]])
    return rotR

def roty(angle):
    rotR = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    return rotR

def rotz(angle):
    rotR = np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return rotR


# In[110]:


def Greedy(fov, n, skymap, resolution, level):
    #Greedy(fov,n,skymap,resolution,level)
    #Greedy finds the locations of the fields using a greedy algorithm;

    #fov       : the size of field of view, in deg^2. In our model, a square
    #            fov is assumed such that the length and the width of the fov
    #            is the square root of the fov.

    #n         : the total number of fields that will be chosen by the greedy
    #            algorithm.

    #skymap    : a string that refers to the name of a txt file. 
    #            GW sky localisation information. The skymap should be a
    #            healpix file translated into a txt file that contains four columns:
    #           1. Index; 2. co-declination; 3. right ascension; 4. GW prob.
    #
    #resolution : a number that defines the resolution of the healpix file.
    #             The number of pixels of a skymap is equal to 12*2^(2*resolution)

    #level : a number that sets the confidence level of the credible region that 
    #        will be considered. The default value if 0.9. To use the default
    #       value, enter [].
 
    #PGW : a vector containing the values of the sum of the GW probability  
    #      within the fields

    #fields_location : a n by 2 matrix containing the location infomration of
    #                  the selected fields. The first coloumn corresponds to
    #                  the declination of the field centers, and the second
    #                  the right ascension.
    DelThe = np.sqrt(fov) * np.pi/180.0
    DelPhi = np.sqrt(fov) * np.pi/180.0
    
    searchrange = np.sqrt(4.0) * np.sqrt(fov) / 2 * np.pi / 180.0
    
    print('Loading the skymap information')
    
    #nos_file = ASDtxt(detector)
    index_str = []
    CoDec_str = []
    Ra_str = []
    GW_prob_str = []
    file = open(skymap, 'r')
    readFile = file.readlines()
    file.close()
    #f = []
    #ASD = []

    for line in readFile:
        p = line.split()
        index_str.append(float(p[0]))
        CoDec_str.append(float(p[1]))
        Ra_str.append(float(p[2]))
        GW_prob_str.append(float(p[3]))
        
    del readFile
    
    index = np.array(index_str)
    CoDec = np.array(CoDec_str)
    Ra = np.array(Ra_str)
    GW_prob = np.array(GW_prob_str)
    
    postinfo = np.array([index, CoDec, Ra, GW_prob])
    
    
    del index, CoDec, Ra, GW_prob
    
    sorted_postinfo = postinfo[:, postinfo[3,:].argsort()[::-1]]
    
    if not level:
        level = 0.9
    elif isinstance(level, numbers.Number) == False: 
        sys.exit('level has to either be an empty array or a number.')     
    
    post = sorted_postinfo[3,:]
    cumulativeSum = np.cumsum(post) / sum(post)
    
    indexthreshold = np.argmax(cumulativeSum >= level)
    
    message1='% Credible Region of the GW Posterior is being considered, and a sky map is being generated. \n';
    display_message=[str(level*100),message1];
    print(''.join(display_message))
    
    sorted_postinfo = sorted_postinfo[:, 0:indexthreshold]
    dec = np.pi / 2.0 - sorted_postinfo[1, :]
    ra = sorted_postinfo[2, :]
    
    ra[np.argwhere(ra > np.pi)] = ra[np.argwhere(ra > np.pi)] - 2.0 * np.pi
    colors = post
    get_ipython().magic(u'matplotlib qt')
    plot.subplot(111, projection='aitoff')
    skymap = plot.scatter(ra, dec, c = post[0:indexthreshold], s = 200)
    plot.colorbar(skymap)
    plot.show()

    
    Prob_in_field = np.zeros(n)
    
    x = np.cos(np.pi / 2.0 - postinfo[1,:]) * np.cos(postinfo[2,:])
    y = np.cos(np.pi / 2.0 - postinfo[1,:]) * np.sin(postinfo[2,:])
    z = np.sin(np.pi / 2.0 - postinfo[1,:])
    
    x1 = np.cos(np.pi / 2.0 - sorted_postinfo[1,:]) * np.cos(sorted_postinfo[2,:])
    y1 = np.cos(np.pi / 2.0 - sorted_postinfo[1,:]) * np.sin(sorted_postinfo[2,:])
    z1 = np.sin(np.pi / 2.0 - sorted_postinfo[1,:])    
    
    Cartesian = np.array([x,y,z])
    
    point_tree = spatial.cKDTree(np.transpose(Cartesian))
    idx = point_tree.query_ball_point(np.transpose(np.array([x1,y1,z1])), searchrange)
    
    num = 1
    PGW = np.zeros(n)
    
    fields_location = np.zeros((n,2))
    
    #margin = np.sqrt(4.0*np.pi/(12.0 * 2 ** (2.0*resolution)))
    
    #micro_movement1=np.array([0, 0, 0, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
    #micro_movement2=np.array([-0.5, 0, 0.5, -0.5, 0, 0.5, -0.5, 0, 0.5]);
    
    calibration = -0.3
    
    location = np.zeros((n, 6))
    resolution_degree = 0
    if resolution < 10:
        check = np.zeros(len(idx))
    while num <= n:
        if num % 10 == 1 and num != 11:
            print(''.join(['Locating the ', str(num), 'st field. Note this does not refer to the observing order.']))
        elif num % 10 == 2 and num != 12:
            print(''.join(['Locating the ', str(num), 'nd field. Note this does not refer to the observing order.']))
        elif num % 10 == 3 and num != 13:
            print(''.join(['Locating the ', str(num), 'rd field. Note this does not refer to the observing order.']))
        else:
            print(''.join(['Locating the ', str(num), 'th field. Note this does not refer to the observing order.']))
        
        P_fov = np.zeros(len(idx))
        filtered_idx = [[] for j in range(len(idx))]
        for i in range(len(idx)):
            phi = sorted_postinfo[1, i]
            theta = sorted_postinfo[2, i]

            filtered = postinfo[:, idx[i]]
            filtered_makezeros = postinfo[:, idx[i]]
            
            newFilter = Cartesian[:, filtered[0,:].astype(int)]
            newCartUp = np.dot(roty(phi - DelPhi/2.0 - np.pi/2.0), np.dot(rotz(theta), newFilter))
            newCartLo = np.dot(roty(phi + DelPhi/2.0 - np.pi/2.0), np.dot(rotz(theta), newFilter))
                
            newCartLeft = np.dot(rotz(DelThe/2.0), np.dot(rotx(np.pi/2 - phi), np.dot(rotz(theta - np.pi/2), newFilter)))
            
            newCartRight = np.dot(rotz(-DelThe/2.0), np.dot(rotx(np.pi/2 - phi), np.dot(rotz(theta - np.pi/2), newFilter)))
            
            
            
            Filter_idx = np.where(np.logical_and.reduce((newCartLeft[0,:]>=0, newCartRight[0,:]<=0,
                                                             newCartUp[2,:] <= 0, newCartLo[2,:] >=0)))
            
            filtered = filtered[:, Filter_idx].squeeze()
            check[i] = np.count_nonzero(filtered[3,:])
            
            P_fov[i] = sum(filtered[3,:])
            
            newCartUp_makezeros = np.dot(roty(phi - DelPhi/2.0 - resolution_degree - np.pi/2.0), np.dot(rotz(theta), newFilter))
            newCartLo_makezeros = np.dot(roty(phi + DelPhi/2.0 + resolution_degree - np.pi/2.0), np.dot(rotz(theta), newFilter))
                
            newCartLeft_makezeros = np.dot(rotz(DelThe/2.0 + resolution_degree), np.dot(rotx(np.pi/2 - phi), np.dot(rotz(theta - np.pi/2), newFilter)))
            newCartRight_makezeros = np.dot(rotz(-DelThe/2.0 - resolution_degree), np.dot(rotx(np.pi/2 - phi), np.dot(rotz(theta - np.pi/2), newFilter)))
    
            Filter_idx_makezeros = np.where(np.logical_and.reduce((newCartLeft_makezeros[0,:]>=0, newCartRight_makezeros[0,:]<=0,
                                                             newCartUp_makezeros[2,:] <= 0, newCartLo_makezeros[2,:] >=0)))
    
            filtered_makezeros = filtered_makezeros[:, Filter_idx_makezeros]
            filtered_idx[i] = filtered_makezeros[0,:]
            
            
        max_id = np.argwhere(P_fov == np.amax(P_fov)).squeeze().astype(int)
        postinfo[3, filtered_idx[max_id].astype(int)] = 0
        Prob_in_field[num-1] = P_fov[max_id]
        declowlim = np.pi / 2.0 - np.amax(postinfo[1, filtered_idx[max_id].astype(int).squeeze()])
        decuplim = np.pi / 2.0 - np.amin(postinfo[1, filtered_idx[max_id].astype(int).squeeze()])
        
        ra_within = postinfo[2, filtered_idx[max_id].astype(int)].squeeze();
        
        ra_within[np.argwhere(ra_within > np.pi)] = ra_within[np.argwhere(ra_within > np.pi)] - 2.0 * np.pi
                             
        ra_uplim = np.amax(ra_within)  
        ra_lowlim = np.amin(ra_within)
        
        ra_lo = sorted_postinfo[2, max_id]
        dec_lo = np.pi/2.0 - sorted_postinfo[1, max_id]
        
        
        location[num - 1, 0] = ra_lowlim
        location[num - 1, 1] = ra_uplim
        location[num - 1, 2] = declowlim
        location[num - 1, 3] = decuplim
        location[num - 1, 4] = ra_lo
        location[num - 1, 5] = dec_lo
         
        if ra_uplim - ra_lowlim < 300.0:
            plot.plot([ra_lowlim, ra_lowlim], [declowlim, decuplim], 'k', linewidth = 2.0)
            plot.plot([ra_lowlim, ra_uplim], [decuplim, decuplim], 'k', linewidth = 2.0)
            plot.plot([ra_uplim, ra_uplim], [declowlim, decuplim], 'k', linewidth = 2.0)
            plot.plot([ra_lowlim, ra_uplim], [declowlim, declowlim], 'k', linewidth = 2.0)
        else:
            plot.plot([ra_lowlim, ra_lowlim], [declowlim, decuplim], 'k', linewidth = 2.0)
            plot.plot([ra_uplim, ra_lowlim], [decuplim, decuplim], 'k', linewidth = 2.0)
            plot.plot([ra_uplim, ra_uplim], [declowlim, decuplim], 'k', linewidth = 2.0)
            plot.plot([ra_uplim, ra_lowlim], [declowlim, declowlim], 'k', linewidth = 2.0)
        
        plot.text(ra_lo, dec_lo, str(num))
        
        num += 1
        
        
            
        
    return location, check, Prob_in_field

def Solver(PGW, nof, tau, prob, T, method):
    #[time_allocated,individual_prob,probability]=Solver(PGW,nos,tau,prob,T)
    #Solver returns the detection probability, and the time allocations 

    #area_index is a vector containing the values of P_GW that is in.
    #descending order of the values of P_GW.

    #nof is the current number of fields being considered.
    #tau, prob are the data for P_EM.

    #T is the total observation time excluding the slew time or readout time 
    #or both.
    
    #time_allocated is the cell array that contains the time allocations.
    #For example, time_allocated{10} is the time allocation for the highest 
    #10 fields in terms of P_GW when only these 10 fields are being observed.

    #individual_prob is the detection prob. probability rendered by an
    #individual field.
    
    prob_a_field = np.zeros(nof)
    
    integral_val_fun = interpolate.splrep(tau, prob, w=1.0*np.ones(len(prob)), s=0)
    
    if method == 'Eq':
        #time_allocated = T/nof * np.ones(nof)
        #Test_solution = 
        time_allocation = T/nof * np.ones(nof)
    elif method == 'LM':
        Test_solution = np.ones(nof + 1)
        time_allocation = np.zeros(nof + 1)
        
        def core(Test_solution):
            integral_val_fun = interpolate.splrep(tau, prob, w=1.0*np.ones(len(prob)), s=0)
            num_var = len(Test_solution)
            dlambda = np.zeros(num_var)

            for i in range(num_var - 1):
                dlambda[i] = PGW[i] * interpolate.splev(Test_solution[i], integral_val_fun, der = 1, ext = 1) + Test_solution[-1]
            dlambda[-1] = sum(Test_solution[0:-1]) - T
            return dlambda
    
    
        time_allocation = optimize.root(core, Test_solution, method = 'lm', options = {'maxiter': 5500, 'ftol': 1e-16, 'xtol': 1e-16 })    
        time_allocation = time_allocation.x
        
    
    
    for i in range(nof):
        prob_a_field[i] = PGW[i] * interpolate.splev(time_allocation[i], integral_val_fun, der = 0, ext = 3)
    
    probability = sum(prob_a_field)           
        
    return time_allocation, prob_a_field, probability






