# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:04:33 2022
This script takes one RPPG dataset and asign a label to each subject based on the average HR ground truth measured
with a 15-second sliding window step=0.5.
It works for pre-processed databases where the ground truth is grouped as follows:
.../<subject1>/<subject1>_gt.txt
.../<subject2>/<subject2>_gt.txt
...
.../<subjectN>/<subjectN>_gt.txt
i.e. Subjects are separated by folder and the ground truth file has the same name of the subject's folder

The output is a pandas dataframe saved in pickle format with the name of each subject followed by the HR vector 
measuren on the ground truth, the average of that HR vector and the std. Finally a label related with the HR values.
@author: Deivid
"""

import argparse
import os
from os.path import join
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# detect_peaks function taken from https://github.com/demotu/BMC/blob/master/functions/detect_peaks.py
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------

    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def get_HR(pulseTrace,Fs=30,winLengthSec=15,stepSec=0.5,lowF=0.7,upF=3.5,VERBOSE=0):
    '''
    Parameters
    ----------
    pulseTrace : array
        traces.
    Fs : int, optional
        Frequency of pulse trace. The default is 30.
    winLengthSec : int, optional
        Length of the sliding windows in seconds. The default is 15.        
    stepSec : int, optional
        length of the step to take between windows, in seconds. The default is 0.5. 
    lowF : int, optional
        low frequency for HR measurement. The default is 0.7
    upF : int, optinal
        up frequency for HR measurement. The defaul is 3.5
    Returns
    -------
    HR: Heart rate vector 
    '''
    #IF rppg is exactly winLengthSec, add one more value to get into the function
    if np.size(pulseTrace)<=winLengthSec*Fs:
        if np.size(pulseTrace)<winLengthSec*Fs:
            print('Can not measure metrics because signals is shorter than %i seconds'%winLengthSec)
        elif np.size(pulseTrace)==winLengthSec*Fs:
            pulseTrace = np.append(pulseTrace,pulseTrace[-1]).copy()           
    
    pulseTrace = np.asarray(pulseTrace).copy()
    # CALCULE Timetrace of rPPG with its frequency 
    timeTrace = np.zeros(pulseTrace.size)
    for j in range(0,len(timeTrace)):
        timeTrace[j] = j*(1/Fs)
    
    # Calculate timeTrace of PPG with its frequency
    
    traceSize = len(pulseTrace);
    winLength = round(winLengthSec*Fs)# length of the sliding window for FFT
    step = round(stepSec*Fs);# length of the steps for FFT
    halfWin = (winLength/2);
    
    show1window = True
    HR = []
    cont=0
    for i in range(int(halfWin),int(traceSize-halfWin),int(step)):#for i=halfWin:step:traceSize-halfWin
        #Uncomment next three lines just to debug
        #if cont == 90:
        #    print('error')
        #print(cont);cont=cont+1
        
        ###
        # GET CURRENT WINDOW
        ## get start/end index of current window
        startInd = int(i-halfWin) #startInd = i-halfWin+1;
        endInd = int(i+halfWin) # endInd = i+halfWin;
        startTime = int(timeTrace[startInd]) # startTime = timeTrace(startInd);
        endTime = int(timeTrace[endInd]) #timeTrace(endInd);
        # get current pulse window
        crtPulseWin = pulseTrace[startInd:endInd]# crtPulseWin = pulseTrace(startInd:endInd);
        crtTimeWin = timeTrace[startInd:endInd]# crtTimeWin = timeTrace(startInd:endInd);
        # Fs_PPG = 1/mean(diff(crtTimePPGWin));       
        if VERBOSE>0 and show1window==True: plt.figure(),plt.plot(crtTimeWin,crtPulseWin)
        
        #########################
        # rPPG: SPECTRAL ANALYSIS
        ### rPPG: Get spectrum by Welch
        # Get power spectral density in Frequency of HR in humans [0.7-3.5]
        rppg_freq_w, rppg_power_w = signal.welch(crtPulseWin, fs=Fs)
        rppg_freq2 = [item1 for item1 in rppg_freq_w if item1 > lowF and item1 < upF]
        rppg_power2 = [item2 for item1,item2 in zip(rppg_freq_w,rppg_power_w) if item1 > lowF and item1 < upF]
        rppg_freq_w = rppg_freq2.copy();rppg_power_w = rppg_power2.copy()
        # Find highest peak in the spectral density and take its frequency value
        loc = detect_peaks(np.asarray(rppg_power_w), mpd=1, edge='rising',show=False)
        if loc.size == 0 :# If no peak was found
            loc = np.array([0])
        loc = loc[np.argmax(np.array(rppg_power_w)[loc])]#just highest peak

        rPPG_peaksLoc = np.asarray(rppg_freq_w)[loc]
        if VERBOSE>0 and show1window==True: 
            plt.figure(),plt.title('rPPG Spectrum,welch'),plt.plot(rppg_freq_w,rppg_power_w)
            plt.axvline(x=rPPG_peaksLoc,ymin=0,ymax=1,c='r'),plt.show(),plt.pause(1)
                

        #PPG: HR
        HR.append(rPPG_peaksLoc*60)#rPPG_peaksLoc(1)*60;
        show1window=False # Just plot first window

    return np.asarray(HR)

def main():
    print('================================================================')
    print('                     Set HR labels                              ')
    print('================================================================') 
    
    """""""""
    START ARGPARSE
    """""""""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_labels', type=int, choices=[2,4,6,8,10], default=4,required=False)
    parser.add_argument('--dataset_name', type=str, choices=['GARAGE','UBFC','DATASET'],default='DATASET',required=True)
    parser.add_argument('--data_path', type=str,required=True)
    
    # JOIN ALL ARGUMENTS
    args = parser.parse_args()
    data_path = os.path.abspath(args.data_path)
    
    """""""""
    SHOW PARAMETERES CHOOSEN FOR THIS EXPERIMENT
    """""""""  
    for arg in vars(args):
        print(f'{arg} : {getattr(args, arg)}')
    print('================================================================')
    print('================================================================')  

    """""""""
    LOAD DATA AND MEASURE HR ON GROUND TRUTH SIGNALS
    """""""""      

    folders = natsorted([folder for folder in os.listdir(data_path) if os.path.isdir(join(data_path,folder))])
    subjects_names_list = []
    HR_list = []
    HR_average_list = []
    HR_std_list = []
    
    # Measure HR in all signals individually with a sliding window, then get HR vector and average
    for subject in folders:
        subjects_names_list.append(subject)
        ground_truth = np.loadtxt(join(data_path,subject,subject+'_gt.txt'))
        HR = get_HR(ground_truth,Fs=30,winLengthSec=15,stepSec=0.5,lowF=0.7,upF=3.5,VERBOSE=0)
        HR_list.append(HR)
        HR_average_list.append(np.mean(HR))
        HR_std_list.append(np.std(HR))

    # Create ranges
    HR_ranges = []
    min_val = np.min(HR_average_list)
    step = (np.max(HR_average_list)-min_val)/args.n_labels
    
    for i in range(0,args.n_labels):
        HR_ranges.append([min_val+(i*step),min_val+(i*step)+step])

    HR_ranges[0][0]=0 # add 0 as minimum value just in case 
    HR_ranges[-1][-1]=300 # add 300 as maximum value just in case 

    print(f'[n_labels = {args.n_labels}]=>Generating the following labels on average HR')
    for i in range(0,len(HR_ranges)):
        print(f'{i} if {HR_ranges[i][0]:.2f}<=HR<{HR_ranges[i][1]:.2f}')
        
    # Assign labels
    labels_list = []
    for HR_al in HR_average_list:
        labels_list.append([i for i,crange in enumerate(HR_ranges) if (crange[0]<=HR_al)and(HR_al<crange[1])][0])
        
   
    # Crate PandasDataframe
    DataPath = pd.DataFrame({'subject':subjects_names_list,
                             'HR':HR_list,
                             'HR_mean':HR_average_list,
                             'HR_std':HR_std_list,
                             'label':labels_list})
    # Sae dataframe                         
    DataPath.to_pickle(join(data_path,args.dataset_name+f'_{args.n_labels}HRlabels.pickle'))


if __name__ == "__main__":
    main()
