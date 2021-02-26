# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 03:29:06 2019

@author: onyekpeu
"""

import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vincenty import vincenty



def seq_data_man(data, batch_size, seq_dim, input_dim, output_dim):
    X,Y,Z=data
    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)
    lx=len(X)

    x = []# 
    y = []#
    z=[]
    for i in range(seq_dim,lx):#(timesteps, upperbound in the data)
        x.append(X[i-seq_dim:i, 0:(input_dim)])# append adds elements to the end of the list. i-60:i takes the values from i-60 to i
        y.append(Y[i-1, 0:output_dim])
        z.append(Z[i-1, 0:output_dim])
    x, y, z = np.array(x), np.array(y), np.array(z)
    return (x, y, z)


def sample_freq(data,sf):
    k=[]
    for i in range(sf,len(data),sf):
        s=data[i]
        k.append(s)
    return np.array(k)



def calib1(data1):
    locPred=np.array(data1)
    Acc1=locPred[:,16:17]
    Acc2=locPred[:,17:18]
    gyro1=locPred[:,14:15] 
    Brkpr=locPred[:,26:27] 
    Acc1_bias=np.mean(Acc1,axis=0)
    Acc2_bias=np.mean(Acc2,axis=0)
    gyro1_bias=np.mean(gyro1,axis=0)
    Brkpr_bias=np.mean(Brkpr,axis=0)
    return Acc1_bias, Acc2_bias, gyro1_bias, Brkpr_bias#, sa_bias, sa1_bias, sa2_bias

def normalise(T1, Tmx, Tmn,Z):
    return (Z*(T1-Tmn))/(Tmx-Tmn)

def inv_normalise(N, Tmx, Tmn, Z):
    return (N*(Tmx-Tmn)/Z)+Tmn


def absolute_disp(lat, long):

    k=[]
    for i in range(1, len(lat)):
        lat1=lat[i-1]
        lat2=lat[i]
        lon1=long[i-1]
        lon2=long[i]
        kk=vincenty((lat1,lon1), (lat2, lon2))
#        kk=kk*1609344
        k.append(kk)
    return np.reshape(k,(len(k),1))

def integrate(data, sf):
    dx=sf/10
    arr=[]
    for i in range(1,len(data)):
        y=data[i-2:i]

        intg=np.trapz(y, dx=dx, axis=0)
        arr.append(intg)
    return np.reshape(arr,(len(arr),1))# *1000000


def head_clean(angle):
    for i in range(1,len(angle)):
        if (angle[i]/angle[i-1])>180 or (angle[i-1]/angle[i])>180:
            angle[i]=angle[i-1]
    return angle


def Get_Cummulative(num):
    l=[]
    l.append(num[0])
    for i in range(len(num)-1):
        g=l[i]+num[i+1]
        l.append(g)
    return (np.array(l))#/1000

def diff(data):
    x=[]
    for i in range(1,len(data)):
        a=data[i]-data[i-1]
        x.append(a)
    return np.reshape(x,(len(x),1))
def clean(data):
    value=40
    for i in range(1,len(data)):
        if data[i]>value:
            data[i]=value
        elif data[i]<-value:
            data[i]=-value
    return np.array(data)


            
def get_average(data,avg):
    x=[]
    data=data*1000
#    print(data.shape)
    for i in range(avg, len(data)+avg, avg):
#        print(i)
        a=(np.sum(data[i-avg:i]))/avg
#        print(a)
        x.append(a)
    return (np.reshape(x,(len(x),1)))/1000

def average_vel(init, vel):
    a=(vel[0]+init)/2
    k=np.zeros(len(vel))
#    k=[]
    k[0]=a#.append(a)
    for i in range(1,len(vel)):
        a=(vel[i]+vel[i-1])/2
        k[1]=a
    return np.reshape(k,(len(k),1))
def sample_freq1(data,sf):
    k=[]
#    x=[]
#    print(len(data))
    for i in range(sf,len(data),sf):
#        print (i)
#        print(k.shape)
        k.append(data[i-sf:i])
    s=np.reshape(k,(len(k),sf))
#    print(s.shape)
    return s

def maxmin17(dat_to_norm,sf, Acc1_bias,gyro1_bias):
    locPred=np.array(dat_to_norm)
    # Acc1=locPred[:,16:17]
    dist1=locPred[:,2:3]
    dist2=locPred[:,3:4]
    rl=locPred[:,12:13]
    rr=locPred[:,13:14]
    gy=locPred[:,14:15] 
    dist1=locPred[:,2:3]
    dist2=locPred[:,3:4] 
    rr=sample_freq(rr,sf)
    rl=sample_freq(rl,sf)    
    dist11=sample_freq(dist1,sf)
    dist21=sample_freq(dist2,sf)    
    dista=absolute_disp(dist11, dist21)
    dist=dista*1000  
    r1=np.mean((dist/rr[1:]), axis=0)
    r2=np.mean((dist/rl[1:]), axis=0)  
    rr_=rr*r1#r1#*0.0910
    rl_=rl*r2#r2#*0.0900
    disptl=rl_#integrate(rl,sf)
    disptr=rr_#integrate(rr,sf)        
    dispt1=(disptl+disptr)/2
    return max(rr), min(rr), r1, r2, max(dist), min(dist), max(gy), min(gy)


def data_process13t(dat, seq_di, input_di, output_di,sf, Acc1_bia, Acc2_bia, gyro1_bia, batch_siz, amx, amn, r1, r2, dgmx, dgmn, gymx, gymn, Z, mode):
#    maxm=False
    if mode=='IDNN':
    
        Xin=np.zeros((1,input_di*seq_di))
        Yin=np.zeros((1,output_di))
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
    elif mode=='MLNN':
        Xin=np.zeros((1,input_di))
        Yin=np.zeros((1,output_di)) 
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
    else:
        Xin=np.zeros((1,seq_di, input_di*4))
        Yin=np.zeros((1,output_di))
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
        
    for i in range(len(dat)):
        locPred=np.array(dat[i])
        # Acc1=locPred[:,16:17]
        dist1=locPred[:,2:3]
        dist2=locPred[:,3:4]
 
        
        rl=locPred[:,12:13]
        rr=locPred[:,13:14]
        
        dist11=sample_freq(dist1,sf)
        dist21=sample_freq(dist2,sf)
        dista=absolute_disp(dist11, dist21)
        dist=dista*1000
        
        rr=sample_freq(rr,sf)
        rl=sample_freq(rl,sf)

        rr_=rr*r1
        rl_=rl*r2         

        dispt1=(rl_[1:]+rr_[1:])/2  
   
        rr=normalise(rr[1:], amx, amn, Z)
        rl=normalise(rl[1:], amx, amn, Z)

        dxi1=pd.concat([pd.DataFrame(rr[3:]), pd.DataFrame(rr[2:len(rr)-1]), pd.DataFrame(rr[1:len(rr)-2]), pd.DataFrame(rr[:len(rr)-3])], axis=1)
        dxi2=pd.concat([pd.DataFrame(rl[3:]), pd.DataFrame(rl[2:len(rl)-1]), pd.DataFrame(rl[1:len(rl)-2]), pd.DataFrame(rl[:len(rl)-3])], axis=1)

        xcn=np.concatenate((dxi1[:len(dist)-3], dxi2[:len(dist)-3]), axis=1)  

        # dx3=xcn#dispt1
        XX=xcn
        YY=(dist[3:]-dispt1[3:])[:len(xcn)]
        ZZ=(dispt1[3:])[:len(xcn)]
        

        x,y,z=seq_data_man((XX, YY, ZZ), batch_siz, seq_di, input_di*4, output_di)

        if mode=='MLNN':

            Xin=np.concatenate((Xin,xcn[seq_di:]), axis=0) 
            Yin=np.concatenate((Yin,dist[seq_di:]), axis=0)
            Zin=np.concatenate((Zin,dispt1[seq_di:]), axis=0) 
            Ain=np.concatenate((Ain,dist[seq_di:]), axis=0) 
            Input_data=Xin[1:] 
            Output_data=Yin[1:]
            INS=Zin[1:]
        elif mode=='IDNN':
            x=np.reshape(x,(len(x),seq_di*input_di))
            Xin=np.concatenate((Xin,x), axis=0) 
            Yin=np.concatenate((Yin,y), axis=0)
            Zin=np.concatenate((Zin,z), axis=0) 
            Ain=np.concatenate((Ain,dist), axis=0) 
            Input_data=Xin[1:] 
            Output_data=Yin[1:]
            INS=Zin[1:]
        else:
            Xin=np.concatenate((Xin,x), axis=0) 
            Yin=np.concatenate((Yin,y), axis=0) 
            Zin=np.concatenate((Zin,z), axis=0)  
            Ain=np.concatenate((Ain,dist), axis=0) 
        Input_data=Xin[1:] 
        Output_data=Yin[1:]
        INS=Zin[1:]
        GPS=Ain[1:]
    return  GPS, INS,  Input_data, Output_data



def get_graphv2(s,t, labels, labelt, labelx, labely, labeltitle,no):#s, ins, t=pred
    plt.plot(s, label=labels)#.format(**)+'INS')
    plt.ylabel(labely)
    plt.xlabel(labelx)
    plt.plot(t, label=labelt)
    plt.legend()
    plt.grid(b=True)
#    plt.ylim(0, )
#    plt.xlim(0,len(s))
    plt.title(labeltitle, fontdict={'fontsize': 15, 'fontweight': 500}, loc='center')
#    plt.savefig(labeltitle+ str(no))
    plt.show() 
def position_plot(gpy, gpx, iny, inx, py, px, outage_length):
    plt.plot(gpx[:outage_length],gpy[:outage_length], label='GPS Trajectory', c='black')#, lw=0.01)
    plt.plot(inx[:outage_length],iny[:outage_length], label='INS Estimation', c='red')#, lw=0.01)
    plt.plot(px[:outage_length],py[:outage_length], label='LSTM INS Solution', c='blue')#, lw=0.01)
#    plt.scatter([px,py], label='Proposed Solution', c='blue', lw=0.05)
    plt.xlabel('East displacement [m]')
    plt.ylabel('North displacement [m]')
#    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.show()    
def get_graph(s,t, labels, labelt, labelx, labely, labeltitle,no):#s, ins, t=pred
    plt.plot(np.array([0,1,2,3,4,5,6,7,8,9,10]),np.concatenate((np.zeros((1,1)),s)), label=labels)#.format(**)+'INS')
    plt.ylabel(labely)
    plt.xlabel(labelx)
    plt.plot(np.array([0,1,2,3,4,5,6,7,8,9,10]),np.concatenate((np.zeros((1,1)),t)), label=labelt)#np.array([1,2,3,4,5,6,7,8,9,10])
    plt.legend()
    plt.grid(b=True)
#    plt.ylim(0, )
    plt.xlim(0,len(s)+1)
    print(len(labeltitle))
    plt.title(labeltitle, fontdict={'fontsize': 15, 'fontweight': 500}, loc='center')
    if len(labeltitle)==88:
        labeltitle=('Displacement CRSE for the Sharp Cornering and Successive Left and Right Turns Scenario')
    elif len(labeltitle)==87:
        labeltitle=('Displacement CAE for the Sharp Cornering and Successive Left and Right Turns Scenario')
    if len(labeltitle)==66:
        labeltitle=('Displacement CRSE for the Quick Changes in Acceleration Scenario')
    elif len(labeltitle)==65:
        labeltitle=('Displacement CAE for the Quick Changes in Acceleration Scenario')    
    plt.savefig(labeltitle+ str(no))
    plt.show()  
    
def get_crse(x,y,z,t, label, mode,no):#x=gps, y=ins, z=pred
    eins=np.sqrt(np.power(x,2))#np.sqrt(np.power(x-y,2))
    epred=np.sqrt(np.power(x-z,2))#np.sqrt(np.power(x-z,2))
    crse_ins=Get_Cummulative(eins[:t])
    crse_pred=Get_Cummulative(epred[:t])
#    get_graph(crse_ins, crse_pred, 'INS DR', mode, 'Time (s)', 'CRSE (m)', 'Displacement CRSE for the ' +label,no)
    return crse_ins[-1], crse_pred[-1]

def get_cae(x,y,z,t, label, mode,no):
    eins=x#x-y
    epred=x-z#x-z
    caeins=Get_Cummulative(eins[:t])
    caepred=Get_Cummulative(epred[:t])   
#    get_graph(caeins, caepred, 'INS DR', mode, 'Time (s)', 'CAE (m)', 'Displacement CAE for the ' +label,no)
    return np.sqrt(np.power(caeins[-1],2)), np.sqrt(np.power(caepred[-1],2))         
    
def get_aeps(x,y,z,t, label, mode,no):
    eins=np.sqrt(np.power(x,2))#np.sqrt(np.power(x-y,2))
    epred=np.sqrt(np.power(z,2))#np.sqrt(np.power(x-z,2))
    return (eins/t)[-1], (epred/t)[-1]

def get_perfmetric(cet, cetp):
    mean=np.mean(cet, axis=0)
    mini=np.amin(cet, axis=0) 
    stdv=np.std(cet, axis=0)
    maxi=np.amax(cet, axis=0)
    meanp=np.mean(cetp, axis=0)
    minip=np.amin(cetp, axis=0) 
    stdvp=np.std(cetp, axis=0) 
    maxip=np.amax(cetp, axis=0)  
    perf_metr=np.concatenate((np.reshape(maxi,(1,1)),np.reshape(mini,(1,1)), np.reshape(mean,(1,1)), np.reshape(stdv,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    perf_metrp=np.concatenate((np.reshape(maxip,(1,1)),np.reshape(minip,(1,1)), np.reshape(meanp,(1,1)), np.reshape(stdvp,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    return perf_metr, perf_metrp
def get_dist_covrd(grth):
    mean=np.mean(grth, axis=0)
    mini=np.amin(grth, axis=0) 
    stdv=np.std(grth, axis=0)
    maxi=np.amax(grth, axis=0)
    perf_metr=np.concatenate((np.reshape(maxi,(1,1)),np.reshape(mini,(1,1)), np.reshape(mean,(1,1)), np.reshape(stdv,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    return perf_metr    

  
def predictcs(xthr,ythr, ithr, gthr, regress,  seq_dim,input_di, mode, Ts, mx, mn, Z, label, outage):
    xthr=xthr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    ythr=ythr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    ithr=ithr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    gthr=gthr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    crset=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    crsetwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    caet=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    caetwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aepst=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aepstwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    dist_covrd=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    cccPred=np.zeros((int(outage/Ts),1)) 
    cccins=np.zeros((int(outage/Ts),1)) 
    cccgps=np.zeros((int(outage/Ts),1)) 
           
    for w in range (0,len(xthr),int(outage/Ts)):
        xtest=xthr[w:w+int(outage/Ts)]     
        ytest=ythr[w:w+int(outage/Ts)]
        ins=ithr[w:w+int(outage/Ts)]
        trav=gthr[w:w+int(outage/Ts)]
        yyTest1=np.array(ytest)
        xtest=torch.tensor(xtest, dtype=torch.float32)
        xtest= Variable(xtest)
        newP=regress(xtest,0)    
        newP=newP.detach().numpy()
        newP=np.array(newP)
        crse_ins, crse_pred=get_crse(yyTest1,trav-ins,newP,int(outage/Ts), label, mode,w)
        cae_ins, cae_pred=get_cae(yyTest1,trav-ins,newP,int(outage/Ts), label, mode,w)
        aeps_ins, aeps_pred=crse_ins/int(outage/Ts), crse_pred/int(outage/Ts) #get_aeps(yyTest1,ins,newP,int(outage/Ts), label, mode,w, int(outage/Ts))
        crset[int(w/int(outage/Ts)),0]=float(crse_pred)
        caet[int(w/int(outage/Ts)),0]=float(cae_pred)
        aepst[int(w/int(outage/Ts)),0]=float(aeps_pred)
        crsetwe[int(w/int(outage/Ts)),0]=float(crse_ins)
        caetwe[int(w/int(outage/Ts)),0]=float(cae_ins)
        aepstwe[int(w/int(outage/Ts)),0]=float(aeps_ins)
        dist_covrd[int(w/int(outage/Ts)),0]=float(sum(np.sqrt(np.power(trav,2))))
        newPreds=np.concatenate((cccPred, np.reshape(newP,(len(newP),1))),axis=1)
        cccPred=newPreds
        INS=np.concatenate((cccins, np.reshape(ins,(len(newP),1))),axis=1)
        cccins=INS  
        GPS=np.concatenate((cccgps, np.reshape(ytest,(len(newP),1))),axis=1)
        cccgps=GPS   
    perf_metr_crsep, perf_metr_crsewe=get_perfmetric(crset, crsetwe)
    perf_metr_caep, perf_metr_caewe=get_perfmetric(caet, caetwe)
    perf_metr_aepsp, perf_metr_aepswe=get_perfmetric(aepst, aepstwe)
    dist_travld=get_dist_covrd(dist_covrd)
    return dist_travld, perf_metr_crsep, perf_metr_crsewe, perf_metr_caep, perf_metr_caewe, perf_metr_aepsp, perf_metr_aepswe, newPreds[:,1:], INS[:,1:], GPS[:,1:]#, cet, cetp#opt_runs, cm_runs, cm_runsPred 
   
    
    




