# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:03:11 2019

@author: onyekpeu
"""
import tensorflow.keras
import tensorflow as tf
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from keras.optimizers.schedules import ExponentialDecay
print('a')
#from function_filesdispV4kitti import *
from function_fileswsdispqgrupaper2 import *
from IO_VNB_Dataset import *
from QGRUv10 import *
#from kittidataloader import data1
print('d')
#scipy.integrate.cumtrap
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

'Parameters'
#############################################################################
#############################################################################
#############################################################################
#4- cia-3.69, hb-3.06, slr-1.31, wr-2.39
#8 - cia-2.98, hb-3.63, slr-2.16, wr-3.26
#16 - cia-2.92, hb-3.55, slr-1.80, wr-3.41
#32- cia-2.89, hb-3.52, slr-1.58, wr-3.38

#64 - cia-3.72, hb-2.94, slr-1.30, wr-2.09
#128 - cia-3.75, hb-3.13, slr-1.32, wr-2.09
#256 - cia-2.91, hb-3.76, slr-1.36, wr-3.45
par=np.arange(1,2,1)#14
#par=[1, 10, 100, 1000, 10000]
#par=[160, 256, 320, 512, 720, 1024]
#par=[4, 8, 16, 32, 48, 64, 72, 96, 128, 160, 256, 320, 512, 720, 1024]
#par=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]#,200,300,400,500,600,700,800,900]#,1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
#par=[4, 8, 16, 32, 48, 64, 72, 96, 128, 160, 192, 256, 512, 600, 720, 1024] #np.arange(0.01,0.11, 0.01)#14
#par=np.arange(416,1152,32)
opt_runsN=np.zeros((8,len(par),4))
#opt_runsE=np.zeros((len(par),4))
#opt_runs2=np.zeros((len(par),4))
for opt_ in range(len(par)):
    opt=par[opt_]
    dropout=0.005#*int(opt)

    input_dim = 2
    output_dim = 1
    num_epochs = 150#300
    layer_dim = 1
    learning_rate = 0.001#0.00002#*int(opt) #0r 0.001 #or 0.001
    batch_size =1024#int(opt)
    test_split =0# 0.0000001
    decay_rate=0.1#*int(opt)#o.4,0.7
    decay_steps=10000
    momentum=0.8
    samplefreq=10
    Ts=int(samplefreq*1*1)#*int(opt)
    seq_dim=int(2*(10/Ts))#*int(opt)#12 FOR LSTM, 10 FOR GRU,12 for idnn
    seq_dim_=int(seq_dim*Ts)
    avg=1
    l1_=0#.01#*int(opt)#0.1#0.8
    l2_=0#.1#*int(opt)#0#.99#0.7
    h2 = 32#int(opt)
    Z=1
    outage=100

    number_of_runs=1#40
    mode='LSTM'
    '''optimisation order
    1. learning rate done, IDNNDONE, 
    2. decay_rate 
    3. l1 and l2
    4. weights
    5. dropout IDNNDONE, 
    7. sequence length IDNNDONE, 
    8. batch_size
    '''
    
    #############################################################################
    #############################################################################
    #############################################################################
    'Dataloading and indexing'
    #############################################################################
    #############################################################################
    #############################################################################
    TrainDat=[V_Vta2, V_Vta1a, V_Vw5, V_Vta8, V_Vta10, V_Vta16, V_Vta17, V_Vta20, V_Vta21, V_Vta22, V_Vta27, V_Vta28, V_Vta29[:800], V_Vta29[1080:6780], V_Vta29[7220:], V_Vta30[:12900], V_Vta30[13180:], V_Vtb1, V_Vtb2, V_Vtb3, V_Vtb5[:1255], V_Vtb5[1267:3720], V_Vtb5[4160:4380], V_Vtb5[4860:6760], V_Vtb5[7220:],  V_Vw4[:4900], V_Vw4[5760:6220], V_Vw4[7420:33340], V_Vw4[33460:80660], V_Vw4[81000:116180], V_Vw4[117160:], V_Vw14b, V_Vw14c[:14060], V_Vw14c[15600:], V_Vfa01, V_Vfa02[:59860], V_Vfa02[59860:], V_Vfb01a[:1520], V_Vfb01a[1980:5360], V_Vfb01a[5740:9360], V_Vfb01a[11660:], V_Vfb01b, V_Vfb02b]

    Acc1_bias, Acc2_bias, gyro1_bias, Brkpr_bias=calib1(Bias)

    RAdat=[V_Vta11, V_Vfb02d] 
    CIAdat=[V_Vfb02e,V_Vta12]   
    HBdat=[V_Vw16b,V_Vw17,V_Vta9, V_Vta13]
    SLRdat=[V_Vw6, V_Vw8, V_Vw7]
    HRdat=[V_Vw12]
    WRdat=[V_Vtb8, V_Vtb11, V_Vtb13]
#    ra1, ra2, dxmx, dxmn, dymx, dymn= maxmin_wheelspd(locPred42,Ts, gyro1_bias)

    amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn= maxmin17(V_Vw12,Ts, Acc1_bias, gyro1_bias)
    gtr,itr,x, y=data_process13t(TrainDat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)


    gthr,ithr,xthr, ythr=data_process13t( HRdat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
    gtra,itra,xtra, ytra=data_process13t(RAdat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
    gtcia,itcia,xtcia, ytcia=data_process13t(CIAdat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
    gthb,ithb,xthb, ythb=data_process13t(HBdat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn,Z, mode)
    gtslr,itslr,xtslr, ytslr=data_process13t(SLRdat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
    gtwr,itwr,xtwr, ytwr=data_process13t(WRdat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)

    cm_runshr=np.zeros((int(number_of_runs),4))
    cm_runsra=np.zeros((int(number_of_runs),4))
    cm_runscia=np.zeros((int(number_of_runs),4))
    cm_runshb=np.zeros((int(number_of_runs),4))

    cm_runsslr=np.zeros((int(number_of_runs),4)) 
    cm_runswr=np.zeros((int(number_of_runs),4))


    'array creation to store  maximum INS physical model error for each scenario after each full training. '
    cm_runshrdr=np.zeros((int(number_of_runs),4))
    cm_runsradr=np.zeros((int(number_of_runs),4))
    cm_runsciadr=np.zeros((int(number_of_runs),4))
    cm_runshbdr=np.zeros((int(number_of_runs),4))

    cm_runsslrdr=np.zeros((int(number_of_runs),4)) 
    cm_runswrdr=np.zeros((int(number_of_runs),4))

    
    
    newPpredhr=[]
    newPpredcia=[]
    newPpredhb=[] 
    newPpredra=[]     

    newPpredslr=[]  
    newPpredwr=[]

    for nfr in range(number_of_runs):
        print('full training run: '+ str(nfr))
        print('optimisation run: '+ str(opt))
        #############################################################################
        #############################################################################
        #############################################################################
        'GRU TRAINING'
        #############################################################################
        #############################################################################
        #############################################################################

        # Run_time, regress=GRU_model(np.array(x),np.array(y), input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps)
        net = QLSTM(input_dim, h2, output_dim)#, layer_dim, output_dim)
        net, Run_time=QGRU_fit(np.array(x),np.array(y),net,input_dim, h2, output_dim, learning_rate, batch_size, num_epochs, dropout)
        regress=net
  
        dist_travldhr, perf_metrhr_crsep, perf_metrhr_crsedr, perf_metrhr_caep, perf_metrhr_caedr, perf_metrhr_aepsp, perf_metrhr_aepsdr,newPpredshr, inshr, gpshr=predictcs(xthr,ythr, ithr, gthr,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Motorway Scenario',outage)
        dist_travldra, perf_metrra_crsep, perf_metrra_crsedr, perf_metrra_caep, perf_metrra_caedr,perf_metrra_aepsp, perf_metrra_aepsdr,newPpredsra,  insra, gpsra=predictcs(xtra,ytra, itra, gtra,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Roundabout Scenario',outage)
        dist_travldcia, perf_metrcia_crsep, perf_metrcia_crsedr,perf_metrcia_caep, perf_metrcia_caedr,perf_metrcia_aepsp, perf_metrcia_aepsdr,newPpredscia,  inscia, gpscia=predictcs(xtcia,ytcia, itcia, gtcia,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Quick Changes in \n Acceleration Scenario',outage)
        dist_travldhb, perf_metrhb_crsep, perf_metrhb_crsedr,perf_metrhb_caep, perf_metrhb_caedr,perf_metrhb_aepsp, perf_metrhb_aepsdr,newPpredshb,  inshb, gpshb=predictcs(xthb,ythb, ithb, gthb,regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Hard Brake Scenario',outage)
        dist_travldslr, perf_metrslr_crsep, perf_metrslr_crsedr,perf_metrslr_caep, perf_metrslr_caedr,perf_metrslr_aepsp, perf_metrslr_aepsdr,newPpredsslr, insslr, gpsslr=predictcs(xtslr,ytslr, itslr, gtslr,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Sharp Cornering and \n Successive Left and Right Turns Scenario',outage)
        dist_travldwr, perf_metrwr_crsep, perf_metrwr_crsedr,perf_metrwr_caep, perf_metrwrp,perf_metrwr_aepsp, perf_metrwr_aepsdr, newPpredswr,  inswr, gpswr=predictcs(xtwr,ytwr, itwr, gtwr,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Wet Road Scenario',outage)

        newPpredhr.append(newPpredshr)
        newPpredra.append(newPpredsra)   
        newPpredcia.append(newPpredscia)
        newPpredhb.append(newPpredshb)      
        newPpredslr.append(newPpredsslr)  
        newPpredwr.append(newPpredswr)
 

        'indexes the maximum prediction crse across each 10 seconds array'
        cm_runshr[nfr]=perf_metrhr_crsep[:]
        cm_runsra[nfr]=perf_metrra_crsep[:]    
        cm_runscia[nfr]=perf_metrcia_crsep[:]
        cm_runshb[nfr]=perf_metrhb_crsep[:]

        cm_runsslr[nfr]=perf_metrslr_crsep[:]       
        cm_runswr[nfr]=perf_metrwr_crsep[:]
     
        
        'indexes the maximum prediction crse across each 10 seconds array'
        cm_runshrdr[nfr]=perf_metrhr_crsedr[:]
        cm_runsradr[nfr]=perf_metrra_crsedr[:]     
        cm_runsciadr[nfr]=perf_metrcia_crsedr[:]
        cm_runshbdr[nfr]=perf_metrhb_crsedr[:]

        cm_runsslrdr[nfr]=perf_metrslr_crsedr[:]       
        cm_runswrdr[nfr]=perf_metrwr_crsedr[:]


    'indexes the best results across the optimisation runs'       
    a10hr=np.amin(cm_runshr,axis=0)
    a10ra=np.amin(cm_runsra,axis=0)
    a10cia=np.amin(cm_runscia,axis=0)
    a10hb=np.amin(cm_runshb,axis=0)

    a10slr=np.amin(cm_runsslr,axis=0)
    a10swr=np.amin(cm_runswr,axis=0)



    a10hrp=np.amin(cm_runshrdr,axis=0)
    a10rap=np.amin(cm_runsradr,axis=0)
    a10ciap=np.amin(cm_runsciadr,axis=0)
    a10hbp=np.amin(cm_runshbdr,axis=0)

    a10slrp=np.amin(cm_runsslrdr,axis=0)
    a10swrp=np.amin(cm_runswrdr,axis=0)

    
    opt_runsN[0,opt_,:4]=a10hr
    opt_runsN[1,opt_,:4]=a10ra
    opt_runsN[2,opt_,:4]=a10cia
    opt_runsN[3,opt_,:4]=a10hb
    opt_runsN[4,opt_,:4]=a10slr     
    opt_runsN[5,opt_,:4]=a10swr

print(count_parameters(net))    
   
#jj=1
#
#np.savetxt('dispGPS_CSHR_'+mode+'.csv', np.concatenate((np.zeros((1,gpshr.shape[1])), gpshr)), delimiter=',')
#np.savetxt('dispINS_CSHR_'+mode+'.csv', np.concatenate((np.zeros((1,inshr.shape[1])), inshr)), delimiter=',')
#np.savetxt('dispPred_CSHR_'+mode+'.csv', np.concatenate((np.zeros((1,newPpredshr.shape[1])), np.array(newPpredhr[jj]))), delimiter=',')
#
#np.savetxt('dispGPS_CSRA_'+mode+'.csv', np.concatenate((np.zeros((1,gpsra.shape[1])), gpsra)), delimiter=',')
#np.savetxt('dispINS_CSRA_'+mode+'.csv', np.concatenate((np.zeros((1,insra.shape[1])), insra)), delimiter=',')
#np.savetxt('dispPred_CSRA_'+mode+'.csv', np.concatenate((np.zeros((1,newPpredsra.shape[1])), np.array(newPpredra[jj]))), delimiter=',')
#
#np.savetxt('dispGPS_CSCIA_'+mode+'.csv', np.concatenate((np.zeros((1,gpscia.shape[1])), gpscia)), delimiter=',')
#np.savetxt('dispINS_CSCIA_'+mode+'.csv', np.concatenate((np.zeros((1,inscia.shape[1])), inscia)), delimiter=',')
#np.savetxt('dispPred_CSCIA_'+mode+'.csv', np.concatenate((np.zeros((1,newPpredscia.shape[1])), np.array(newPpredcia[jj]))), delimiter=',')
#
#np.savetxt('dispGPS_CSSLR_'+mode+'.csv', np.concatenate((np.zeros((1,gpsslr.shape[1])), gpsslr)), delimiter=',')
#np.savetxt('dispINS_CSSLR_'+mode+'.csv', np.concatenate((np.zeros((1,insslr.shape[1])), insslr)), delimiter=',')
#np.savetxt('dispPred_CSSLR_'+mode+'.csv', np.concatenate((np.zeros((1,newPpredsslr.shape[1])), np.array(newPpredslr[jj]))), delimiter=',')
#
#np.savetxt('dispGPS_CSWR_'+mode+'.csv', np.concatenate((np.zeros((1,gpswr.shape[1])), gpswr)), delimiter=',')
#np.savetxt('dispINS_CSWR_'+mode+'.csv', np.concatenate((np.zeros((1,inswr.shape[1])), inswr)), delimiter=',')
#np.savetxt('dispPred_CSWR_'+mode+'.csv', np.concatenate((np.zeros((1,newPpredswr.shape[1])), np.array(newPpredwr[jj]))), delimiter=',')
#
#np.savetxt('dispGPS_CSS_'+mode+'.csv', np.concatenate((np.zeros((1,gpss.shape[1])), gpss)), delimiter=',')
#np.savetxt('dispINS_CSS_'+mode+'.csv', np.concatenate((np.zeros((1,inss.shape[1])), inss)), delimiter=',')
#np.savetxt('dispPred_CSS_'+mode+'.csv', np.concatenate((np.zeros((1,newPpredss.shape[1])), np.array(newPpreds[jj]))), delimiter=',')
#
#np.savetxt('dispGPS_CSHB_'+mode+'.csv', np.concatenate((np.zeros((1,gpshb.shape[1])), gpshb)), delimiter=',')
#np.savetxt('dispINS_CSHB_'+mode+'.csv', np.concatenate((np.zeros((1,inshb.shape[1])), inshb)), delimiter=',')
#np.savetxt('dispPred_CSHB_'+mode+'.csv', np.concatenate((np.zeros((1,newPpredshb.shape[1])), np.array(newPpredhb[jj]))), delimiter=',')
#
#np.savetxt('dispGPS_CSMS_'+mode+'.csv', np.concatenate((np.zeros((1,gpsms.shape[1])), gpsms)), delimiter=',')
#np.savetxt('dispINS_CSMS_'+mode+'.csv', np.concatenate((np.zeros((1,insms.shape[1])), insms)), delimiter=',')
#np.savetxt('dispPred_CSMS_'+mode+'.csv', np.concatenate((np.zeros((1,newPpredsms.shape[1])), np.array(newPpredms[jj]))), delimiter=',')
