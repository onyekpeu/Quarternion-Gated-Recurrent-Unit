# -*- coding: utf-8 -*-
"""
Created on Sat May 30 04:52:46 2020

@author: onyekpeu
"""

import pandas as pd
import numpy as np


Bias=pd.read_csv('V-Vw1.csv') 


V_S1=pd.read_csv('V-S1.csv')
V_S2=pd.read_csv('V-S2.csv')
V_S3a=pd.read_csv('V-S3a.csv')
V_S3b=pd.read_csv('V-S3b.csv')
V_S3c=pd.read_csv('V-S3c.csv')
V_S4=pd.read_csv('V-S4.csv')
V_S=np.concatenate((V_S1,V_S2,V_S3a, V_S3b, V_S3c, V_S4), axis=0)

V_M=pd.read_csv('V-M.csv')

V_Y1=pd.read_csv('V-Y1.csv')
V_Y2=pd.read_csv('V-Y2.csv')
V_Y=np.concatenate((V_Y1, V_Y2), axis=0)

V_St1=pd.read_csv('V-St1.csv')
V_St4=pd.read_csv('V-St4.csv')
V_St6=pd.read_csv('V-St6.csv')
V_St7=pd.read_csv('V-St7.csv')
V_St=np.concatenate((V_St1, V_St4, V_St6, V_St7), axis=0)

V_Vta1a=pd.read_csv('V-Vta1a.csv')
V_Vta1b=pd.read_csv('V-Vta1b.csv')
V_Vta2=pd.read_csv('V-Vta2.csv')
V_Vta3=pd.read_csv('V-Vta3.csv')
V_Vta4=pd.read_csv('V-Vta4.csv')
V_Vta5=pd.read_csv('V-Vta5.csv')
V_Vta6=pd.read_csv('V-Vta6.csv')
V_Vta7=pd.read_csv('V-Vta7.csv')
V_Vta8=pd.read_csv('V-Vta8.csv')
V_Vta9=pd.read_csv('V-Vta9.csv')
V_Vta10=pd.read_csv('V-Vta10.csv')
V_Vta11=pd.read_csv('V-Vta11.csv')
V_Vta12=pd.read_csv('V-Vta12.csv')
V_Vta13=pd.read_csv('V-Vta13.csv')
V_Vta14=pd.read_csv('V-Vta14.csv')
V_Vta15=pd.read_csv('V-Vta15.csv')
V_Vta16=pd.read_csv('V-Vta16.csv')
V_Vta17=pd.read_csv('V-Vta17.csv')
V_Vta18=pd.read_csv('V-Vta18.csv')
V_Vta19=pd.read_csv('V-Vta19.csv')
V_Vta20=pd.read_csv('V-Vta20.csv')
V_Vta21=pd.read_csv('V-Vta21.csv')
V_Vta22=pd.read_csv('V-Vta22.csv')
V_Vta23=pd.read_csv('V-Vta23.csv')
V_Vta24=pd.read_csv('V-Vta24.csv')
V_Vta25=pd.read_csv('V-Vta25.csv')
V_Vta26=pd.read_csv('V-Vta26.csv')
V_Vta27=pd.read_csv('V-Vta27.csv')
V_Vta28=pd.read_csv('V-Vta28.csv')
V_Vta29=pd.read_csv('V-Vta29.csv')
V_Vta30=pd.read_csv('V-Vta30.csv')
V_Vta=np.concatenate((V_Vta1a, V_Vta1b, V_Vta2, V_Vta3, V_Vta4, V_Vta5, V_Vta6, V_Vta7, V_Vta8, V_Vta9, V_Vta10, V_Vta11, V_Vta12, V_Vta13, V_Vta14, V_Vta15, V_Vta16, V_Vta17, V_Vta18, V_Vta19, V_Vta20, V_Vta21, V_Vta22, V_Vta23, V_Vta24, V_Vta25, V_Vta26, V_Vta27, V_Vta28, V_Vta29, V_Vta30), axis=0)

V_Vtb1=pd.read_csv('V-Vtb1.csv')
V_Vtb2=pd.read_csv('V-Vtb2.csv')
V_Vtb3=pd.read_csv('V-Vtb3.csv')
V_Vtb4=pd.read_csv('V-Vtb4.csv')
V_Vtb5=pd.read_csv('V-Vtb5.csv')
V_Vtb6=pd.read_csv('V-Vtb6.csv')
V_Vtb7=pd.read_csv('V-Vtb7.csv')
V_Vtb8=pd.read_csv('V-Vtb8.csv')
V_Vtb9=pd.read_csv('V-Vtb9.csv')
V_Vtb10=pd.read_csv('V-Vtb10.csv')
V_Vtb11=pd.read_csv('V-Vtb11.csv')
V_Vtb12=pd.read_csv('V-Vtb12.csv')
V_Vtb13=pd.read_csv('V-Vtb13.csv')
V_Vtb=np.concatenate((V_Vtb1, V_Vtb2, V_Vtb3, V_Vtb4, V_Vtb5, V_Vtb6, V_Vtb7, V_Vtb8, V_Vtb9, V_Vtb10, V_Vtb11, V_Vtb12, V_Vtb13), axis=0)


V_Vw1=pd.read_csv('V-Vw1.csv')
V_Vw2=pd.read_csv('V-Vw2.csv')
V_Vw3=pd.read_csv('V-Vw3.csv')
V_Vw4=pd.read_csv('V-Vw4.csv')
V_Vw5=pd.read_csv('V-Vw5.csv')
V_Vw6=pd.read_csv('V-Vw6.csv')
V_Vw7=pd.read_csv('V-Vw7.csv')
V_Vw8=pd.read_csv('V-Vw8.csv')
V_Vw9=pd.read_csv('V-Vw9.csv')
V_Vw10=pd.read_csv('V-Vw10.csv')
V_Vw11=pd.read_csv('V-Vw11.csv')
V_Vw12=pd.read_csv('V-Vw12.csv')
V_Vw13=pd.read_csv('V-Vw13.csv')
V_Vw14a=pd.read_csv('V-Vw14a.csv')
V_Vw14b=pd.read_csv('V-Vw14b.csv')
V_Vw14c=pd.read_csv('V-Vw14c.csv')
V_Vw15=pd.read_csv('V-Vw15.csv')
V_Vw16a=pd.read_csv('V-Vw16a.csv')
V_Vw16b=pd.read_csv('V-Vw16b.csv')
V_Vw17=pd.read_csv('V-Vw17.csv')
V_Vw=np.concatenate((V_Vw1, V_Vw2, V_Vw3, V_Vw4, V_Vw5, V_Vw6, V_Vw7, V_Vw8, V_Vw9, V_Vw10, V_Vw11, V_Vw12, V_Vw13, V_Vw14a, V_Vw14b, V_Vw14c, V_Vw15, V_Vw16a, V_Vw16b, V_Vw17))

V_Vfa01=pd.read_csv('V-Vfa01.csv')
V_Vfa02=pd.read_csv('V-Vfa02.csv')
V_Vfa=np.concatenate((V_Vfa01, V_Vfa02), axis=0)


V_Vfb01a=pd.read_csv('V-Vfb01a.csv')
V_Vfb01b=pd.read_csv('V-Vfb01b.csv')
V_Vfb01c=pd.read_csv('V-Vfb01c.csv')
V_Vfb01d=pd.read_csv('V-Vfb01d.csv')
V_Vfb02a=pd.read_csv('V-Vfb02a.csv')
V_Vfb02b=pd.read_csv('V-Vfb02b.csv')
V_Vfb02c=pd.read_csv('V-Vfb02c.csv')
V_Vfb02d=pd.read_csv('V-Vfb02d.csv')
V_Vfb02e=pd.read_csv('V-Vfb02e.csv')
V_Vfb02f=pd.read_csv('V-Vfb02f.csv')
V_Vfb02g=pd.read_csv('V-Vfb02g.csv')
V_Vfb=np.concatenate((V_Vfb01a, V_Vfb01b, V_Vfb01c, V_Vfb01d, V_Vfb02a, V_Vfb02b, V_Vfb02c, V_Vfb02d, V_Vfb02e, V_Vfb02f, V_Vfb02g), axis=0)

all_data=np.concatenate((V_S, V_M, V_Y, V_St, V_Vta, V_Vtb, V_Vw, V_Vfa, V_Vfb), axis=0)
