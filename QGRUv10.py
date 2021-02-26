# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 01:34:54 2020

@author: onyekpeu
"""
import torch
from torch.nn.parameter      import Parameter
from numpy.random            import RandomState
import numpy as np
from torch.nn import Module
import torch.nn as nn
from torch.autograd import Variable
def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):
    
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features  
        fan_out         = out_features 

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    rng = RandomState(123)
    
    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape) 
    v_i = np.random.uniform(0.0,1.0,number_of_weights)
    v_j = np.random.uniform(0.0,1.0,number_of_weights)
    v_k = np.random.uniform(0.0,1.0,number_of_weights)
    
    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
    	norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
    	v_i[i]/= norm
    	v_j[i]/= norm
    	v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)


def affect_init(r_weight, i_weight, j_weight, k_weight, init_func, rng, init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
    r_weight.size() != k_weight.size() :
         raise ValueError('The real and imaginary weights '
                 'should have the same size . Found: r:'
                 + str(r_weight.size()) +' i:'
                 + str(i_weight.size()) +' j:'
                 + str(j_weight.size()) +' k:'
                 + str(k_weight.size()))

    elif r_weight.dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(r_weight.dim()))
    kernel_size = None
    r, i, j, k  = init_func(r_weight.size(0), r_weight.size(1), rng, kernel_size, init_criterion)
    r, i, j, k  = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)

def unitary_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):
    
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features  
        fan_out         = out_features 

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape) 
    v_r = np.random.uniform(0.0,1.0,number_of_weights)
    v_i = np.random.uniform(0.0,1.0,number_of_weights)
    v_j = np.random.uniform(0.0,1.0,number_of_weights)
    v_k = np.random.uniform(0.0,1.0,number_of_weights)
    
    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i]**2 + v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
        v_r[i]/= norm
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r * s
    weight_i = v_i * s
    weight_j = v_j * s
    weight_k = v_k * s
    return (weight_r, weight_i, weight_j, weight_k)

class QuaternionLinearAutograd(Module):


    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features = in_features#//4
        self.input_dim= in_features
        self.out_features = out_features#//4
        self.out = out_features//3

        self.r_weight, self.i_weight, self.j_weight, self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features)), Parameter(torch.Tensor(self.in_features, self.out_features)), Parameter(torch.Tensor(self.in_features, self.out_features)), Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i2=self.out*2
        self.i3=self.out*3

        if bias is True:
            self.bias = Parameter(torch.Tensor(self.out_features*3))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else 1337
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input,drop):

        
        if self.bias is True:
            self.bias1 = self.bias[:self.out]
            self.bias2 = self.bias[self.out:self.i2]
            self.bias3 = self.bias[self.i2:self.i3]
        
        self.ri_weight = self.r_weight[:,:self.out]
        self.rf_weight = self.r_weight[:,self.out:self.i2]
        self.ra_weight = self.r_weight[:,self.i2:self.i3]

        self.ii_weight = self.i_weight[:,:self.out]
        self.if_weight = self.i_weight[:,self.out:self.i2]
        self.ia_weight = self.i_weight[:,self.i2:self.i3]

        
        self.ji_weight = self.j_weight[:,:self.out]
        self.jf_weight = self.j_weight[:,self.out:self.i2]
        self.ja_weight = self.j_weight[:,self.i2:self.i3]

        
        self.ki_weight = self.k_weight[:,:self.out]
        self.kf_weight = self.k_weight[:,self.out:self.i2]
        self.ka_weight = self.k_weight[:,self.i2:self.i3]

        
        cat_kernels_4_r1 = torch.cat([self.ri_weight,  self.ii_weight, self.ji_weight,  self.ki_weight], dim=0)
        cat_kernels_4_i1 = torch.cat([self.ii_weight,  self.ri_weight, -self.ki_weight, self.ji_weight], dim=0)
        cat_kernels_4_j1 = torch.cat([self.ji_weight,  self.ki_weight, self.ri_weight, -self.ii_weight], dim=0)
        cat_kernels_4_k1 = torch.cat([self.ki_weight,  -self.ji_weight, self.ii_weight, self.ri_weight], dim=0)
        
        cat_kernels_4_r2 = torch.cat([self.rf_weight,  self.if_weight, self.jf_weight,  self.kf_weight], dim=0)
        cat_kernels_4_i2 = torch.cat([self.if_weight,  self.rf_weight, -self.kf_weight, self.jf_weight], dim=0)
        cat_kernels_4_j2 = torch.cat([self.jf_weight,  self.kf_weight, self.rf_weight, -self.if_weight], dim=0)
        cat_kernels_4_k2 = torch.cat([self.kf_weight,  -self.jf_weight, self.if_weight, self.rf_weight], dim=0)
        
        cat_kernels_4_r3 = torch.cat([self.ra_weight,  self.ia_weight, self.ja_weight,  self.ka_weight], dim=0)
        cat_kernels_4_i3 = torch.cat([self.ia_weight,  self.ra_weight, -self.ka_weight, self.ja_weight], dim=0)
        cat_kernels_4_j3 = torch.cat([self.ja_weight,  self.ka_weight, self.ra_weight, -self.ia_weight], dim=0)
        cat_kernels_4_k3 = torch.cat([self.ka_weight,  -self.ja_weight, self.ia_weight, self.ra_weight], dim=0)
        
     
#        wxf, wxi, wxo, wxa
        wi = torch.cat([cat_kernels_4_r1, cat_kernels_4_i1, cat_kernels_4_j1, cat_kernels_4_k1], dim=1)
        
        wf = torch.cat([cat_kernels_4_r2, cat_kernels_4_i2, cat_kernels_4_j2, cat_kernels_4_k2], dim=1)
        
        wa = torch.cat([cat_kernels_4_r3, cat_kernels_4_i3, cat_kernels_4_j3, cat_kernels_4_k3], dim=1)
        if self.bias is True:
            output1 = torch.mm(input, wi)+self.bias1
            output2 = torch.mm(input, wf)+self.bias2
            output3 = torch.mm(input, wa)+self.bias3

        else:

            output1 = torch.mm(input, wi)
            output2 = torch.mm(input, wf)
            output3 = torch.mm(input, wa)
            
#        output4 = torch.mm(input, wo)

        return output1, output2, output3#, output4


    
class QGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):#, CUDA):#CUDA):
        super(QGRU, self).__init__()
        
        # Reading options:
        self.act=nn.Tanh()
        self.act_gate=nn.Sigmoid()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
#        self.CUDA=CUDA

#        self.i2=input_dim*2
#        self.i3=input_dim*3
#        self.h2=hidden_dim*2
#        self.h3=hidden_dim*3
        self.num_classes=output_dim
    
        # Gates initialization
        self.wf  = QuaternionLinearAutograd(self.input_dim, self.hidden_dim*3, bias=True) # Forget

        self.uf  = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim*3, bias=True) # Forget
        

        # Output layer initialization
        self.fco = nn.Linear(self.hidden_dim*4, self.num_classes, bias=True)
#        self.tanh1 = torch.nn.Tanh()
    def forward(self, x, recurrent_dropout):

        self.dropout=nn.Dropout(p=float(recurrent_dropout))
        h = Variable(torch.zeros(x.shape[0], self.hidden_dim*4))#.to(device)#, Variable(torch.zeros(x.shape[0], hidden_dim)).to(device)
#        c = Variable(torch.zeros(x.shape[0], self.hidden_dim*4))#.to(device)

        for k in range(x.shape[1]):
            x_=x[:,k,:]   
#            x=self.dropout(x)
#            c=self.dropout(c)
            h=self.dropout(h)
            wxf, wxi, wxn=(self.wf(x_,recurrent_dropout))
            uxf, uxi, uxn=(self.uf(h,recurrent_dropout))
            rt, zt=self.act_gate(wxf+uxf), self.act_gate(wxi+uxi)#, self.act_gate(wxo+uxo) 
            nt=self.act(wxn+(rt*uxn))
#            nt=self.act(wxn+uxn)
#            c=it*self.act(wxa+uxa)+ft*c
#            h=nt+zt*(h-nt)
            h=zt*h+(1-zt)*nt
#        h__=self.tanh1(h)    
        output = self.fco(h)

     
        return output#h#, c#, h, c#torch.cat(out,0)



class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = QGRU(input_dim, hidden_dim, output_dim) 
#        # Non-linearity 1
#        self.relu1 = torch.nn.ReLU()
#        # Linear function 2: 100 --> 100
#        self.fc2 = torch.nn.Linear(hidden_dim*4, output_dim)
#        # Non-linearity 2
#        self.relu2 = torch.nn.ReLU()
#        self.sigmoid=torch.nn.Sigmoid
#        
#        # Linear function 3: 100 --> 100
#        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
#        # Non-linearity 3
#        self.tanh1 = torch.nn.Tanh()
#        
#        # Linear function 4 (readout): 100 --> 10
#        self.fc4 = torch.nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x0, s):
        # Linear function 1
        x1 = self.fc1(x0, s)

#        # Non-linearity 1
#        x2 = self.fc2(x1)
#        x3=self.sigmoid(x2)

        return x1
def QGRU_fit(x,y,net,input_dim, weight, num_classes, learning_rate, batch_size, epochs, dropout):


    import time
    #net.cuda()   
    net=net#.to(device)   
    criterion = nn.L1Loss()
#        criterion = nn.BCELoss()  
    optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  #torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#        decayRate = 0.96
#        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    leng=len(x)
    n_o_batches=leng/batch_size
    start=time.time()
    kw1=[]
    lss=0.0
#        regularization_loss = 0
#        for param in net.parameters():
#            regularization_loss += torch.sum(abs(param))        
    for epoch in range(epochs):
        
        kw=[]
        running_loss = 0.0
        for k in range(int(np.ceil(leng/(batch_size)))):
            j=k*batch_size
            xin=x[j:j+batch_size]
            yin=y[j:j+batch_size]

            xin=torch.tensor(xin, dtype=torch.float32)
            xin = Variable(xin)
            xin=xin#
            yin=torch.tensor(yin, dtype=torch.float32)
            yin=Variable(yin)

            optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  #torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

            optimizer.zero_grad()  # zero the gradient buffer
#                print(xin.shape)
            outputs = net(xin, dropout)
#                output=torch.round(outputs)

            def exp_decay(epoch, decay_rate):
                lrate=learning_rate*np.exp(-decay_rate*epoch)
                return lrate
#                learning_rate=exp_decay(epoch, decayRate)
            loss = criterion(outputs, yin)
            loss = loss# + l1_ * regularization_loss
            loss.backward(retain_graph=True)
            optimizer.step()
            kw.append(loss.data)
            running_loss =+ loss.item()# * xin.size(0)
     #   loss_values.append(running_loss / len(train_dataset))
            #print('epoch {}/{}, batch {}/{}, [....................], loss {}'.format(epoch, num_epochs, k, round(np.ceil(n_o_batches),0), loss.data))
    #print('epoch {}/{}, batch {}/{}, [....................], loss {}'.format(epoch, num_epochs, k, round(np.ceil(n_o_batches),0), loss.data))
        final_loss=np.mean(kw)
        kw1.append(final_loss)
        print('epoch {}/{}, batch {}/{}, [....................], loss {}'.format(epoch, epochs, k, round(np.ceil(n_o_batches),0), final_loss))
#        lr_scheduler.step()
    #    kw.append(running_loss / leng)
    #plt.plot(kw)
    import matplotlib.pyplot as plt
    plt.plot(kw1)
    plt.show()
    plt.savefig('QGRU_LOSS'+ str('nfr'))
    print('Finished Training')  
    end=time.time()
    Computation_time=end-start

    Run_time=Computation_time
    return net, Run_time

