import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.utils import shuffle
import os.path
from sklearn.metrics import accuracy_score,log_loss,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import librosa
import seaborn as sns
import csv
import cirq
import sympy
import numpy as np
import pandas as pd
import collections
import random

# visualization tools
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from cirq.contrib.svg import SVGCircuit
n_qubits=2
n_layer=1

## See  issue PQC in the middle of network, contd. #267: https://github.com/tensorflow/quantum/issues/267
## Thanks to MichaelBroughton


# Using One-Hot Encoding for Feature cp, fbs and thal? 0 or 1
oh=1
### Generating data
import pandas as pd
import os
das=pd.read_csv('/home/h/Desktop/PodcastFillers.csv')
y=das['label_full_vocab']
wavname=das['clip_name']
folder=das['clip_split_subset']
#y=np.array(y)
#y=np.array([[1,0] if i==1 else [0,1] for i in y])
#df=df.drop(labels='class', axis=1)
ntotal=np.size(y)
print(ntotal)
umlist=([])
k=5000
ummatrix=np.zeros((16000,k))
uhmatrix=np.zeros((16000,k))
brmatrix=np.zeros((16000,k))
agreematrix=np.zeros((16000,k))
musicmatrix=np.zeros((16000,k))
Laughtermatrix=np.zeros((16000,k))
fin=np.zeros(6)
uhlist=([])
brlist=([])
wordlist=([])
agreelist=([])
musiclist=([])
Laughterlist=([])
print(umlist)
t='um'
t1='uh'
umlist.insert(len(umlist),t)
print(umlist)
#uhlist=[]
#breathlist=[]
for i in range (ntotal):
   if y[i]=='Um' and fin[0]<k:
        t=wavname[i]
        umlist.insert(len(umlist),t)
        print(folder[i])
        print(t)
        if folder[i]=='train':
           pathq='/home/h/Desktop/audio/clip_wav/train/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[0]))
              #np.append(ummatrix,np.transpose(sig))
              ummatrix[:,int(fin[0])]=np.transpose(sig)
              fin[0]=fin[0]+1
        if folder[i]=='test':
           pathq='/home/h/Desktop/audio/clip_wav/test/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[0]))
              #np.append(ummatrix,np.transpose(sig))
              ummatrix[:,int(fin[0])]=np.transpose(sig)
              fin[0]=fin[0]+1
        if folder[i]=='extra':
           pathq='/home/h/Desktop/audio/clip_wav/extra/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[0]))
              #np.append(ummatrix,np.transpose(sig))
              ummatrix[:,int(fin[0])]=np.transpose(sig)
              fin[0]=fin[0]+1
   if y[i]=='Uh' and fin[1]<k:
        t=wavname[i]
        uhlist.insert(len(uhlist),t)
        if folder[i]=='train':
           pathq='/home/h/Desktop/audio/clip_wav/train/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[1]))
              #np.append(ummatrix,np.transpose(sig))
              uhmatrix[:,int(fin[1])]=np.transpose(sig)
              fin[1]=fin[1]+1
        if folder[i]=='test':
           pathq='/home/h/Desktop/audio/clip_wav/test/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[1]))
              #np.append(ummatrix,np.transpose(sig))
              uhmatrix[:,int(fin[1])]=np.transpose(sig)
              fin[1]=fin[1]+1
        if folder[i]=='extra':
           pathq='/home/h/Desktop/audio/clip_wav/extra/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[1]))
              #np.append(ummatrix,np.transpose(sig))
              uhmatrix[:,int(fin[1])]=np.transpose(sig)
              fin[1]=fin[1]+1 
   if y[i]=='Breath'and fin[2]<k:
        t=wavname[i]
        brlist.insert(len(brlist),t)
        if folder[i]=='train':
           pathq='/home/h/Desktop/audio/clip_wav/train/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[2]))
              #np.append(ummatrix,np.transpose(sig))
              if np.size(sig)==16000:
                brmatrix[:,int(fin[2])]=np.transpose(sig)
                fin[2]=fin[2]+1
        if folder[i]=='test':
           pathq='/home/h/Desktop/audio/clip_wav/test/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[2]))
              #np.append(ummatrix,np.transpose(sig))
              brmatrix[:,int(fin[2])]=np.transpose(sig)
              fin[2]=fin[2]+1
        if folder[i]=='extra':
           pathq='/home/h/Desktop/audio/clip_wav/extra/'
           if  os.path.exists(os.path.join(pathq, t)):
              file=os.path.join(pathq, t)
              (sig, rate) = librosa.load(file,sr=16000)
              print(sig)
              print(int(fin[2]))
              #np.append(ummatrix,np.transpose(sig))
              brmatrix[:,int(fin[2])]=np.transpose(sig)
              fin[2]=fin[2]+1
   if y[i]=='Music'and fin[3]<k:
        t=wavname[i]
        musiclist.insert(len(musiclist),t)
   if y[i]=='Agree'and fin[4]<k:
        t=wavname[i]
        agreelist.insert(len(agreelist),t)
   if y[i]=='Laughter' and fin[5]<k:
        t=wavname[i]
        Laughterlist.insert(len(Laughterlist),t)

print ('um',np.size(umlist))
print ('uh',np.size(uhlist))
print ('breath',np.size(brlist))
print ('Music',np.size(musiclist))
print ('agree',np.size(agreelist))
print ('Laughter',np.size(Laughterlist))
#print(umlist)
#pd.to_csv("/path/to/file.csv", header=None, index=None)
with open('um.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(umlist)
print(fin[2])
#print(np.size(np.double(ummatrix(1))))
ktrain=500
ktest=150
print((ummatrix[:,ktest]))
FRAME_SIZE = 20 #the values we initially set to our batch executed npy files
HOP_SIZE = 18000
sr = 16000
test_file = librosa.stft(ummatrix[:,1], n_fft=FRAME_SIZE, hop_length=HOP_SIZE) 
Y_test = librosa.power_to_db(np.abs(test_file) ** 2)
print(np.size(Y_test))
umdb=np.zeros([np.size(Y_test),ktrain+ktest])
uhdb=np.zeros([np.size(Y_test),ktrain+ktest])
brdb=np.zeros([np.size(Y_test),ktrain+ktest])
i=0
for i in range(ktrain+ktest):
    test_file = librosa.stft(ummatrix[:,i], n_fft=FRAME_SIZE, hop_length=HOP_SIZE) 
    umdb[:,i]= np.transpose(librosa.power_to_db(np.abs(test_file) ** 2))
    test_file = librosa.stft(uhmatrix[:,i], n_fft=FRAME_SIZE, hop_length=HOP_SIZE) 
    uhdb[:,i]= np.transpose(librosa.power_to_db(np.abs(test_file) ** 2))
    test_file = librosa.stft(brmatrix[:,i], n_fft=FRAME_SIZE, hop_length=HOP_SIZE) 
    brdb[:,i]= np.transpose(librosa.power_to_db(np.abs(test_file) ** 2))
print(brdb)
S0=np.zeros([np.size(Y_test),(ktrain+ktest)])
S1=np.zeros([np.size(Y_test),(ktrain+ktest)])
L=np.zeros([2,2*(ktrain+ktest)])
R=np.zeros(ktrain+ktest)
x_train=np.zeros([np.size(Y_test),ktrain*2])
y_train=np.zeros([2,ktrain*2])
x_test=np.zeros([np.size(Y_test),ktest*2])
y_test=np.zeros([2,ktest*2])

lab1=[0, 1]
lab2=[1, 0]
print(lab1)
for j in range(ktrain+ktest):
   R[j]=int(j)
R1=shuffle(R)
R2=shuffle(R)
print(R)
print(R1)
for j in range(ktrain+ktest):
     S0[:,j]=umdb[:,int(R1[j])]
     S1[:,j]=uhdb[:,int(R2[j])]
for j in range(ktrain):   
     x_train[:,j]=S0[:,j]#umdb[:,int(R1[j])]
     x_train[:,ktrain+j]=S1[:,j]#hdb[:,int(R2[j])]
     y_train[:,j]=np.transpose(lab1)
     y_train[:,ktrain+j]=np.transpose(lab2)
for j in range(ktest):   
     x_test[:,j]=S0[:,ktrain+j]#umdb[:,int(R1[j])]
     x_test[:,ktest+j]=S1[:,ktrain+j]#hdb[:,int(R2[j])]
     y_test[:,j]=lab1
     y_test[:,ktest+j]=lab2
#print(np.size(Y_test))
y_train = np.asarray(y_train)
x_train= np.asarray(x_train)
print('ytrain',np.size(y_train))
y_train=y_train.reshape(2*ktrain,2)
x_train=x_train.reshape(2*ktrain,11)
class SplitBackpropQ(tf.keras.layers.Layer):

    def __init__(self, upstream_symbols, managed_symbols, managed_init_vals,
                 operators):
        """Create a layer that splits backprop between several variables.


        Args:
            upstream_symbols: Python iterable of symbols to bakcprop
                through this layer.
            managed_symbols: Python iterable of symbols to backprop
                into variables managed by this layer.
            managed_init_vals: Python iterable of initial values
                for managed_symbols.
            operators: Python iterable of operators to use for expectation.

        """
        #super().__init__(SplitBackpropQ)
        super(SplitBackpropQ, self).__init__()
        self.all_symbols = upstream_symbols + managed_symbols
        self.upstream_symbols = upstream_symbols
        self.managed_symbols = managed_symbols
        self.managed_init = managed_init_vals
        self.ops = operators

    def build(self, input_shape):
        self.managed_weights = self.add_weight(
            shape=(1, len(self.managed_symbols)),
            initializer=tf.constant_initializer(self.managed_init))

    def call(self, inputs):
        # inputs are: circuit tensor, upstream values
        upstream_shape = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_weights = tf.tile(self.managed_weights, [upstream_shape, 1])
        joined_params = tf.concat([inputs[1], tiled_up_weights], 1)
        return tfq.layers.Expectation()(inputs[0],
                                        operators=ops,
                                        symbol_names=self.all_symbols,
                                        symbol_values=joined_params)
n_layer=4
n_qubits=14
#x_train=([S0[:,0:1500] S1[:,0:1500]])
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]
def Qmodel(n_qubits,n_layer,switch,ent):

    qubits = [cirq.GridQubit(x, y) for x in range(1) for y in range(n_qubits)]
    q_model = cirq.Circuit()
    if ent=="CX":
        if switch==0:
            ## With A2-A2-CNOT-Architecture
            ## Compare: Predicting Toxicity by Quantum Machine Learning
            ## archivx: 2008.07715

            control_params = sympy.symbols('theta:{}'.format(n_qubits))
            control_params1 = sympy.symbols('phi:{}'.format(n_qubits*3*n_layer))
            
            for i in range(0,n_qubits):
                q_model.append(cirq.rz(control_params[i])(qubits[i]))
                q_model.append(cirq.ry(control_params[i])(qubits[i]))

            if n_qubits==2:
                q_model.append(cirq.CX(qubits[0],qubits[1]))
            elif n_qubits>2:
                q_model.append(cirq.CX(qubits[n_qubits-1],qubits[0])) 
                for k in range(0,n_qubits-1):
                    q_model.append(cirq.CX(qubits[k],qubits[k+1]))

            for i in range(0,n_qubits):
                q_model.append(cirq.rz(control_params[i])(qubits[i]))
                q_model.append(cirq.ry(control_params[i])(qubits[i]))

            if n_qubits==2:
                q_model.append(cirq.CX(qubits[0],qubits[1]))
            elif n_qubits>2:
                q_model.append(cirq.CX(qubits[n_qubits-1],qubits[0])) 
                for k in range(0,n_qubits-1):
                    q_model.append(cirq.CX(qubits[k],qubits[k+1]))


            for j in range(0,n_layer):
                for i in range(0,n_qubits):
                    q_model.append(cirq.rx(control_params1[3*i+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.rz(control_params1[3*i+1+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.rx(control_params1[3*i+2+3*j*n_qubits])(qubits[i]))
                if n_qubits==2:
                    q_model.append(cirq.CX(qubits[0],qubits[1]))
                elif n_qubits>2:
                    for k in range(0,n_qubits-1):
                        q_model.append(cirq.CX(qubits[k],qubits[k+1]))
                    q_model.append(cirq.CX(qubits[n_qubits-1],qubits[0]))

        elif switch==1:
            ###
            ### Without data-reuploading
            ###
            control_params = sympy.symbols('theta:{}'.format(n_qubits*3))
            control_params1 = sympy.symbols('phi:{}'.format(n_qubits*3*n_layer))
            for i in range(0,n_qubits):
                q_model.append(cirq.ry(control_params[3*i])(qubits[i]))
                q_model.append(cirq.rz(control_params[3*i+1])(qubits[i]))
                q_model.append(cirq.ry(control_params[3*i+2])(qubits[i]))

            if n_qubits==2:
                q_model.append(cirq.CX(qubits[0],qubits[1]))
            elif n_qubits>2:
                q_model.append(cirq.CX(qubits[n_qubits-1],qubits[0])) 
                for k in range(0,n_qubits-1):
                    q_model.append(cirq.CX(qubits[k],qubits[k+1]))


            for j in range(0,n_layer):
                for i in range(0,n_qubits):
                    q_model.append(cirq.ry(control_params1[3*i+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.rz(control_params1[3*i+1+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.ry(control_params1[3*i+2+3*j*n_qubits])(qubits[i]))
                if n_qubits==2:
                    q_model.append(cirq.CX(qubits[0],qubits[1]))
                elif n_qubits>2:
                    for k in range(0,n_qubits-1):
                        q_model.append(cirq.CX(qubits[k],qubits[k+1]))
                    q_model.append(cirq.CX(qubits[n_qubits-1],qubits[0]))
        elif switch==2:
            ##
            ## With Data Reuploading
            ##

            control_params = sympy.symbols('theta:{}'.format(n_qubits*3))
            control_params1 = sympy.symbols('phi:{}'.format(n_qubits*3*n_layer))
            for j in range(0,n_layer):

                for i in range(0,n_qubits):
                    
                    q_model.append(cirq.ry(control_params[3*i])(qubits[i]))
                    q_model.append(cirq.rz(control_params[3*i+1])(qubits[i]))
                    q_model.append(cirq.ry(control_params[3*i+2])(qubits[i]))

                if n_qubits==2:
                    q_model.append(cirq.CX(qubits[0],qubits[1]))
                elif n_qubits>2:
                    q_model.append(cirq.CX(qubits[n_qubits-1],qubits[0])) 

                for k in range(0,n_qubits-1):
                    q_model.append(cirq.CX(qubits[k],qubits[k+1]))
                
                for i in range(0,n_qubits):
                    
                    q_model.append(cirq.ry(control_params1[3*i+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.rz(control_params1[3*i+1+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.ry(control_params1[3*i+2+3*j*n_qubits])(qubits[i]))
                if n_qubits==2:
                    q_model.append(cirq.CX(qubits[0],qubits[1]))
                elif n_qubits>2:
                    for k in range(0,n_qubits-1):
                        q_model.append(cirq.CX(qubits[k],qubits[k+1]))
                    q_model.append(cirq.CX(qubits[n_qubits-1],qubits[0]))

    if ent=="CZ":
        if switch==0:
            ## With A2-A2-CNOT-Architecture
            ## Compare: Predicting Toxicity by Quantum Machine Learning
            ## archivx: 2008.07715

            control_params = sympy.symbols('theta:{}'.format(n_qubits))
            control_params1 = sympy.symbols('phi:{}'.format(n_qubits*3*n_layer))
            
            for i in range(0,n_qubits):
                q_model.append(cirq.rz(control_params[i])(qubits[i]))
                q_model.append(cirq.ry(control_params[i])(qubits[i]))

            if n_qubits==2:
                q_model.append(cirq.CZ(qubits[0],qubits[1]))
            elif n_qubits>2:
                q_model.append(cirq.CZ(qubits[n_qubits-1],qubits[0])) 
                for k in range(0,n_qubits-1):
                    q_model.append(cirq.CZ(qubits[k],qubits[k+1]))

            for i in range(0,n_qubits):
                q_model.append(cirq.rz(control_params[i])(qubits[i]))
                q_model.append(cirq.ry(control_params[i])(qubits[i]))

            if n_qubits==2:
                q_model.append(cirq.CZ(qubits[0],qubits[1]))
            elif n_qubits>2:
                q_model.append(cirq.CZ(qubits[n_qubits-1],qubits[0])) 
                for k in range(0,n_qubits-1):
                    q_model.append(cirq.CZ(qubits[k],qubits[k+1]))


            for j in range(0,n_layer):
                for i in range(0,n_qubits):
                    q_model.append(cirq.rx(control_params1[3*i+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.rz(control_params1[3*i+1+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.rx(control_params1[3*i+2+3*j*n_qubits])(qubits[i]))
                if n_qubits==2:
                    q_model.append(cirq.CZ(qubits[0],qubits[1]))
                elif n_qubits>2:
                    for k in range(0,n_qubits-1):
                        q_model.append(cirq.CZ(qubits[k],qubits[k+1]))
                    q_model.append(cirq.CZ(qubits[n_qubits-1],qubits[0]))

        elif switch==1:
            ###
            ### Without data-reuploading
            ###
            control_params = sympy.symbols('theta:{}'.format(n_qubits*3))
            control_params1 = sympy.symbols('phi:{}'.format(n_qubits*3*n_layer))
            for i in range(0,n_qubits):
                q_model.append(cirq.ry(control_params[3*i])(qubits[i]))
                q_model.append(cirq.rz(control_params[3*i+1])(qubits[i]))
                q_model.append(cirq.ry(control_params[3*i+2])(qubits[i]))

            if n_qubits==2:
                q_model.append(cirq.CZ(qubits[0],qubits[1]))
            elif n_qubits>2:
                q_model.append(cirq.CZ(qubits[n_qubits-1],qubits[0])) 
                for k in range(0,n_qubits-1):
                    q_model.append(cirq.CZ(qubits[k],qubits[k+1]))


            for j in range(0,n_layer):
                for i in range(0,n_qubits):
                    q_model.append(cirq.ry(control_params1[3*i+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.rz(control_params1[3*i+1+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.ry(control_params1[3*i+2+3*j*n_qubits])(qubits[i]))
                if n_qubits==2:
                    q_model.append(cirq.CZ(qubits[0],qubits[1]))
                elif n_qubits>2:
                    for k in range(0,n_qubits-1):
                        q_model.append(cirq.CZ(qubits[k],qubits[k+1]))
                    q_model.append(cirq.CZ(qubits[n_qubits-1],qubits[0]))
        elif switch==2:
            ##
            ## With Data Reuploading
            ##

            control_params = sympy.symbols('theta:{}'.format(n_qubits*3))
            control_params1 = sympy.symbols('phi:{}'.format(n_qubits*3*n_layer))
            for j in range(0,n_layer):

                for i in range(0,n_qubits):
                    q_model.append(cirq.ry(control_params[3*i])(qubits[i]))
                    q_model.append(cirq.rz(control_params[3*i+1])(qubits[i]))
                    q_model.append(cirq.ry(control_params[3*i+2])(qubits[i]))

                if n_qubits==2:
                    q_model.append(cirq.CZ(qubits[0],qubits[1]))
                elif n_qubits>2:
                    q_model.append(cirq.CZ(qubits[n_qubits-1],qubits[0])) 

                for k in range(0,n_qubits-1):
                    q_model.append(cirq.CZ(qubits[k],qubits[k+1]))

                for i in range(0,n_qubits):
                    q_model.append(cirq.ry(control_params1[3*i+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.rz(control_params1[3*i+1+3*j*n_qubits])(qubits[i]))
                    q_model.append(cirq.ry(control_params1[3*i+2+3*j*n_qubits])(qubits[i]))
                if n_qubits==2:
                    q_model.append(cirq.CZ(qubits[0],qubits[1]))
                elif n_qubits>2:
                    for k in range(0,n_qubits-1):
                        q_model.append(cirq.CZ(qubits[k],qubits[k+1]))
                    q_model.append(cirq.CZ(qubits[n_qubits-1],qubits[0]))                    
                    
                    
#    print(q_model)
    return q_model, control_params, control_params1, qubits

# Specify the quantum circuit:
# qmodel(n_qubit,n_layer,circuit_typ,entanglement)
# With:
# circuit_type= 0 (RyRzCXRyRz...RxRyRy-Encoding); 1 (RyRzRy ...RyRzRy-Encoding); 2 (Same as 1 but with data-reuploading)

q_model, control_params, control_params1, qubits=Qmodel(n_qubits,n_layer,2,"CX")
#Qmodel(12,4,0,"CZ")
#print(q_model)

len(control_params1)
def initiere(control_params1):
    np.random.seed(seed=42)
    int_values=np.random.rand((len(control_params1)))*np.pi
    return int_values
ops=[]
for i in range(n_qubits):
    ops.append(cirq.Z(qubits[i]))
## Definition of the hybrid network

data_input = tf.keras.Input(shape=(x_train.shape[1],), dtype=tf.dtypes.float32)
int_values=initiere(control_params1)

# Emcoding clasical layer
encod = tf.keras.layers.Dense(len(control_params), activation="relu")(data_input)

#encod_do=tf.keras.layers.Dropout(0.5)(encod)

# This is needed because of Note here:
# https://www.tensorflow.org/quantum/api_docs/python/tfq/layers/Expectation
unused = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

expectation = SplitBackpropQ(control_params, control_params1, int_values,
                             ops)([unused, encod])

# Cassical layer for postprocessing
classifier = tf.keras.layers.Dense(2, activation="softmax")

classifier_output = classifier(expectation)

model = tf.keras.Model(inputs=[unused, data_input], outputs=classifier_output)

tf.keras.utils.plot_model(model, show_shapes=True, dpi=70)
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
#              loss=tf.keras.losses.mean_squared_error)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
loss=tf.keras.losses.BinaryCrossentropy(),metrics=METRICS)
              
#model.compile(optimizer='Adam', loss='mse')
# Now we can see 37 parameters. Two of which belong to "SplitBackpropQ"
# since we told it above on L81 that we want it to manage ["beta", "gamma"]
#print(model.trainable_variables)
model.summary()
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


Loops=10
# Define per-fold score containers
acc_per_fold = []
auroc_per_fold = []

ma_precision=[]
ma_recall=[]
ma_f1=[]

wa_precision=[]
wa_recall=[]
wa_f1=[]
wa_precision=[]

len(control_params1)
def initiere(control_params1):
    np.random.seed(seed=42)
    int_values=np.random.rand((len(control_params1)))*np.pi
    return int_values
ops=[]
for i in range(n_qubits):
    ops.append(cirq.Z(qubits[i]))
## Definition of the hybrid network
n=x_train.shape[0]
print('n',n)
data_input = tf.keras.Input(shape=(x_train.shape[1],), dtype=tf.dtypes.float32)
int_values=initiere(control_params1)

# Emcoding clasical layer
encod = tf.keras.layers.Dense(len(control_params), activation="relu")(data_input)

#encod_do=tf.keras.layers.Dropout(0.5)(encod)

# This is needed because of Note here:
# https://www.tensorflow.org/quantum/api_docs/python/tfq/layers/Expectation
unused = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

expectation = SplitBackpropQ(control_params, control_params1, int_values,
                             ops)([unused, encod])

# Cassical layer for postprocessing
classifier = tf.keras.layers.Dense(2, activation="softmax")

classifier_output = classifier(expectation)

model = tf.keras.Model(inputs=[unused, data_input], outputs=classifier_output)

tf.keras.utils.plot_model(model, show_shapes=True, dpi=70)
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
#              loss=tf.keras.losses.mean_squared_error)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
loss=tf.keras.losses.BinaryCrossentropy(),metrics=METRICS)
              
#model.compile(optimizer='Adam', loss='mse')
# Now we can see 37 parameters. Two of which belong to "SplitBackpropQ"
# since we told it above on L81 that we want it to manage ["beta", "gamma"]
#print(model.trainable_variables)
model.summary()
#from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
quantum_tensor=[tfq.convert_to_tensor([q_model for _ in range(3000)]), x_train]

history=model.fit(quantum_tensor,
              y_train, batch_size=10, validation_split=0.1,
              epochs=30,verbose=1)

pd.DataFrame(history.history)[["auc","val_auc"]].plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0., 1) # set the vertical range to [0-1]
plt.show()

Loops=10
# Define per-fold score containers
acc_per_fold = []
auroc_per_fold = []

ma_precision=[]
ma_recall=[]
ma_f1=[]

wa_precision=[]
wa_recall=[]
wa_f1=[]
wa_precision=[]


# Merge inputs and targets

