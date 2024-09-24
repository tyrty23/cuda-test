import matplotlib.pyplot as plt
import numpy as np

def read(filename):
    x=[]
    with open(f"../TextFiles/{filename}.txt", 'r') as file:
        NM = file.readline().split(" ")
        N,M=int(NM[0]),int(NM[1])
        for line in file:
            new_line = line.split(" ")
            x.append(complex(float(new_line[0]), float(new_line[1])))
    return x,N,M


plt.subplot(1,2,1)
C,N,M=read("cpu")
real,imag=[],[]
h_n=2/(N-1)
h_m=3/(M-1)
for i in range(len(C)):
    imag.append(-2+h_m*C[i].imag)
    real.append(1-h_n*C[i].real)

plt.scatter(imag,real,s=0.5 color='k')
plt.xlabel("real")
plt.ylabel("imag")
plt.title("cpu")


C,N,M=read("gpu")
real,imag=[],[]
h_n=2/(N-1)
h_m=3/(M-1)
for i in range(len(C)):
    imag.append(-2+h_m*C[i].imag)
    real.append(1-h_n*C[i].real)
plt.subplot(1,2,2)
plt.title("gpu")
plt.scatter(imag,real,s=0.5, color='k')
plt.xlabel("real")
plt.ylabel("imag")
plt.show()