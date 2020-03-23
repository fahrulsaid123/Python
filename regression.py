# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 07:46:41 2020

@author: SAID
"""

import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

class regression:
    """x = np.array([0])
    y = np.array([0])"""
    
    def __init__(self, x, y, w=[[0]]):#constructor
        self.x = np.array(x)#fitur
        self.y = np.array(y)#target
        self.w = np.array(w)#bobot/weight
        self.model = "model"#nama model
    
    def rataX(self):#mendapatkan nilai rata-rata data x(fitur)
        return self.x.mean()
    
    def rataY(self):#mendapatkan nilai rata-rata data y(target)
        return self.y.mean()
    
    def __pangkat(self,x,pangkat):#mendapatkan nilai pangkat n dari x
        return x**pangkat
    
    def __kuadrat(self,x):#mendapatkan nilai pangkat 2 dari x
        return self.__pangkat(x,2)
    
    def __var(self):#mendapatkan nilai varian dari data
        temp = 0
        for i in range(len(self.x)):
            temp += self.__kuadrat(self.x[i]-self.rataX())
        return temp/(len(self.x)-1)
        
    def __cov(self):#mendapatkan nilai covariance dari data
        temp = 0
        for i in range(len(self.x)):
            temp += (self.x[i]-self.rataX())*(self.y[i]-self.rataY())
        return temp/(len(self.x)-1)
    
    def LinearDirect(self):#regression dengan model direct equation
        self.model = "linear"
        temp = [[0],[0]]
        temp[1] = self.__cov()/self.__var()
        temp[0] = self.rataY()-(temp[1]*self.rataX())
        self.w = np.array(temp)
        
    def MultivariateLinear(self):#regression dengan model Multivariate
        self.model = "multivariatelinear"
        one = np.ones((len(self.x), 1))
        self.x = np.append(one, self.x, axis=1)
        xt = np.transpose(self.x)
        self.w = inv(xt.dot(self.x)).dot(xt.dot(self.y))
        
    def SGD(self,epoch,rate,w = []):#regression dengan model Stochastic Gradient Descent
        self.model = "SGD"
        #self.w = np.array([[0.3],[0.2]])
        if len(w)==0:
            self.w = np.random.rand(len(self.x[0])+1,1)
        else:
            self.w = np.array(w)
        print("w :",self.w)
        for i in range(epoch):
            for x in range(len(self.x)):
                print("epoch : ",i+1)
                print("iterasi: ",x+1)
                #fx = (self.w[0]*1)+(self.w[1]*self.x[x][i])
                #print("x: ",self.x[x][i])
                fx = self.prediksi(x)
                temp = fx-self.y[x]
                for j in range(len(self.w)):
                    if j==0:
                        self.w[j] -= rate*(temp)
                    else:
                        self.w[j] -= rate*(temp)*self.x[x][j-1]
                #self.w[0] -= rate*(fx-self.y[x])
                #self.w[1] -= rate*(fx-self.y[x])*self.x[x][i]
                print("w :",self.w)
                print("error :",temp)
                print("----------------")
    
    def BGD(self,epoch,rate,w = []):#regression dengan model Batch Gradient Descent
        self.model = "BGD"
        #self.w = np.array([[0.3],[0.2]])
        if len(w)==0:
            self.w = np.random.rand(len(self.x[0])+1,1)
        else:
            self.w = np.array(w)
        print("w :",self.w)
        for i in range(epoch):
            print("epoch : ",i+1)
            tampung = np.zeros((len(self.w),1))
            for x in range(len(self.x)):
                #print("iterasi: ",x+1)
                #fx = (self.w[0]*1)+(self.w[1]*self.x[x][i])
                #print("x: ",self.x[x][i])
                fx = self.prediksi(x)
                temp = fx-self.y[x]
                for j in range(len(self.w)):
                    if j==0:
                        tampung[j] += temp
                    else:
                        tampung[j] += (temp)*self.x[x][j-1]
                #self.w[0] -= rate*(fx-self.y[x])
                #self.w[1] -= rate*(fx-self.y[x])*self.x[x][i]
            for j in range(len(w)):
                self.w[j] -= rate*(1/len(self.x))*(tampung[j])
            print("w :",self.w)
            print("error :",temp)
            print("----------------")
    
    def Polynomial(self,epoch,rate,derajat,w = []):#Polynomial Regression
        self.model = "polynomial"
        #self.w = np.array([[0.3],[0.2]])
        if len(w)==0:
            self.w = np.random.rand(len(self.x[0])+1,derajat)
            for i in range(1,len(self.w[0])):
                self.w[0][i] = 0
        else:
            self.w = np.array(w)
        print("w :",self.w)
        for i in range(epoch):
            for x in range(len(self.x)):
                print("epoch : ",i+1)
                print("iterasi: ",x+1)
                #fx = (self.w[0]*1)+(self.w[1]*self.x[x][i])
                #print("x: ",self.x[x][i])
                fx = self.prediksi(x)
                temp = fx-self.y[x]
                for j in range(len(self.w)):
                    if j==0:
                        self.w[j][0] -= rate*(temp)*1
                    else:
                        for k in range(len(self.w[j])):
                            self.w[j][k] -= rate*(temp)*self.__pangkat(self.x[x][j-1],k+1)
                #self.w[0] -= rate*(fx-self.y[x])
                #self.w[1] -= rate*(fx-self.y[x])*self.x[x][i]
                print("w :",self.w)
                print("error :",temp)
                print("----------------")
    
    def logistic(self,epoch,rate,w = []):#Logistic Regression
        self.model = "logistic"
        #self.w = np.array([[0.3],[0.2]])
        if len(w)==0:
            self.w = np.random.rand(len(self.x[0])+1,1)
        else:
            self.w = np.array(w)
        print("w :",self.w)
        for i in range(epoch):
            for x in range(len(self.x)):
                print("epoch : ",i+1)
                print("iterasi: ",x+1)
                #fx = (self.w[0]*1)+(self.w[1]*self.x[x][i])
                #print("x: ",self.x[x][i])
                fx = self.prediksi(x)
                temp = ((self.y[x]-fx)*fx)*(1-fx)
                for j in range(len(self.w)):
                    if j==0:
                        self.w[j] += rate*(temp)
                    else:
                        self.w[j] += rate*(temp)*self.x[x][j-1]
                #self.w[0] -= rate*(fx-self.y[x])
                #self.w[1] -= rate*(fx-self.y[x])*self.x[x][i]
                print("w :",self.w)
                print("error :",self.y[x]-fx)
                print("----------------")
    
    def __prediksi(self,baris=0,x=[]):#fungsi prediksi
        if len(x)==0:
            x = self.x
        else:
            x = np.array(x)
        temp = 0
        for i in range(len(self.w)):
            if i==0:
                temp += 1*self.w[i][0]
            else:
                #temp += x[baris][i-1]*self.w[i]
                for j in range(len(self.w[i])):
                    temp += self.__pangkat(x[baris][i-1],j+1)*self.w[i][j]
                #print("x: ",x[baris][i-1])
        return temp
    
    def prediksi(self,baris=0,x=[]):#fungsi modif prediksi
        if self.model == "logistic":
            temp = 1+math.exp(-self.__prediksi(baris,x))
            return 1/temp
        else:
            return self.__prediksi(baris,x)
    
    def Akurasi_RSquare(self):#fungsi akurasi dengan menggunakan model R Square
        SStot = 0
        SSres = 0
        for i in range(len(self.y)):
            SStot += self.__kuadrat(self.y[i]-self.rataY())
            SSres += self.__kuadrat(self.y[i]-self.prediksi(i))
        print("SStot",SStot)
        print("SSres",SSres)
        return 1-(SSres/SStot)
    
    def grafik(self):#fungsi membuat grafik permodelan
        plt.title('Hasil Prediksi Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.plot(self.x,self.y,"k.")
        plt.grid(True)
        dataX=np.linspace(1,7,100)
        dataY = 0
        for i in range(len(self.w)):
            if i==0:
                dataY += self.w[0][0]
            else:
                for j in range(len(self.w[i])):
                    dataY += self.w[i][j]*self.__pangkat(dataX,j+1)
        plt.plot(dataX, dataY, 'r-')
        
    
x = [[1],[2],[4],[3],[5]]
y = [[1],[3],[3],[2],[5]]
"""
#Data Logistic
x = [[3,3],[1,2],[3,4],[1,2],[3,3],[8,3],[5,2],[7,2],[9,0],[8,4]]
y = [[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]]
"""
a = regression(x,y)
"""
print("++++++++++++++Logistic+++++++++++++++++++++")
a.logistic(10,0.3,[[0.0],[0.0],[0.0]])
print("akurasi",a.Akurasi_RSquare())
a.grafik()
print("--------------------------------------")
"""
"""
print("++++++++++++++Polynomial+++++++++++++++++++++")
#Polynomial(self,epoch,rate,derajat,w = [])
a.Polynomial(1000,0.001,2)
print("akurasi",a.Akurasi_RSquare())
a.grafik()
print("--------------------------------------")
"""
"""
print("++++++++++++++Linear Direct+++++++++++++++++++++")
a.LinearDirect()
print("akurasi",a.Akurasi_RSquare())
a.grafik()
print("--------------------------------------")
"""
"""
print("++++++++++++++SGD+++++++++++++++++++++")
a.SGD(100,0.001,[[0.3],[0.2]])
print("akurasi",a.Akurasi_RSquare())
a.grafik()
print("--------------------------------------")
"""
"""
print("++++++++++++++BGD+++++++++++++++++++++")
a.BGD(300,0.001,[[0.3],[0.2]])
print("akurasi",a.Akurasi_RSquare())
a.grafik()
print("--------------------------------------")
"""
"""
print("++++++++++++++Multivariate+++++++++++++++++++++")
a.MultivariateLinear()
print(a.prediksi(x=[[6]]))
a.grafik()
print("--------------------------------------")
"""