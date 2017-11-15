'''
By: Andrew Castillo
Date: Nov 9, 2017
Description: Class for simple 1-D interpolants
'''
import numpy as np
import basisfuncs
import matplotlib.pyplot as plt

class interp():
    xtrain=[]
    ytrain=[]
    feat=[]
    model=[]
    ftrue=[]

    def polybasis(self):
        deg=self.xtrain.shape[0]-1 #max degree is n-1
        self.feat=basisfuncs.polynomial(deg)

    def rbfbasis(self,l):
        mu=self.xtrain
        self.feat=basisfuncs.rbf(l,mu)
        self.model=np.zeros((len(mu),1)) # initialize model

    def traindata(self,xtrain,ytrain):
        self.xtrain=np.atleast_2d(xtrain).reshape((len(xtrain),1))
        self.ytrain=np.atleast_2d(ytrain).reshape((len(ytrain),1))

    def interp(self):
        X=self.feat(self.xtrain)
        self.model=np.linalg.solve(X,self.ytrain)

    def eval(self,x):
        return(np.dot(self.feat(x),self.model))

    def plot(self):
        test=np.linspace(0,1,100)
        fig3=plt.figure()
        ax3=fig3.add_subplot(111)
        ax3.plot(test,self.eval(test))
        ax3.scatter(self.xtrain,self.ytrain)
        #plt.show()
        return(fig3)

    def plottrue(self):
        if not self.ftrue:
            return('Please Provide True Function, self.ftrue=f(x)')
        test=np.linspace(0,1,100)
        fig3=plt.figure()
        ax3=fig3.add_subplot(111)
        ax3.plot(test,self.eval(test),label='Interp')
        ax3.plot(test,self.ftrue(test),label='True')
        ax3.scatter(self.xtrain,self.ytrain)
        plt.legend()
        #plt.show()
        return(fig3)
