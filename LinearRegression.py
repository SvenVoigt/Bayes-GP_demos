'''
By: Andrew Castillo
Date: Nov 9, 2017
Description: Class for simple 1-D linear regression
'''
import numpy as np
import basisfuncs
import matplotlib.pyplot as plt

class regress():
    xtrain=[]
    ytrain=[]
    feat=[]
    model=[]
    rsqrd=[]
    ftrue=[]

    def polybasis(self,d):
        self.feat=basisfuncs.polynomial(d)

    def rbfbasis(self,l,mu=None):
        if not mu:
            if not self.xtrain:
                mu=self.xtrain
            else:
                print('DEFINE TRAINING DATA or MU')
                return(None)
        self.feat=basisfuncs.rbf(l,mu)

    def traindata(self,xtrain,ytrain):
        self.xtrain=np.atleast_2d(xtrain).reshape((len(xtrain),1))
        self.ytrain=np.atleast_2d(ytrain).reshape((len(ytrain),1))

    def trainmodel(self):
        X=self.feat(self.xtrain)
        self.model=np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),self.ytrain)

    def eval(self,x):
        return(np.dot(self.feat(x),self.model))

    def plot(self):
        test=np.linspace(0,1,100)
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        ax1.plot(test,self.eval(test))
        ax1.scatter(self.xtrain,self.ytrain)
        #plt.show()
        return(fig)
