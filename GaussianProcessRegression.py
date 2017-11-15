'''
By: Andrew Castillo
Date: Nov 9, 2017
Description: Class for simple 1-D gaussian process regression
'''
import numpy as np
import basisfuncs
from scipy import optimize
import matplotlib.pyplot as plt
#%matplotlib notebook

class gpprocess():
    xtrain=[]
    ytrain=[]
    feat=[]
    model=[]
    rsqrd=[]
    mu=[]
    sigma2=[]
    l=[]
    ftrue=[]
    Rinv=[]

    def rbfbasis(self,l):
        self.l=l
        mu=self.xtrain
        self.feat=basisfuncs.rbf(l,mu)
        self.model=(np.zeros((len(mu),1))) # initialize model

    def traindata(self,xtrain,ytrain):
        self.xtrain=np.atleast_2d(xtrain).reshape((len(xtrain),1))
        self.ytrain=np.atleast_2d(ytrain).reshape((len(ytrain),1))

    def trainGP(self):
        #if len(self.Rinv)==0:
        self.PHImat=self.feat(self.xtrain)
        R=self.PHImat #this is thought of as the correlation matrix as well..
        Rinv=np.linalg.inv(R+10e-10*np.eye(R.shape[0])) #this is to help stability... common practice
        self.Rinv=Rinv
        if self.mu!=0:
            self.mu=np.dot(np.dot(np.ones((1,R.shape[0])),Rinv),self.ytrain)/np.dot(np.dot(np.ones((1,R.shape[0])),Rinv),np.ones((R.shape[0],1)))
        self.sigma2=1/float(R.shape[0])*np.dot(np.dot((self.ytrain-self.mu).transpose(),Rinv),(self.ytrain-self.mu))
        self.model=np.linalg.solve(R,(self.ytrain-self.mu)) #just like the interpolation scheme

    def optimize(self):
        """
        Maximizing parameters based on training data only
        """
        if len(self.PHImat)==0:
            self.PHImat=self.feat(self.xtrain)

        func=self.logliklihood #this calculates the loglikelihood
        out=optimize.fmin(func,self.l) #different optimization procedures available
        self.l=float(out[0])
        print('Updating Hyperparameters')
        self.rbfbasis(self.l)
        print('Training GP')
        self.trainGP()
        lval=[]
        ltest=np.linspace(0.001,0.05,100)
        for i in ltest:
            lval.append(-self.logliklihood(i)) #we minimize the negative log-liklihood using built-in
        plt.close('all')
        fig6=plt.figure()
        ax6=fig6.add_subplot(111)
        ax6.set_xlabel('Hyperparameter value')
        ax6.set_ylabel('log-liklihood')
        ax6.plot(ltest,lval)
        plt.show(fig6)
        return(fig6)

    def logliklihood(self,l):
        liklifeat=basisfuncs.rbf(l,self.xtrain)
        R=liklifeat(self.xtrain)
        Rinv=np.linalg.inv(R+10e-10*np.eye(R.shape[0]))
        if self.mu!=0:
            self.mu=np.dot(np.dot(np.ones((1,R.shape[0])),Rinv),self.ytrain)/np.dot(np.dot(np.ones((1,R.shape[0])),Rinv),np.ones((R.shape[0],1)))
        self.sigma2=1/float(R.shape[0])*np.dot(np.dot((self.ytrain-self.mu).transpose(),Rinv),(self.ytrain-self.mu))
        lnlikli=float(R.shape[0]*np.log(self.sigma2)+np.log(np.linalg.det(R)))
        return(lnlikli)

    def eval(self,x):
        fhat=np.dot(self.feat(x),self.model)+self.mu
        var=self.sigma2*np.diag((np.ones((len(x),1))-np.dot(np.dot(self.feat(x),self.Rinv),self.feat(x).transpose()))).reshape((len(fhat),1))
        return(fhat,var)

    def plot(self):
        xtest=np.linspace(0,1,100).reshape((100,1))
        fstar,var=self.eval(xtest)
        se=2*np.sqrt(var.reshape((np.max(var.shape),1)))
        yvect=np.vstack((self.ytrain,fstar+se,fstar-se))
        y_range=np.vstack((yvect.max(),yvect.min()))
        xvect=np.vstack((self.xtrain,xtest))
        x_range=np.vstack((xvect.max(),xvect.min()))

        fig4=plt.figure()
        ax4=fig4.add_subplot(111)
        ax4.scatter(self.xtrain,self.ytrain,marker="+",color="b")
        ax4.plot(xtest,fstar+se,color="r")
        ax4.plot(xtest,fstar-se,color="r")
        ax4.plot(xtest,fstar,color="b")
        top=fstar+se
        bot=fstar-se
        plt.fill_between(xtest[:,0],top[:,0],bot[:,0],color='blue',alpha=0.3)
        plt.legend()
        #plt.show(fig4)
        return(fig4)

    def plottrue(self):
        if not self.ftrue:
            return('Please Provide True Function, self.ftrue=f(x)')
        xtest=np.linspace(0,1,100).reshape((100,1))
        fstar,var=self.eval(xtest)
        se=2*np.sqrt(var.reshape((np.max(var.shape),1)))
        yvect=np.vstack((self.ytrain,fstar+se,fstar-se))
        y_range=np.vstack((yvect.max(),yvect.min()))
        xvect=np.vstack((self.xtrain,xtest))
        x_range=np.vstack((xvect.max(),xvect.min()))

        fig5=plt.figure()
        ax5=fig5.add_subplot(111)
        ax5.scatter(self.xtrain,self.ytrain,marker="+",color="b")
        ax5.plot(xtest,fstar+se,color="r")
        ax5.plot(xtest,fstar-se,color="r", label='0.95 conf')
        ax5.plot(xtest,fstar,color="b",label='GP mean')
        top=fstar+se
        bot=fstar-se
        plt.fill_between(xtest[:,0],top[:,0],bot[:,0],color='blue',alpha=0.3)
        ax5.plot(xtest,self.ftrue(xtest),color="orange",label='True')
        plt.legend()
        #plt.show(fig5)
        return(fig5)
