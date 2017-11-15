'''
By: Andrew Castillo
Date: Nov 9, 2017
Description: Class for simple 1-D bayesian regression
'''
import numpy as np
import basisfuncs
import matplotlib.pyplot as plt

class bayesregress():
    xtrain=[]
    ytrain=[]
    feat=[]
    model=[]
    rsqrd=[]
    alpha=[]
    beta=[]
    PHImat=[]
    m0=[]
    XtXeigs=[]
    Sn=[]
    ftrue=[]

    def polybasis(self,d):
        self.feat=basisfuncs.polynomial(d)
        self.model=np.zeros((d+1,1)) # initialize model
        self.m0=self.model

    def rbfbasis(self,l,mu=None):
        if not mu:
            if not self.xtrain:
                mu=self.xtrain
            else:
                print('DEFINE TRAINING DATA or MU')
                return(None)
        self.feat=basisfuncs.rbf(l,mu)
        self.model=np.zeros((len(mu),1)) # initialize model
        self.m0=self.model

    def traindata(self,xtrain,ytrain):
        self.xtrain=np.atleast_2d(xtrain).reshape((len(xtrain),1))
        self.ytrain=np.atleast_2d(ytrain).reshape((len(ytrain),1))

    def sethyper(self,alpha,beta):
        self.alpha=alpha
        self.beta=beta

    def trainmodel(self):
        if len(self.PHImat)==0:
            self.PHImat=self.feat(self.xtrain)

        X=self.PHImat
        XtX=np.dot(X.transpose(),X)
        S0=np.eye(X.shape[1])*1/self.alpha#this assumes all points have same precision
        Sninv=np.linalg.inv(S0)+self.beta*XtX#
        Sn=np.linalg.inv(Sninv)
        self.Sn=Sn
        mn=np.dot(Sn,(np.dot(np.linalg.inv(S0),self.m0)+self.beta*np.dot(X.transpose(),self.ytrain)))#;%this does not assume a mean gaussian at zero
        self.model=mn


    def optimize(self):
        """
        Maximizing parameters based on training data only
        """
        if len(self.PHImat)==0:
            self.PHImat=self.feat(self.xtrain)


        if len(self.XtXeigs)==0:
            #X=self.beta*np.dot(self.PHImat.transpose(),self.PHImat)
            #w,v=np.linalg.eig(X)
            XtX=np.dot(self.PHImat.transpose(),self.PHImat)
            w,v=np.linalg.eig(self.beta*XtX)
            self.XtXeigs=w #eigenvalues


        mn=self.model
        #lam=np.diag(w)
        lam=self.XtXeigs
        gamma=np.sum(lam/(self.alpha+lam))
        alpha=gamma/(np.dot(mn.transpose(),mn))
        self.alpha=alpha

        N=float(self.PHImat.shape[0])
        M=float(self.PHImat.shape[1])
        tmp=1/(N-gamma)*np.sum(np.square(self.ytrain-np.dot(self.PHImat,mn)))
        beta=1/tmp
        self.beta=beta
        self.trainmodel()

    def eval(self,x):
        x=np.atleast_2d(x).reshape((len(x),1))
        fhat=np.dot(self.feat(x),self.model)
        var=[]
        for i in x:
            var.append(1/self.beta+np.dot(self.feat(i),np.dot(self.Sn,self.feat(i).transpose())))
        var=np.array(var).reshape((len(x),1))
        return(fhat,var)

    def plot(self):
        #scale around confidence interval
        xtest=np.linspace(0,1,100).reshape((100,1))
        fstar,var=self.eval(xtest)
        se=2*np.sqrt(var.reshape((np.max(var.shape),1)))
        yvect=np.vstack((self.ytrain,fstar+se,fstar-se))
        y_range=np.vstack((yvect.max(),yvect.min()))
        xvect=np.vstack((self.xtrain,xtest))
        x_range=np.vstack((xvect.max(),xvect.min()))

        fig1=plt.figure()
        ax=fig1.add_subplot(111)
        ax.scatter(self.xtrain,self.ytrain,marker="+",color="b")
        ax.plot(xtest,fstar+se,color="r")
        ax.plot(xtest,fstar-se,color="r")
        ax.plot(xtest,fstar,color="b")
        top=fstar+se
        bot=fstar-se
        plt.fill_between(xtest[:,0],top[:,0],bot[:,0],color='blue',alpha=0.3)
        #fig1.show()
        return (fig1)
