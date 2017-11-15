import numpy as np

# This is a collection of the basis functions we will use
def polynomial(n):
    deg=range(n+1)
    def poly(X):
        dmat=[]
        #dmat=np.zeros((x.shape[0],len(deg)))
        for x in X:
            dmat.append(np.power(x,deg))
        return(np.array(dmat))
    return(poly)

def rbf(l,mu):
    #mu is the locations of the basis functions, 1 d basis function
    mu=np.atleast_2d(mu)
    def RBF_(x):
        #'''dmat=np.zeros((x.shape[0],np.max(xtrain.shape[0])))'''
        x=np.atleast_2d(x)
        x=x.reshape((np.max(x.shape),1))
        dmat=np.zeros((x.shape[0],mu.shape[0]))
        for i in range(len(x)):
            for j in range(len(mu)):
                #dmat[i,j]=np.exp(-np.square(np.linalg.norm(x[i]-mu[j]))/(2*np.square(l))) ##for multidimensional
                dmat[i,j]=np.exp(-np.square(x[i,0]-mu[j,0])/(l))
        return (dmat)
    return(RBF_)
