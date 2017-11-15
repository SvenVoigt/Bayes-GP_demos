import numpy as np

def linearmodel():
    w0=0.4
    w1=1
    def f(X):
        out=[]
        X=list(X)
        for x in X:
            out.append(w0+w1*x)
        out=np.array(out).reshape((len(out),1))
        return(out)
    return(f)

def noisylinearmodel(alpha=1):
    mu=0
    w0=0.4
    w1=1
    def f(X):
        out=[]
        X=list(X)
        for x in X:
            out.append(w0+w1*x+np.random.randn()*np.sqrt(alpha))
        out=np.array(out).reshape((len(out),1))
        return(out)
    return(f)

def testfuncinterp():
    def f(X):
        out=[]
        X=list(X)
        for x in X:
            out.append(1/(1+16*x**2))
        out=np.array(out).reshape((len(out),1))
        return(out)
    return(f)

def testfuncgp():
    sin=np.sin
    cos=np.cos

    def f(X):
        out=[]
        X=list(X)
        for x in X:
            out.append((sin(30*(x-0.9)**4))*cos(2*(x-0.9))+(x-0.9)/2)
        out=np.array(out).reshape((len(out),1))
        return(out)
    return(f)
