import numpy as np

def activation_functions(x: float, activation: str) -> list:
    x=float(x)
    if activation=="relu": output=max(0.0,x); derivative=1.0 if x>0.0 else 0.0
    elif activation=="leaky_relu": output=x if x>0.0 else 0.01*x; derivative=1.0 if x>0.0 else 0.01
    elif activation=="sigmoid": output=1.0/(1.0+np.exp(-x)); derivative=output*(1.0-output)
    elif activation=="tanh": output=float(np.tanh(x)); derivative=1.0-output**2
    elif activation=="swish": sigmoid=1.0/(1.0+np.exp(-x)); output=x*sigmoid; derivative=sigmoid+x*sigmoid*(1.0-sigmoid)
    else:
        c=np.sqrt(2.0/np.pi); inner=c*(x+0.044715*x**3); tanh_inner=np.tanh(inner); output=0.5*x*(1.0+tanh_inner); derivative=0.5*(1.0+tanh_inner)+0.5*x*(1.0-tanh_inner**2)*c*(1.0+3.0*0.044715*x**2)
    return [round(float(output),4),round(float(derivative),4)]
