import math
import random

        ##engine##
class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.00
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value({self.data})"

    def __add__(self, other):
        if isinstance(other, Value):
            pass
        else:
            other = Value(other)
        obj = self.data + other.data
        
        out = Value(obj, _children=(self, other), _op="+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward

        return out
             
    def __mul__(self, other):
        if isinstance(other, Value):
            pass
        else:
            other = Value(other)
        obj = self.data * other.data
        out = Value(obj, _children=(self, other), _op="*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def tanh(self):
        t = (math.exp(2*self.data) -1) / (math.exp(2*self.data) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - out.data**2)
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        obj = self.data ** other
        out = Value(obj, (self,), f"**{other}")

        #Power is always constant like int, float. Also Andrej Karpahty did like this.
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def backward(self):
        topo = []         
        visited = set()  
    
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child) 
                topo.append(v)        
        
        build_topo(self) 
        
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

        ##neural networks##
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for i in range(nin)]

        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        y = sum((a * b for a, b in zip(x, self.w)), self.b)
 
        return y.tanh()


class Layer:
  
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin) for i in range(nout)]
        
    def __call__(self, x):
        return [i(x) for i in self.neurons]


class MLP:

    def __init__(self, nin, nouts):
        
        self.nin = nin
        self.nouts = nouts
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz-1))]

    def __call__(self, x):
        
        for layer in self.layers:
            x = layer(x)

        return x[0] if len(x) == 1 else x
            


        
