
    
    def __init__(self, value, grad):
        self.value=value
        self.grad=grad

class MultiplyGate:
        """
        Attributes not in constructor:
        input1: first gate that inputs into the multiply gate
        input2: second gate that inputs into the multiply gate
        output: the output version of the gate that goes as input elsewhere
        """
        
        def forward(self, input1, input2): #input1,input2 are also Units
            # store pointers to input gates:
            self.input1=input1
            self.input2=input2
            self.output=Unit(input1.value*input2.value, 0.0)
            return(self.output)
            
        #take the gradient in output and chain it with local gradients    
        def backward(self):
            self.input1.grad += self.input2.value * self.output.grad 
            self.input2.grad += self.input1.value * self.output.grad 

class AddGate:
           
        def forward(self, input1, input2):
            self.input1=input1
            self.input2=input2
            self.output=Unit(input1.value+input2.value, 0.0)
            return(self.output)
            
        def backward(self):
            self.input1.grad += 1 * self.output.grad 
            self.input2.grad += 1 * self.output.grad 
            
class SigmoidGate:
           
        def forward(self, input1):
            self.input1=input1
            self.output=Unit(1/(1+exp(-input1.value)), 0.0)
            return(self.output)
            
        def backward(self):
            self.input1.grad += (self.output.value*(1-self.output.value)) * self.output.grad 

### Having defined the gates and units, let's run the forward pass to generate output values:

# Forward Pass

x=Unit(-2,0)
y=Unit(5,0)
z=Unit(-4,0)

a=AddGate()
q=a.forward(x,y)

m=MultiplyGate()
f=m.forward(q,z)

print(f.value) #should come out to (-2 + 5)*-4 , i.e., -12
-12
Now let us run the backward pass to decipher the gradient df/dx:

# Backward Pass

f.grad=1
m.backward()
a.backward()
a.input1.grad


class Unit:
    
    def __init__(self, value, grad):
        self.value=value
        self.grad=grad

class MultiplyGate:
        """
        Attributes not in constructor:
        input1: first gate that inputs into the multiply gate
        input2: second gate that inputs into the multiply gate
        output: the output version of the gate that goes as input elsewhere
        """
        
        def forward(self, input1, input2): #input1,input2 are also Units
            # store pointers to input gates:
            self.input1=input1
            self.input2=input2
            self.output=Unit(input1.value*input2.value, 0.0)
            return(self.output)
            
        #take the gradient in output and chain it with local gradients    
        def backward(self):
            self.input1.grad += self.input2.value * self.output.grad 
            self.input2.grad += self.input1.value * self.output.grad 

class AddGate:
           
        def forward(self, input1, input2):
            self.input1=input1
            self.input2=input2
            self.output=Unit(input1.value+input2.value, 0.0)
            return(self.output)
            
        def backward(self):
            self.input1.grad += 1 * self.output.grad 
            self.input2.grad += 1 * self.output.grad 
            
class SigmoidGate:
           
        def forward(self, input1):
            self.input1=input1
            self.output=Unit(1/(1+exp(-input1.value)), 0.0)
            return(self.output)
            
        def backward(self):
            self.input1.grad += (self.output.value*(1-self.output.value)) * self.output.grad 
            
            
 # custom exception if user enters a 'wrt' argument different from x or y
class InvalidWRTargError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

# write a function that numerically computes the partial derivative
# for a gate defined by function 'func', w.r.t a given input 'wrt'

# assume multiplicative gate if gate function not specified

def getNumericalForwardGradient(x, y, wrt, func=forwardMultiplyGate):
    initial=func(x,y)
    h=0.0001
    try:
        if wrt==x:
            final=func(x+h, y)
        elif wrt==y:
            final=func(x,y+h)
        else:
            raise InvalidWRTargError
    except InvalidWRTargError:
        return("third argument (wrt) should equal one of the inputs (first 2 arguments)")
    return((final-initial)/h)
