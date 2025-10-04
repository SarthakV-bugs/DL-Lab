class Node:
    """
    Base node
    """
    pass

class ConstantNode(Node):
    def __init__(self, value):
        self.value =  value #initializes the value
        self.grad = 0 #stores the gradient,initialized to zero

    def differentiate(self,variable):
        return ConstantNode(0) #returns a new derivative node with value 0

    def forward_pass(self):
        return self.value

    def backward_pass(self):
        return 0


class VariableNode(Node):
    def __init__(self,name, value):
        self.name = name #create the var name
        self.value = value #store the val in this var
        self.grad = 0.0

    def differentiate(self, variable):
        """

        :param variable: variable differentiating w.r.t
        :return: if variable name matches then return 1 else 0
        """
        if self.name == variable: #store node vs the variable being passed
            return ConstantNode(1)
        else:
            return ConstantNode(0)

    def forward_pass(self):
        return self.value



class MathOp(Node):
    """Mathematical operation on child nodes, left and right"""
    def __init__(self, left, right):
        self.left = left
        self.right = right

class AddNode(MathOp):
    # def __repr__(self):
    #     return f"({self.left.__repr__()} + {self.right.__repr__()})" ##to check for the expression

    def __init__(self, left, right):
        super().__init__(left, right)
        self.value = None
        self.grad = 0

    def differentiate(self, variable):
        """perform differentiation using the addition rule,
        differentiate on both the nodes recursively"""
        left_node_der = self.left.differentiate(variable)
        right_node_der = self.right.differentiate(variable)

        return AddNode(left_node_der,right_node_der)

    def forward_pass(self):
        self.value =  self.left.forward_pass() + self.right.forward_pass()
        return self.value


class SubtractNode(MathOp):
    # def __repr__(self):
    #     return f"({self.left.__repr__()} + {self.right.__repr__()})" ##to check for the expression

    def __init__(self, left, right):
        super().__init__(left, right)
        self.value = None
        self.grad = 0

    def differentiate(self, variable):
        """perform differentiation using the addition rule,
        differentiate on both the nodes recursively"""
        left_node_der = self.left.differentiate(variable)
        right_node_der = self.right.differentiate(variable)

        return SubtractNode(left_node_der,right_node_der)

    def forward_pass(self):
        self.value =  self.left.forward_pass() + self.right.forward_pass()
        return self.value


class MultiplyNode(MathOp):

    def __init__(self, left, right):
        super().__init__(left, right)
        self.value = None
        self.grad = 0

    def differentiate(self,variable):
        """perform differentiation on the basis of multiplication rule i.e. f' = y.f'(x) + x.f'(y) """

        # Incorrect, cannot add two objects using mathematical operations
        # left_node_der = self.left * self.right.differentiate(variable) + self.right * self.left.differentiate(variable)
        # right_node_der = self.right * self.left.differentiate(variable) + self.left * self.right.differentiate(variable)
        # return MultiplyNode(left_node_der,right_node_der)


        #first take derivatives of each class, then use multiplication rule and multiple them crossly with their derivatives'
        left_der = self.left.differentiate(variable) #i.e f'(x) let's say
        right_der = self.right.differentiate(variable) # f'(y)

        # use the multiply node to multiply
        left_prod = MultiplyNode(left_der, self.right)
        right_prod = MultiplyNode(right_der, self.left)

        ##Add them using the Add node
        return AddNode(left_prod, right_prod)

    def forward_pass(self):
        self.value =  self.left.forward_pass() * self.right.forward_pass()
        return self.value


 # f(x, y) = (x + y) * x

x = VariableNode("x",5)
y = VariableNode("y", 10)

add = AddNode(x,y)
add.forward_pass()
print(add.value)