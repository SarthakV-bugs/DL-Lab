##Define different classes for say constant, variable, mathematical operation
##differentiate on each of these classes, constant -> 0, if variable same then 1 else , and math-op such as +,*,/
##Parse a string input equation and design the graph by creating nodes for each of the features inside it, while
## defining the subclass



class Node:
    """
    Base node
    """
    pass

class ConstantNode(Node):
    def __init__(self, value):
        self.value =  value #initializes the value


    def differentiate(self,variable):
        return ConstantNode(0) #returns a new derivative node with value 0



class VariableNode(Node):
    def __init__(self,name):
        self.name = name #takes the variable name

    def differentiate(self, variable):
        """

        :param variable: variable differentiating w.r.t
        :return: if variable name matches then return 1 else 0
        """
        if self.name == variable: #store node vs the variable being passed
            return ConstantNode(1)
        else:
            return ConstantNode(0)



class MathOp(Node):
    """Mathematical operation on child nodes, left and right"""
    def __init__(self, left, right):
        self.left = left
        self.right = right

class AddNode(MathOp):
    # def __repr__(self):
    #     return f"({self.left.__repr__()} + {self.right.__repr__()})" ##to check for the expression

    def differentiate(self, variable):
        """perform differentiation using the addition rule,
        differentiate on both the nodes recursively"""
        left_node_der = self.left.differentiate(variable)
        right_node_der = self.right.differentiate(variable)

        return AddNode(left_node_der,right_node_der)

class SubtractNode(MathOp):
    # def __repr__(self):
    #     return f"({self.left.__repr__()} + {self.right.__repr__()})" ##to check for the expression

    def differentiate(self, variable):
        """perform differentiation using the addition rule,
        differentiate on both the nodes recursively"""
        left_node_der = self.left.differentiate(variable)
        right_node_der = self.right.differentiate(variable)

        return SubtractNode(left_node_der,right_node_der)


class MultiplyNode(MathOp):

    def differentiate(self,variable):
        """perform differentiation on the basis of multiplication rule i.e. f' = y.f'(x) + x.f'(y) """

        # Incorrect, cannot add two objects using mathematical operations
        # left_node_der = self.left * self.right.differentiate(variable) + self.right * self.left.differentiate(variable)
        # right_node_der = self.right * self.left.differentiate(variable) + self.left * self.right.differentiate(variable)
        #
        # return MultiplyNode(left_node_der,right_node_der)


        #first take derivatives of each class, then use multiplication rule and multiple them crossly with their derivatives'

        left_der = self.left.differentiate(variable) #i.e f'(x) let's say
        right_der = self.right.differentiate(variable) # f'(y)

        # use the multiply node to multiply
        left_prod = MultiplyNode(left_der, self.right)
        right_prod = MultiplyNode(right_der, self.left)

        ##Add them using the Add node
        return AddNode(left_prod, right_prod)



#
# class PowerNode(MathOp):
#     """binary mathematical operation,
#      two main considerations when differentiating :
#
#      1) var is raised to the constant (x^2) -> 2x
#      2) const is raised to the var (2^x) -> 2^x*ln(2)
#
#      checks for the type of node to solve the above 2 conditions
#       """
#
#     def differentiate(self,variable):
#         if isinstance(self.right, ConstantNode): #to check if the exponent in the right child is constant
#
#             exp_val = self.right.value
#             new_exp_val = ConstantNode(exp_val-1) #n-1
#             new_power_node = PowerNode(self.left, new_exp_val) #x^n-1
#
#             prod = MultiplyNode(self.right, new_power_node) #n * x^n-1
#
#         elif isinstance(self.right, VariableNode): #to check if the exponent in the right child is variable
#
#              =






#
#
# class DivisionNode(MathOp):
#     """requires a node to represent power function in the denominator and a subtract node to show a subtract output """
#     def differentiate(self,variable):
#         left_derivative = self.left.differentiate(variable)
#         right_derivative = self.right.differentiate(variable)
#
#         left_term = DivisionNode(self.left,right_derivative)
#         right_term = DivisionNode(self.right,left_derivative)






##AddNode
# variable = VariableNode("x")
# number = ConstantNode(5)
# expression = AddNode(variable,number)
# # print(expression.__repr__())

##NumberNode
# number = ConstantNode(5)
# print(number.value)
# derivative = number.differentiate('x')
# print(derivative.value)

##VariableNode
# variable = VariableNode("x")
# print(variable.name)
# derivative = variable.differentiate("y")
# print(derivative.value)

