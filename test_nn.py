# testing functions
from graphflow.graphflow import *


def test_backprop():
    X,W,b = Input(),Input(),Input()
    y = Input()
    f = Linear(X,W,b)
    a = Sigmoid(f)
    loss = MSE(y,a)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2.], [3.]])
    b_ = np.array([-3.])
    y_ = np.array([1, 2])

    feed_dict = {
    X: X_,
    y: y_,
    W: W_,
    b: b_,
        }
    graph = topological_sort(feed_dict)
    forward_and_backward(graph)
    gradients = [t.gradients[t] for t in [X, y, W, b]]
    assert(gradients == [np.array([[-0.00067025, -0.00100538],[-0.00134073, -0.00201109]]),
                         np.array([[0.99966465],[1.99966465]]),
                         np.array([[0.00100549],[0.00201098]]),
                         np.array([[-0.00100549]])])

test_backprop()

# def test_add():
#     x,y,z = Input(),Input(),Input()
#     add = Add(x,y,z)
#     feed_dict = {x:4,y:5,z:20}
#     sorted_nodes = topological_sort(feed_dict)
#     output = forward_pass(add,sorted_nodes)
#     assert(output == 29)
    
# def test_mul():
#     x,y = Input(),Input()
#     mul = Mul(x,y)
#     feed_dict = {x:50,y:10}
#     sorted_nodes = topological_sort(feed_dict)
#     output = forward_pass(mul,sorted_nodes)
#     assert(output == 500)

# def test_linear():
#     inputs,weights,bias = Input(),Input(),Input()
#     linear = Linear(inputs,weights,bias)
#     feed_dict = {
#         inputs: np.array([[2,4,6],[1,3,5]]),
#         weights: np.array([[1],[2,],[3]]),
#         bias: np.array([1])
#     }
#     sorted_nodes = topological_sort(feed_dict)
#     output = forward_pass(linear,sorted_nodes)
#     assert((output == np.array([[29],[23]])).all())

# def test_sigmoid():
#     z = Input()
#     sig = Sigmoid(z)
#     feed_dict = {z:0}
#     graph = topological_sort(feed_dict)
#     output = forward_pass(sig,graph)
#     assert(output == 0.5)
    
# def test_MSE():
#     y,a = Input(),Input()
#     cost = MSE(y,a)
#     feed_dict={
#         y:np.array([[1,2,3]]),
#         a:np.array([[4.5,5,10]])
#     }
#     graph = topological_sort(feed_dict)
#     output = forward_pass(cost,graph)
#     assert(round(output,2) == 23.42)
