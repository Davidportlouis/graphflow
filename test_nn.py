# testing functions
from miniflow.miniflow import Add,Mul,forward_pass,topological_sort,Input

def test_add():
    x,y,z = Input(),Input(),Input()
    add = Add(x,y,z)
    feed_dict = {x:4,y:5,z:20}
    sorted_nodes = topological_sort(feed_dict)
    output = forward_pass(add,sorted_nodes)
    assert(output == 29)
    
def test_mul():
    x,y = Input(),Input()
    mul = Mul(x,y)
    feed_dict = {x:50,y:10}
    sorted_nodes = topological_sort(feed_dict)
    output = forward_pass(mul,sorted_nodes)
    assert(output == 500)
    
    