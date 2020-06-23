# testing functions
from miniflow.miniflow import Input,Add,topological_sort,forward_pass

def test_add():
    x,y = Input(),Input()
    add = Add(x,y)
    feed_dict = {x:20,y:5}
    sorted_nodes = topological_sort(feed_dict)
    output = forward_pass(add,sorted_nodes)
    assert(output == 25)

    
    