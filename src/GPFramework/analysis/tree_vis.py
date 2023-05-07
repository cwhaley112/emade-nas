"""
- Converts list of individuals (in string format) and converts them to PNGs
- Requires you to download graphviz (free) and add it to your system's path: https://graphviz.org/download/
- Can alter shapes and colors with cmd line args
- Default PNG destination path: emade directory


Sample commands: 

- Run this file:
Put individual strings in list called "trees" at the end, then run
python src/GPFramework/analysis/tree_vis.py

- From another file:




Written by Cameron Whaley
"""

# NOTE: You will have to download graphviz and add it to your system's path
import pydot
import os

def findEnd(ind,i):
    # helper function to find the end (index) of a primitive's arguments
    paren = 1
    j=i
    while paren>0:
        j+=1 # since we're starting with paren=1, we don't want to count the one at the start
        char = ind[j]
        if char=='(':
            paren+=1
        elif char==')':
            paren-=1
    return j

def split(string, aList):
    # split primitive arguments separated by commas
    for aStr in string.split(','):
        aList.append(aStr)
    return aList

def process_string(ind,start=0,stop=None):
    # converts individual's string representation into nested lists
    # example input: "add(5,subtract(5,2))"
    # output: ["add", ["subtract", [5,2]]]
    # primitives without args are followed by an empty list
    
    result = []
    i=start
    if stop==None:
        stop=len(ind[start:])
    while i<stop:
        if ind[i] == '(':
            result = split(ind[start:i],result)
            end = findEnd(ind,i)
            result.append(process_string(ind,i+1,end))
            start=end+2 # move start to end of primitive. +2 to skip '),'
            i=end
        i+=1
    leftover = ind[start:stop] # catches arguments after final primitive
    if leftover != '':
        result = split(leftover,result)
    return result

def make_nodes(tree,namespace,graph,p_col,a_col,p_shp,a_shp):
    # recursively converts tree from process_string into nested lists of pydot nodes
    # adds nodes to graph
    # can change formatting in "else" code
    nodes = []
    for i in range(len(tree)):
        if type(tree[i])==list:
            subnodes, namespace = make_nodes(tree[i], namespace,graph,p_col,a_col,p_shp,a_shp)
            if subnodes != []:
                nodes.append(subnodes)
        else:
            #######################################################
            ############ RELEVANT CODE IF YOU WANT TO CHANGE VISUALIZATION STUFF ############
            #######################################################
            shape = a_shp
            color = a_col
            if i+1<len(tree) and type(tree[i+1])==list: # 1st condition should prevent index errors
                # is a primitive
                shape = p_shp 
                color = p_col
            label = tree[i] # the text shown on graph
            if label in namespace.keys():
                namespace[label]+=1 
            else:
                namespace[label]=1
            name = label + str(namespace[label]) # pydot nodes need unique names
            try:
                node = pydot.Node(name, label=label, shape=shape, color=color)
            except:
                raise ValueError(f"{shape} is not a valid pydot shape and/or {color} is not a valid pydot color")
            graph.add_node(node)
            nodes.append(node)
    return nodes, namespace

def make_graph(graph,nodes,parent):
    # connects nodes of graph
    for i in range(len(nodes)):
        if type(nodes[i])==list:
            make_graph(graph, nodes[i], nodes[i-1])
        else:
            graph.add_edge(pydot.Edge(parent, nodes[i]))

def write_png(name:str,individual:str, prim_col="black", arg_col="black", prim_shape="rect", arg_shape="oval"):
    # converts individual to dot tree and writes to name.png
    if ".png" not in name:
        name+=".png"
    individual = individual.strip().replace(' ','')
    dest_path = os.getcwd()

    graph = pydot.Dot("my_graph", graph_type='graph')
    tree = process_string(individual)
    nodes = make_nodes(tree,{}, graph,prim_col,arg_col,prim_shape,arg_shape)[0] # don't care about namespace returned here

    make_graph(graph,nodes[1],nodes[0]) # nodes[0] is global parent node. nodes[1] is a list of its leaves and subtrees
    graph.write_png(os.path.join(dest_path,name))

def get_img(individual:str, prim_col="black", arg_col="black", prim_shape="rect", arg_shape="oval"):
    # Display graph image in a notebook

    individual = individual.strip().replace(' ','')

    graph = pydot.Dot("my_graph", graph_type='graph')
    tree = process_string(individual)
    nodes = make_nodes(tree,{}, graph,prim_col,arg_col,prim_shape,arg_shape)[0] # don't care about namespace returned here

    make_graph(graph,nodes[1],nodes[0]) # nodes[0] is global parent node. nodes[1] is a list of its leaves and subtrees

    from IPython.display import Image, display
    plt = Image(graph.create_png())
    display(plt)

if __name__=="__main__":
    #############################################
    ####### PUT INDIVIDUALS IN THIS LIST: #######
    #############################################

    trees = ['NNLearner(ARG0, OutputLayer(Conv1DLayer(6, eluActivation, 3, 32, myNot(trueBool), 44, LSTMLayer(16, defaultActivation, 0, ifThenElseBool(falseBool, trueBool, trueBool), trueBool, EmbeddingLayer(97, ARG0, glorotNormalWeights, InputLayer())))), 150, RMSpropOptimizer)']

    #############################################
    #############################################
    #############################################

    # parsing args:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prim_col','-pc', type=str, default="black", help="color of primitive nodes (default: black)")
    parser.add_argument('--arg_col','-ac', type=str, default="black", help="color of argument or terminal nodes (default: black)")
    parser.add_argument('--prim_shape','-ps', type=str, default="rect", help="shape of primitive nodes (default: rect)")
    parser.add_argument('--arg_shape','-as',type=str,default="oval", help="shape of argument or terminal nodes (default: oval)")
    args = parser.parse_args()
    p_col = args.prim_col
    a_col = args.arg_col
    p_shp = args.prim_shape
    a_shp = args.arg_shape

    # cleaning individuals:
    trees = [t.strip().replace(' ','') for t in trees]

    names = [trees[i][0:trees[i].find('(')]+str(i) for i in range(len(trees))] # default name is parent node + number in queue

    # creating graphs and saving to PNGs:
    dest_path = os.getcwd()
    for i in range(len(trees)):
        graph = pydot.Dot("my_graph", graph_type='graph')
        tree = process_string(trees[i])
        nodes = make_nodes(tree,{}, graph,p_col,a_col,p_shp,a_shp)[0] # don't care about namespace returned here

        make_graph(graph,nodes[1],nodes[0]) # nodes[0] is global parent node. nodes[1] is a list of its leaves and subtrees
        graph.write_png(os.path.join(dest_path,names[i]+".png"))