import math
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to inspect the Block structure

def inspect_blocks(b1, b2, RAW, dict_EC, blockD, G_con, filter_out, Node_Labels):
    print(RAW[b1,b2])

    '''
    --------------------
    
    A function that takes in two block labels from our stochastic block model 
    and produces a printed list of the blocks and draws a subgraph for visual
    analysis of the blocks. We also return the edge count between the blocks, 
    and the normalised edge count.
    
    The filter_out variable can be changed to either 'disease' or 'gene' and
    it will filter out the node type given to it
    --------------------
    '''
    
    ### Edge Counts
    
    if (b1, b2) in RAW:
        print('The number of edges between these blocks is: ',RAW[b1,b2])
    elif (b2,b1) in RAW:
        print('The number of edges between these blocks is: ',RAW[b2,b1])
        
    if (b1, b2) in dict_EC:
        print('The normalised edge counts between these blocks is: ',dict_EC[b1,b2])
    elif (b2,b1) in dict_EC:
        print('The normalised edge counts between these blocks is: ',dict_EC[b2,b1])

    #### Node Lists 
    source_block_sought = b1

    List_A = []
    for i in blockD[source_block_sought]:
        #k = g_connected.vp["node_name"][blockDictionary[source_block_sought][i]]
        List_A.append(i)
    
    if filter_out != None:
        if filter_out == 'disease':
            List_A_revised = []
            for i in List_A:
                if i[:4] != 'OMIM':
                    List_A_revised.append(i)
            List_A = List_A_revised
        elif filter_out == 'gene':
            List_A_revised = []
            for i in List_A:
                if i[:4] == 'OMIM':
                    List_A_revised.append(i)
            List_A = List_A_revised
        
    target_block_sought = b2

    List_B = []
    for i in blockD[target_block_sought]:
        #k = g_connected.vp["node_name"][blockDictionary[target_block_sought][i]]
        List_B.append(i)
    
    ### Creating a subgraph ############################
    
    node_list = List_A + List_B
    ### Node Count
    print('The number of nodes in the subgraph is:', len(node_list))
    subgraph = G_con.subgraph(node_list)
    
    ### Size of the largest component in the subgraph ## 
    
    largest_cc=max(nx.connected_components(subgraph),key=len)
    subgraph=subgraph.subgraph(largest_cc)
    size_of_the_largest_component = nx.number_of_nodes(subgraph)
    print('The size of the largest component in the subgraph of these two blocks is:',size_of_the_largest_component)
    
    ### print blocks ##################################
    
    print('')
    print(List_A)
    print('')
    print(List_B)
    
    ###### Drawing a Subgraph (The largest connected component of the subgraph between blocks)
    plt.figure(figsize=(15,15))
    
    ### Color Map
    ColorMap = []
    for node in subgraph:
        if node in List_A:
            ColorMap.append('r')
        else:
            ColorMap.append('b')
            
    nx.draw_kamada_kawai(subgraph, node_color=ColorMap, node_size = 60, edge_color='silver', with_labels=Node_Labels)
    plt.title("Subgraph of 2 connected blocks", fontsize = 14)
    
    EDGES = subgraph.edges()
    #print(len(EDGES))
    Unique_EDGES = np.unique(EDGES)
    #print(len(Unique_EDGES))



def COMPUTE_ENTROPY(contingency_matrix, C_or_R ='Rows'):
    
    rows = len(contingency_matrix.axes[0])
    cols = len(contingency_matrix.axes[1])
    
    if C_or_R == 'Rows':
        sum_list = list(contingency_matrix.sum(axis=1))  ### Sum across each row
    elif C_or_R == 'Cols':
        sum_list = list(contingency_matrix.sum(axis=0))  ### Sum down each column

    N = contingency_matrix.sum().sum()
    
    Average_Entropy = 0
    Pointwise_Entropy_List = []
    for i in range(len(sum_list)):
        x = sum_list[i]
        x_prob = x/N
        x_log = -(math.log(x_prob))
        X = x_prob * x_log
        
        Average_Entropy += X
        Pointwise_Entropy_List.append(x_log)
    
    return Average_Entropy, Pointwise_Entropy_List


def COMPUTE_JOINT_ENTROPY(contingency_matrix):
    
    rows = len(contingency_matrix.axes[0])
    cols = len(contingency_matrix.axes[1])

    #rows_sum_list = list(contingency_matrix.sum(axis=1))  ### Sum across each row
    #cols_sum_list = list(contingency_matrix.sum(axis=0))  ### Sum down each column

    N = contingency_matrix.sum().sum()
    
    Pointwise_Joint_Entropy_Matrix = np.zeros((rows,cols))
    
    Average_Joint_Entropy = 0
    Pointwise_Joint_Entropy_List = []
    for i in range(rows):
        for j in range(cols):
            x = contingency_matrix.iloc[i,j]
            x_prob = x/N
            x_prob += 0.000000001
            x_log = -(math.log(float(x_prob)))
            X = x_prob * x_log

            Average_Joint_Entropy += X
            Pointwise_Joint_Entropy_Matrix[i,j] = x_log
    
    return Average_Joint_Entropy, Pointwise_Joint_Entropy_Matrix


def COMPUTE_MUTUAL_INFORMATION(contingency_matrix):
    
    rows = len(contingency_matrix.axes[0])
    cols = len(contingency_matrix.axes[1])

    rows_sum_list = list(contingency_matrix.sum(axis=1))  ### Sum across each row
    cols_sum_list = list(contingency_matrix.sum(axis=0))  ### Sum down each column

    N = contingency_matrix.sum().sum()
    
    Pointwise_MI_Matrix = np.zeros((rows,cols))
    
    Average_MI = 0
    
    for i in range(rows):
        for j in range(cols):
            
            n_ij = contingency_matrix.iloc[i,j]
            
            if n_ij == 0:
                Average_MI += 0
                Pointwise_MI_Matrix[i,j] = 0   
            else: 
                
                x_prob = n_ij/N
                #x_prob += 0.0000000000000000001

                a_i = rows_sum_list[i] #+0.0000000000000000001
                b_j = cols_sum_list[j] #+0.0000000000000000001

                if b_j == 0:
                    denominator = a_i/(N**2)
                else:
                    denominator = (a_i*b_j)/(N**2)

                x_log = (math.log(x_prob/denominator))
                X = x_prob * x_log

                Average_MI += X
                Pointwise_MI_Matrix[i,j] = x_log
    
    return Average_MI, Pointwise_MI_Matrix



def contingency_matrix(dictA,dictB):

    contingency_matrix = np.zeros((len(dictA),len(dictB)))

    for key in dictA.keys():     
        for i in range(len(dictA[key])):     
            for new_key in (dictB.keys()):  
                for element in (dictB[new_key]):
                    if  dictA[key][i] == element:
                        contingency_matrix[int(key)-1,new_key] += 1
    
    return pd.DataFrame(contingency_matrix)




