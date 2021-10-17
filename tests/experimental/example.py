import numpy as np
import ray

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.application_manager import instance
from nums.experimental.optimizer.grapharray import GraphArray
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.tree_search import RandomTS

app: ArrayApplication = instance()

cluster_state = ClusterState(app.cm.system.devices())
A_matrix = app.array([[1, 2, 3], 
                      [3, 4, 5],
                      [6, 7, 8]], block_shape=(3, 1))
B_matrix = app.array([[5, 6, 7], 
                      [7, 8, 9], 
                      [10, 11, 12]], block_shape=(3, 1))
C = GraphArray.op_fusion(A_matrix, B_matrix)
print(C.get())
print("Done")