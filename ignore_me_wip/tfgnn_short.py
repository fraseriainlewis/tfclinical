"""
library(reticulate)
use_virtualenv("/Users/work/gnn", required = TRUE)
#use_python("/Users/work/gnn/bin/python", required = TRUE)




"""


import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For TF2.16+.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

print(f'Running TF-GNN {tfgnn.__version__} with TensorFlow {tf.__version__}.')




train_path = os.path.join(os.getcwd(), 'mutag', 'train.tfrecords')
val_path = os.path.join(os.getcwd(), 'mutag', 'val.tfrecords')
#get_ipython().system('ls -l {train_path} {val_path}')


graph_tensor_spec = tfgnn.GraphTensorSpec.from_piece_specs(
    context_spec=tfgnn.ContextSpec.from_field_specs(features_spec={
                  'label': tf.TensorSpec(shape=(1,), dtype=tf.int32)
    }),
    node_sets_spec={
        'atoms':
            tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        tf.TensorSpec((None, 7), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32))
    },
    edge_sets_spec={
        'bonds':
            tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        tf.TensorSpec((None, 4), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'atoms', 'atoms'))
    })


def decode_fn(record_bytes):
  graph = tfgnn.parse_single_example(
      graph_tensor_spec, record_bytes, validate=True)

  # extract label from context and remove from input graph
  context_features = graph.context.get_features_dict()
  label = context_features.pop('label')
  new_graph = graph.replace_features(context=context_features)

  return new_graph, label


# In[7]:


train_ds = tf.data.TFRecordDataset([train_path]).map(decode_fn)
val_ds = tf.data.TFRecordDataset([val_path]).map(decode_fn)


# ### Look at one example from the dataset

# In[8]:


g, y = train_ds.take(1).get_single_element()


# #### Node features
# 
# Node features represent the 1-hot encoding of the atom type (0=C, 1=N, 2=O, 3=F,
# 4=I, 5=Cl, 6=Br).

# In[9]:


print(g.node_sets['atoms'].features[tfgnn.HIDDEN_STATE])


# #### Bond Edges
# 
# In this example, we consider the bonds between atoms undirected edges. To encode
# them in the GraphsTuple, we store the undirected edges as pairs of directed
# edges in both directions.
# 
# `adjacency.source` contains the source node indices, and `adjacency.target` contains the corresponding target node indices.

# In[10]:


g.edge_sets['bonds'].adjacency.source


# In[11]:


g.edge_sets['bonds'].adjacency.target


# #### Edge features
# 
# Edge features represent the bond type as one-hot encoding.

# In[12]:


g.edge_sets['bonds'].features[tfgnn.HIDDEN_STATE]


# ### Label
# The label is binary, indicating the mutagenicity of the molecule. It's either 0 or 1.

# In[13]:


y


