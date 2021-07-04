import torch
import torch.nn as nn
import torch.nn.functional as F

from recochef.datasets.synthetic import Synthetic
from recochef.preprocessing.split import chrono_split
from recochef.preprocessing.encode import label_encode as le

# generate synthetic implicit data
synt = Synthetic()
df = synt.implicit()

# drop duplicates
df = df.drop_duplicates()

# chronological split
df_train, df_valid = chrono_split(df)
print(f"Train set:\n\n{df_train}\n{'='*100}\n")
print(f"Validation set:\n\n{df_valid}\n{'='*100}\n")

# label encoding
df_train, uid_maps = le(df_train, col='USERID')
df_train, iid_maps = le(df_train, col='ITEMID')
df_valid = le(df_valid, col='USERID', maps=uid_maps)
df_valid = le(df_valid, col='ITEMID', maps=iid_maps)

# an Embedding module containing 10 user or item embedding size 3
# embedding will be initialized at random
embed = nn.Embedding(10, 2)

# given a list of ids we can "look up" the embedding corresponing to each id
ids = [1,2,0,4,5,1]
a = torch.LongTensor([ids])
print(f"Randomly initialized Embeddings of a list of ids {ids}:\n\n{embed(a)}\n{'='*100}\n")

# initializing and multiplying users, items embeddings for the sample dataset
emb_size = 2
user_emb = nn.Embedding(df_train.USERID.nunique(), emb_size)
item_emb = nn.Embedding(df_train.ITEMID.nunique(), emb_size)
users = torch.LongTensor(df_train.USERID.values)
items = torch.LongTensor(df_train.ITEMID.values)
U = user_emb(users)
V = item_emb(items)
print(f"User embeddings of length {emb_size}:\n\n{U}\n{'='*100}\n")
print(f"Item embeddings of length {emb_size}:\n\n{V}\n{'='*100}\n")
print(f"Element-wise multiplication of user and item embeddings:\n\n{U*V}\n{'='*100}\n")
print(f"Dot product per row:\n\n{(U*V).sum(1)}\n{'='*100}\n")


"""
Train set:

    USERID  ITEMID     EVENT   TIMESTAMP
0        1       1     click  2000-01-01
2        1       2     click  2000-01-02
5        2       1     click  2000-01-01
6        2       2  purchase  2000-01-01
7        2       1       add  2000-01-03
8        2       2  purchase  2000-01-03
10       3       3     click  2000-01-01
11       3       3     click  2000-01-03
12       3       3       add  2000-01-03
13       3       3  purchase  2000-01-03
====================================================================================================

Validation set:

    USERID  ITEMID     EVENT   TIMESTAMP
4        1       2  purchase  2000-01-02
9        2       3  purchase  2000-01-03
14       3       1     click  2000-01-04
====================================================================================================

Randomly initialized Embeddings of a list of ids [1, 2, 0, 4, 5, 1]:

tensor([[[-0.4989, -0.0017],
         [ 0.2724,  0.1308],
         [-0.3845,  1.0548],
         [ 0.0951, -0.7816],
         [-1.2381,  0.4325],
         [-0.4989, -0.0017]]], grad_fn=<EmbeddingBackward>)
====================================================================================================

User embeddings of length 2:

tensor([[-0.7574, -1.1494],
        [-0.7574, -1.1494],
        [ 1.3911,  1.0157],
        [ 1.3911,  1.0157],
        [ 1.3911,  1.0157],
        [ 1.3911,  1.0157],
        [ 0.0271, -1.2206],
        [ 0.0271, -1.2206],
        [ 0.0271, -1.2206],
        [ 0.0271, -1.2206]], grad_fn=<EmbeddingBackward>)
====================================================================================================

Item embeddings of length 2:

tensor([[ 0.0406,  0.4805],
        [-0.7570, -1.6676],
        [ 0.0406,  0.4805],
        [-0.7570, -1.6676],
        [ 0.0406,  0.4805],
        [-0.7570, -1.6676],
        [-0.9237,  1.2666],
        [-0.9237,  1.2666],
        [-0.9237,  1.2666],
        [-0.9237,  1.2666]], grad_fn=<EmbeddingBackward>)
====================================================================================================

Element-wise multiplication of user and item embeddings:

tensor([[-0.0308, -0.5522],
        [ 0.5733,  1.9167],
        [ 0.0565,  0.4880],
        [-1.0530, -1.6937],
        [ 0.0565,  0.4880],
        [-1.0530, -1.6937],
        [-0.0251, -1.5460],
        [-0.0251, -1.5460],
        [-0.0251, -1.5460],
        [-0.0251, -1.5460]], grad_fn=<MulBackward0>)
====================================================================================================

Dot product per row:

tensor([-0.5830,  2.4900,  0.5445, -2.7467,  0.5445, -2.7467, -1.5711, -1.5711,
        -1.5711, -1.5711], grad_fn=<SumBackward1>)
====================================================================================================
"""