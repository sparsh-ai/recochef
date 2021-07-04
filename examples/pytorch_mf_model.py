import torch
import torch.nn.functional as F

from recochef.datasets.synthetic import Synthetic
from recochef.preprocessing.split import chrono_split
from recochef.preprocessing.encode import label_encode as le
from recochef.models.factorization import MF, MF_bias
from recochef.models.dnn import CollabFNet

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

# event implicit to rating conversion
event_weights = {'click':1, 'add':2, 'purchase':4}
event_maps = dict({'EVENT_TO_IDX':event_weights})
df_train = le(df_train, col='EVENT', maps=event_maps)
df_valid = le(df_valid, col='EVENT', maps=event_maps)
print(f"Processed Train set:\n\n{df_train}\n{'='*100}\n")
print(f"Processed Validation set:\n\n{df_valid}\n{'='*100}\n")

# get number of unique users and items
num_users = len(df_train.USERID.unique())
num_items = len(df_train.ITEMID.unique())
print(f"There are {num_users} users and {num_items} items.\n{'='*100}\n")

# training and testing related helper functions
def train_epocs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(df_train.USERID.values) # .cuda()
        items = torch.LongTensor(df_train.ITEMID.values) #.cuda()
        ratings = torch.FloatTensor(df_train.EVENT.values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item()) 
    test_loss(model, unsqueeze)

def test_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_valid.USERID.values) #.cuda()
    items = torch.LongTensor(df_valid.ITEMID.values) #.cuda()
    ratings = torch.FloatTensor(df_valid.EVENT.values) #.cuda()
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())

# training MF model
model = MF(num_users, num_items, emb_size=100) # .cuda() if you have a GPU
print(f"Training MF model:\n")
train_epocs(model, epochs=10, lr=0.1)
print(f"\n{'='*100}\n")

# training MF with bias model
model = MF_bias(num_users, num_items, emb_size=100) #.cuda()
print(f"Training MF+bias model:\n")
train_epocs(model, epochs=10, lr=0.05, wd=1e-5)
print(f"\n{'='*100}\n")

# training MLP model
model = CollabFNet(num_users, num_items, emb_size=100) #.cuda()
print(f"Training MLP model:\n")
train_epocs(model, epochs=15, lr=0.05, wd=1e-6, unsqueeze=True)
print(f"\n{'='*100}\n")


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

Processed Train set:

    USERID  ITEMID  EVENT   TIMESTAMP
0        0       0      1  2000-01-01
2        0       1      1  2000-01-02
5        1       0      1  2000-01-01
6        1       1      4  2000-01-01
7        1       0      2  2000-01-03
8        1       1      4  2000-01-03
10       2       2      1  2000-01-01
11       2       2      1  2000-01-03
12       2       2      2  2000-01-03
13       2       2      4  2000-01-03
====================================================================================================

Processed Validation set:

    USERID  ITEMID  EVENT   TIMESTAMP
4        0       1      4  2000-01-02
9        1       2      4  2000-01-03
14       2       0      1  2000-01-04
====================================================================================================

There are 3 users and 3 items.
====================================================================================================

Training MF model:

5.836816787719727
1.993103265762329
4.549840450286865
1.5779536962509155
1.285771131515503
1.926152229309082
2.242276191711426
2.270019054412842
2.3635096549987793
2.272618055343628
test loss 9.208 

====================================================================================================

Training MF+bias model:

5.8399200439453125
3.7661311626434326
1.8716331720352173
1.6015545129776
1.5306222438812256
1.2995147705078125
1.1046849489212036
1.1331274509429932
1.2109991312026978
1.2451963424682617
test loss 5.625 

====================================================================================================

Training MLP model:

4.953568458557129
9.649873733520508
1.1805670261383057
2.54287052154541
2.6113314628601074
2.1839144229888916
1.4144573211669922
0.8893814086914062
0.7603365182876587
1.240354061126709
1.1316341161727905
0.8014519810676575
0.7997692823410034
0.8474739789962769
0.9691768884658813
test loss 5.207 

====================================================================================================
"""