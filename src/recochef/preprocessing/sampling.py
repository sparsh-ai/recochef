import numpy as np
import pandas as pd


def simple_negative_sampling(data,
                             num_negatives=4,
                             binarization=False,
                             feedback_column='RATING'):
  
  # Get a list of all Item IDs
  all_itemsIds = data['ITEMID'].unique()

  # Placeholders that will hold the data
  users, items, labels = [], [], []

  if binarization:
    data.loc[:,feedback_column] = 1

  user_item_set = set(zip(data['USERID'], data['ITEMID'], data[feedback_column]))

  for (u, i, r) in user_item_set:
    users.append(u)
    items.append(i)
    labels.append(r)
    for _ in range(num_negatives):
      # randomly select an item
      negative_item = np.random.choice(all_itemsIds) 
      # check that the user has not interacted with this item
      while (u, negative_item) in user_item_set:
          negative_item = np.random.choice(all_itemsIds)
      users.append(u)
      items.append(negative_item)
      labels.append(0) # items not interacted with are negative
  ns_data = pd.DataFrame(list(zip(users, items, labels)),
                         columns=['USERID','ITEMID',feedback_column])
  return ns_data