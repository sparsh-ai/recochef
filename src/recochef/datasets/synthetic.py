from data_cache import pandas_cache
import pandas as pd
import numpy as np


class Synthetic():
  def __init__(self, version='v1'):
    self.version = version

  @pandas_cache
  def explicit(self):
    """
    In the "explicit feedback" scenario, interactions between users and items
    are numerical / ordinal ratings or binary preferences such as like or
    dislike. These types of interactions are termed as explicit feedback.
    """
    if self.version=='v1':
      """
      The following shows a dummy data for the explicit rating type of feedback. In the data,
        - There are 3 users whose IDs are 1, 2, 3.
        - There are 3 items whose IDs are 1, 2, 3.
        - Items are rated by users only once. So even when users interact with items at different timestamps, the ratings are kept the same. This is seen in some use cases such as movie recommendations, where users' ratings do not change dramatically over a short period of time.
        - Timestamps of when the ratings are given are also recorded.
      """
      data = pd.DataFrame({
          "USERID": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
          "ITEMID": [1, 1, 2, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 1],
          "RATING": [4, 4, 3, 3, 3, 4, 5, 4, 5, 5, 5, 5, 5, 5, 4],
          "TIMESTAMP": [
              '2000-01-01', '2000-01-01', '2000-01-02', '2000-01-02', '2000-01-02',
              '2000-01-01', '2000-01-01', '2000-01-03', '2000-01-03', '2000-01-03',
              '2000-01-01', '2000-01-03', '2000-01-03', '2000-01-03', '2000-01-04'
          ]
      })

      return data

  @pandas_cache
  def implicit(self):
    """
    Many times there are no explicit ratings or preferences given by users,
    that is, the interactions are usually implicit. For example, a user may
    puchase something on a website, click an item on a mobile app, or order
    food from a restaurant. This information may reflect users' preference
    towards the items in an implicit manner.
    """
    if self.version=='v1':
      """
      The following shows a dummy data for the implicit rating type of feedback. In the data,
        - There are 3 users whose IDs are 1, 2, 3.
        - There are 3 items whose IDs are 1, 2, 3.
        - There are no ratings or explicit feedback given by the users. Sometimes there are the types of events. In this dummy dataset, for illustration purposes, there are three types for the interactions between users and items, that is, click, add and purchase, meaning "click on the item", "add the item into cart" and "purchase the item", respectively.
        - Sometimes there is other contextual or associative information available for the types of interactions. E.g., "time-spent on visiting a site before clicking" etc. For simplicity, only the type of interactions is considered in this notebook.
        - The timestamp of each interaction is also given.
      """
      data = pd.DataFrame({
          "USERID": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
          "ITEMID": [1, 1, 2, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 1],
          "EVENT": [
              'click', 'click', 'click', 'click', 'purchase',
              'click', 'purchase', 'add', 'purchase', 'purchase',
              'click', 'click', 'add', 'purchase', 'click'
          ],
          "TIMESTAMP": [
              '2000-01-01', '2000-01-01', '2000-01-02', '2000-01-02', '2000-01-02',
              '2000-01-01', '2000-01-01', '2000-01-03', '2000-01-03', '2000-01-03',
              '2000-01-01', '2000-01-03', '2000-01-03', '2000-01-03', '2000-01-04'
          ]
      })

      return data


class Session():
  def __init__(self, version='trivago'):
    self.version = version

  @pandas_cache
  def train(self):

    if self.version=='trivago':

      d_train = {
          "user_id": [
              "64BL89", "64BL89", "64BL89", "64BL89",
              "64BLF", "64BLF",
              "64BL89", "64BL89", "64BL89", "64BL89"
          ],
          "session_id": [
              "3579f89", "3579f89", "3579f89", "3579f89",
              "4504h9", "4504h9",
              "5504hFL", "5504hFL", "5504hFL", "5504hFL"
          ],
          "timestamp": [
              1, 2, 3, 4,
              2, 4,
              7, 8, 9, 10
          ],
          "step": [
              1, 2, 3, 4,
              1, 2,
              1, 2, 3, 4
          ],
          "action_type": [
              "interaction item image", "clickout item", 
                  "interaction item info", "filter selection",
              "interaction item image", "clickout item",
                  "filter selection", "clickout item", 
              "interaction item image", "clickout item"
          ],
          "reference": [
              "5001", "5002", "5003", "unknown",
              "5010", "5001",
              "unknown", "5004", "5001", "5001"
          ],
          "impressions": [
              np.NaN, "5014|5002|5010", np.NaN, np.NaN,
              np.NaN, "5001|5023|5040|5005",
              np.NaN, "5010|5001|5023|5004|5002|5008", 
                  np.NaN, "5010|5001|5023|5004|5002|5008"
          ],
          "prices": [
              np.NaN, "100|125|120", np.NaN, np.NaN,
              np.NaN, "75|110|65|210",
              np.NaN, "120|89|140|126|86|110", np.NaN, "120|89|140|126|86|110"
          ]
      }

      df_train = pd.DataFrame(d_train)

      return df_train


  def test(self):

    if self.version=='trivago':

      d_test = {
          "user_id": [
              "64BL89", "64BL89",
              "64BL91F2", "64BL91F2", "64BL91F2"
          ],
          "session_id": [
              "3579f90", "3579f90",
              "3779f92", "3779f92", "3779f92"
          ],
          "timestamp": [
              5, 6,
              9, 10, 11
          ],
          "step": [
              1, 2,
              1, 2, 3
          ],
          "action_type": [
              "interaction item image", "clickout item",
              "interaction item info", "clickout item", "filter selection"
          ],
          "reference": [
              "5023", np.NaN,
              "5010", np.NaN, "unknown"
          ],
          "impressions": [
              np.NaN, "5002|5003|5010|5004|5001|5023",
              np.NaN, "5001|5004|5010|5014", np.NaN
          ],
          "prices": [
              np.NaN, "120|75|110|105|89|99",
              np.NaN, "76|102|115|124", np.NaN
          ]
      }

      df_test = pd.DataFrame(d_test)

      return df_test


  def items(self):

    if self.version=='trivago':

      d_item_meta = {
          "item_id": ["5001", "5002", "5003", "5004"],
          "properties": [
              "Wifi|Croissant|TV",
              "Wifi|TV",
              "Croissant",
              "Shoe dryer"
          ]
      }

      df_item_meta =  pd.DataFrame(d_item_meta)

      return df_item_meta


class SequentialMarkov():

  """Originally published here: https://github.com/maciejkula/spotlight/blob/master/spotlight/datasets/synthetic.py"""

  @staticmethod
  def _build_transition_matrix(num_items,
                              concentration_parameter,
                              random_state,
                              atol=0.001):

      def _is_doubly_stochastic(matrix, atol):

          return (np.all(np.abs(1.0 - matrix.sum(axis=0)) < atol) and
                  np.all(np.abs(1.0 - matrix.sum(axis=1)) < atol))

      transition_matrix = random_state.dirichlet(
          np.repeat(concentration_parameter, num_items),
          num_items)

      for _ in range(100):

          if _is_doubly_stochastic(transition_matrix, atol):
              break

          transition_matrix /= transition_matrix.sum(axis=0)
          transition_matrix /= transition_matrix.sum(1)[:, np.newaxis]

      return transition_matrix

  @staticmethod
  def _generate_sequences(num_steps,
                          transition_matrix,
                          order,
                          random_state):

      elements = []

      num_states = transition_matrix.shape[0]

      transition_matrix = np.cumsum(transition_matrix,
                                    axis=1)

      rvs = random_state.rand(num_steps)
      state = random_state.randint(transition_matrix.shape[0], size=order,
                                  dtype=np.int64)

      for rv in rvs:

          row = transition_matrix[state].mean(axis=0)
          new_state = min(num_states - 1,
                          np.searchsorted(row, rv))

          state[:-1] = state[1:]
          state[-1] = new_state

          elements.append(new_state)

      return np.array(elements, dtype=np.int32)


  def generate_sequential(num_users=100,
                          num_items=1000,
                          num_interactions=10000,
                          concentration_parameter=0.1,
                          order=3,
                          random_state=None):
      """
      Generate a dataset of user-item interactions where sequential
      information matters.
      The interactions are generated by a n-th order Markov chain with
      a uniform stationary distribution, where transition probabilities
      are given by doubly-stochastic transition matrix. For n-th order chains,
      transition probabilities are a convex combination of the transition
      probabilities of the last n states in the chain.
      The transition matrix is sampled from a Dirichlet distribution described
      by a constant concentration parameter. Concentration parameters closer
      to zero generate more predictable sequences.
      Parameters
      ----------
      num_users: int, optional
          number of users in the dataset
      num_items: int, optional
          number of items (Markov states) in the dataset
      num_interactions: int, optional
          number of interactions to generate
      concentration_parameter: float, optional
          Controls how predictable the sequence is. Values
          closer to zero give more predictable sequences.
      order: int, optional
          order of the Markov chain
      random_state: numpy.random.RandomState, optional
          random state used to generate the data
      """

      if random_state is None:
          random_state = np.random.RandomState()

      transition_matrix = _build_transition_matrix(
          num_items - 1,
          concentration_parameter,
          random_state)

      user_ids = np.sort(random_state.randint(0,
                                              num_users,
                                              num_interactions,
                                              dtype=np.int32))
      item_ids = _generate_sequences(num_interactions,
                                    transition_matrix,
                                    order,
                                    random_state) + 1
      timestamps = np.arange(len(user_ids), dtype=np.int32)
      ratings = np.ones(len(user_ids), dtype=np.float32)

      _df = pd.DataFrame(list(zip(user_ids, item_ids, ratings, timestamps)),
                        columns=['USERIDS','ITEMIDS','RATINGS','TIMESTAMP'])
      _df = _df.drop_duplicates()

      return _df