from recochef.models.thompson import ThompsonSimulation

# define the class object
thsim = ThompsonSimulation(n_bandits=5)

# define the arm reward probabilities
p_bandits = [0.05, 0.25, 0.10, 0.10, 0.50]

# pull arms N(=10) times
for i in range(10):
    x = thsim.step(p_bandits)
    print(p_bandits, x)

# change the rewards and continue pulling for N'(=10) more time
p_bandits = [0.05, 0.25, 0.60, 0.05, 0.05]
for i in range(10):
    x = thsim.step(p_bandits)
    print(p_bandits, x)