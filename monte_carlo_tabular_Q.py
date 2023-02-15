
import numpy as np
import gym
import matplotlib.pyplot as plt

###############
# User Inputs #
###############

render = False  # Render the environment
nBins = 8
gamma = 0.99 # Future reward decay
eps = 0.2 # Random policy threshold
epsDecay = 0.9999 # Decay rate applied to epsilon each episode
nEpisodes = 10000
epMaxIter = 1000
firstVisit = True # Only update Q for the first visit to the state per episode

#######################################
# Create Environment Get Measurements #
#######################################

env = gym.make('CartPole-v1')
if render:
    env = gym.make('CartPole-v1', render_mode='human')
else:
    env = gym.make('CartPole-v1')

observBoundHigh = env.observation_space.high
observBoundLow = env.observation_space.low
nObserv = np.size(observBoundHigh)
nStates = nBins ** nObserv
nActions = env.action_space.n

################
# RL Functions #
################

# Make bins for each observation dimension
# Bins are given by a list of endpoints for each bin
def make_observ_bins(min, max, nBins):
    # Clip the min and max values to something sensible
    min = np.clip(min, -5, None)
    max = np.clip(max, None, 5)

    # Calculate the step and generate a range of values using the min/max/step
    step = (float(max) - float(min)) / nBins
    bins = list(np.arange(min, max + step, step))
    return bins

# From a given observation, generate the state index (max = nStates) associated with the observation
def observ_to_state_idx(observ, observDim, observBins, nBins):

    # Generate array coordinates (list of bin indices for each observation dimension)
    coord = []
    for i in range(observDim):
        coord.append(np.digitize(observ[i], observBins[i]) - 1)

    # Convert the array coordinate to a single index
    # e.g., coordinate [0, 1, 2, 3] for 5 bins in each of 4 dims gives 0*5^3 + 1*5^2 + 2*5^1 + 3*5^0 = 38
    idx = np.ravel_multi_index(coord, [nBins] * observDim)
    return idx

# Given the values for each action associated with a Q[state], determine optimal action (small chance of random action)
def tabular_epsilon_greedy_policy(QState, eps):

    # Choose action associated with maximum value given state
    if np.random.random_sample() > eps:
        # Get all maximal indices. If multiple max indices, randomly select one
        maxIndices = np.where(QState == np.amax(QState))[0].tolist()
        if len(maxIndices) > 1:
            actionIdx = np.random.choice(maxIndices)
        else:
            actionIdx = maxIndices[0]
    # Choose random action given state
    else:
        actionIdx = np.random.randint(len(QState))
    return actionIdx

# Calculates the discounted rewards associated with each step of an episode. Future rewards are discounted by gamma
def get_discounted_rewards(rewardsList, gamma):

    # Initialize counter and cumulative rewards list
    R = 0
    cumulRewards = []

    # Calculate discounted future rewards and accumulate
    # If rewardsList is [1,1,1,3], then cumulRewards is [1+gam(1+gam(1+gam*3)), 1+gam(1+gam*3), 1+gam*3, 3]
    for reward in rewardsList[::-1]:
        R = reward + gamma * R
        cumulRewards.insert(0, R)
    return cumulRewards

##################################
# Execute Reinforcement Learning #
##################################

# Cycle through each observation dimension and create bins based on their min/max values
observBins = []
for i in range(nObserv):
    observBins.append(make_observ_bins(observBoundLow[i], observBoundHigh[i], nBins))

# Initialize Q(s,a) and nVisits(s,a) arrays
Q = np.zeros((nStates, nActions))
nVisits = np.zeros((nStates, nActions))

# Plotting lists
plotIters = []
plotRewards = []

# Cycle through episodes
for i in range(nEpisodes + 1):

    # Initialize states/actions/rewards
    statesList = []
    actionsList = []
    rewardsList = []

    # Policy evaluation (run an episode)
    observ = env.reset()[0].tolist()
    for _ in range(epMaxIter):

        # From observation, get state index and choose action
        stateIdx = observ_to_state_idx(observ, nObserv, observBins, nBins)
        actionIdx = tabular_epsilon_greedy_policy(Q[stateIdx, :], eps)

        # Step environment
        observ, reward, done, info, _ = env.step(actionIdx)

        # Update state/actions/rewards lists and end episode if done
        statesList.append(stateIdx)
        actionsList.append(actionIdx)
        rewardsList.append(reward)
        if done:
            break

    # Calculate rewards
    cumulRewardsList = get_discounted_rewards(rewardsList, gamma)
    totalRewards = sum(rewardsList)

    # Policy improvement
    updatedStates = set()
    for stateIdx, actionIdx, cumulReward in zip(statesList, actionsList, cumulRewardsList):

        # Only update Q if this is the first time visiting the given state this episode
        if firstVisit:
            if stateIdx in updatedStates:
                continue
            updatedStates.add(stateIdx)

        # Increase the visits counter for the specific state/action and then update the Q(s,a) value
        # Note that Q(s,a) can go up or down depending on cumulReward relative to the current value
        nVisits[stateIdx, actionIdx] += 1
        Q[stateIdx, actionIdx] += (cumulReward - Q[stateIdx, actionIdx]) / nVisits[stateIdx, actionIdx]

    # Update epsilon
    eps = eps * epsDecay

    # Print state and update plotters
    if i % 100 == 0:
        print('Episode ' + str(i) + ' Total Reward: ' + str(totalRewards))
    plotIters.append(i)
    plotRewards.append(totalRewards)

# Plot progress
plt.plot(plotIters, plotRewards)
plt.show()