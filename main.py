import gym
from models import get_model
import numpy as np
from funcs import decay_and_normalize
import random


# create gym
env = gym.make("CartPole-v0")
env.reset()

# assign model to variable
model = get_model(3e-4)

# totaal

for _ in range(1000):
    all_obs = []
    all_rewards = []
    all_done = []
    all_actions = []
    for i in range(3):
        # game niveau
        observations = []
        rewards = []
        dones = []
        done = False
        actions = []
        obs = env.reset()
        while not done:
            # turn niveau, we hebben info niet nodig
            p = model(obs.reshape(1, -1))[0]

            if random.random() < p:
                action = 1
            else: action = 0

            observations.append(obs)
            dones.append(done)
            obs, reward, done, _ = env.step(action)

            rewards.append(reward)
            actions.append(action)
            # env.render()

        # laatste turn voor game over
        all_done.append(dones)
        all_rewards.append(rewards)
        all_obs.append(observations)
        all_actions.append(actions)

    print(all_done)
    import time
    time.sleep(1000)

    all_obs = np.concatenate([np.concatenate([i]) for i in all_obs])
    input = np.array(all_obs).reshape((-1, 4))

    all_actions = np.concatenate([np.concatenate([i]) for i in all_actions])
    output = np.array(all_actions).reshape((-1, 1))
    print(output.shape)


    all_rewards = decay_and_normalize(all_rewards, 0.97)
    semi = model(input)
    model.fit(input, all_actions)

print(all_rewards)
