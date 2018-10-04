import numpy as np
import pickle
import gym

from gym.wrappers import Monitor

render = False
hidden = 10
in_size = 4
out_size = 1
batch_size = 10
resume = False

class RL(object):
    def __init__(self, in_size, out_size):
        np.random.seed(1)
        self.gamma = 0.995
        self.decay_rate = 0.99
        self.learning_rate = 0.002

        if resume:
            self.model = pickle.load(open('Cart.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(hidden, in_size) / np.sqrt(in_size)
            self.model['W2'] = np.random.randn(hidden, hidden) / np.sqrt(hidden)
            self.model['W3'] = np.random.randn(hidden) / np.sqrt(hidden)

        self.grad_buffer = []
        for i in range(batch_size):
            self.grad_buffer.append({k: np.zeros_like(v) for k, v in self.model.items()})
        self.rmsprop_chche = {k: np.zeros_like(v) for k, v in self.model.items()}

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def disount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add

        return discounted_r

    def policy_forward(self, x):
        h1 = np.dot(self.model['W1'], x)
        h1 = self.sigmoid(h1)
        h2 = np.dot(self.model['W2'], h1)
        h2 = self.sigmoid(h2)
        logp = np.dot(self.model['W3'], h2)
        p = self.sigmoid(logp)

        return p, h1, h2

    def policy_backward(self, stack_layer1s, stack_layer2s, stack_policygradient_errors, stack_states, ep_num):
        dW3 = np.dot(stack_layer2s.T, stack_policygradient_errors).ravel()

        dh2 = np.outer(stack_policygradient_errors, self.model['W3'])

        print(np.size(dh2, 0))
        print(np.size(dh2, 1))
        exit()

        stack_layer2s_dot = stack_layer2s * (1 - stack_layer2s)
        dW2 = dh2 * stack_layer2s_dot

        dh1 = np.dot(dW2, self.model['W2'])
        stack_layer1s_dot = stack_layer1s * (1 - stack_layer1s)
        dW1 = dh1 * stack_layer1s_dot

        dW2 = np.dot(dW2.T, stack_layer1s)
        dW1 = np.dot(dW1.T, stack_states)

        self.grad_buffer[ep_num % batch_size] = {'W1': dW1, 'W2': dW2, 'W3': dW3}

    def learning(self):
        tmp = self.grad_buffer[0]
        for i in range(1, batch_size):
            for k, v in self.model.items():
                tmp[k] += self.grad_buffer[i][k]

        for k, v in self.model.items():
            g = tmp[k]
            self.rmsprop_chche[k] = self.decay_rate * self.rmsprop_chche[k] + (1 - self.decay_rate) * g**2
            self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_chche[k]) + 1e-5)

## End Class RL

def train(env):
    rl = RL(in_size, out_size)
    running_reward = None
    reward_sum, episode_num = 0, 0
    
    # State, Hidden1, Hidden2, Policy Gradient, Reward
    states, layer1s, layer2s, posicygradient_errors, rewards = [], [], [], [], []
    
    for i_episode in range(2000):
        done = False
        observation = env.reset()
        
        while not done:
            x = observation
    
            if render:
                env.render()
    
            action_probability, h1, h2 = rl.policy_forward(x)

            action = 1 if np.random.uniform() < action_probability else 0
            y = action

            states.append(x)
            layer1s.append(h1)
            layer2s.append(h2)
            posicygradient_errors.append(y - action_probability)

            observation, reward, done, info = env.step(action)
            reward_sum += reward
            rewards.append(reward)

            if done:
                episode_num += 1

                print('Episode: %d, Reward: %s' % (episode_num, '{:,}'.format(reward_sum)))

                stack_states = np.vstack(states)
                stack_layer1s = np.vstack(layer1s)
                stack_layer2s = np.vstack(layer2s)
                stack_policygradient_errors = np.vstack(posicygradient_errors)
                stack_rewards = np.vstack(rewards)
                states, layer1s, layer2s, posicygradient_errors, rewards = [], [], [], [], []

                discounted_stack_rewards = rl.disount_rewards(stack_rewards)
                discounted_stack_rewards -= np.mean(discounted_stack_rewards)
                discounted_stack_rewards /= np.std(discounted_stack_rewards)
                stack_policygradient_errors *= discounted_stack_rewards

                rl.policy_backward(stack_layer1s, stack_layer2s, stack_policygradient_errors, stack_states, episode_num)
                rl.learning()

                reward_sum = 0

if __name__ == '__main__':
    env = gym.make('InnostockLearning-v0')

    monitor = Monitor(env, 'CartPole/', force=True)
    #env.monitor.start('CartPole/', force=True)
    train(env)
    monitor.close()


