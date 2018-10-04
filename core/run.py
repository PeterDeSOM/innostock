import sys

from core.environments import *
from gym.wrappers import Monitor



def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def main():
	epoch = 10

	env_act = Activities(0, epoch)
	env_dnn = DeepNeuralNetwork(11, 41)
	env_ril = ReinforcementLearning()

	total_running = env_act.get_size_of_an_epoch()
	total_running *= epoch

	episode = 0

	_, i = env_act.reset()

	while i < epoch:
		run_times = env_act.run_times()
		printProgress(run_times,
		              total_running,
		              '',
		              ' (%s / %s) Epoch: %s,   Episod: %s (%s)' % (
			              '{0:9,d}'.format(run_times),
			              '{0:9,d}'.format(total_running),
			              '{0:2d}'.format(i),
			              '{0:9,d}'.format(episode),
			              '{0:2d}'.format(int(env_act.monitor.show('episod')))
		              ), 4, 20)
		done = False

		cache_states = []
		cache_policygradient_errs = []
		cache_rewards = []

		while not done:
			state = env_act.get_observation()

			if state is not None:
				action_probability = env_dnn.feed_forward(state)

				action = np.argmax(action_probability) if np.random.uniform() < np.max(action_probability) else int(round(np.random.uniform(0, 11)))
				# action = int(round(np.random.uniform(0, 11)))
				# action = np.argmax(action_probability)

				reward, done = env_act.step(np.argmax(action_probability), action)
				if done: episode += 1

				cache_states.append(state)
				cache_policygradient_errs.append(1. - action_probability)
				cache_rewards.append(reward)

			else:
				_, i = env_act.reset()
				if len(cache_states) == 0: break

				done = True

			if done:
				cache_states = np.vstack(cache_states)
				cache_policygradient_errs = np.vstack(cache_policygradient_errs)
				cache_rewards = np.vstack(cache_rewards)

				discounted_rewards = env_act.disount_rewards(cache_rewards)
				discounted_rewards -= np.mean(discounted_rewards)

				standard_deviation = np.std(discounted_rewards)
				if standard_deviation != 0:
					discounted_rewards /= standard_deviation
					cache_policygradient_errs *= discounted_rewards

				env_dnn.feed_backward(cache_states, cache_policygradient_errs, episode)
				env_dnn.deep_learning()


if __name__ == '__main__':
    main()

