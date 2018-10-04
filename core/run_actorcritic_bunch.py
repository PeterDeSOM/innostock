from core.env_actorcritic_bunch import *


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
	sess = tf.Session()
	kb.set_session(sess)

	epoch = 5
	training_size = 117
	episode = 0.
	done = False
	in_dim = 35
	out_dim = 9

	env_act = Activities(0, epoch, out_dim, in_dim)
	env_kac = Keractoritic(sess, out_dim, in_dim)

	total_running = env_act.get_size_of_an_epoch()
	# total_running = 40394827
	total_running *= epoch

	_, epoch_current = env_act.reset()
	state_current = env_act.get_observation()
	# Reshape (axis=0:35, axis=1:?) to (axis=0:1, axis=1:35)
	state_current = state_current.reshape((1, env_kac.dim_input()))
	sending_massage = ''

	while epoch_current < epoch:
		if done:
			# ACTOR&CRITIC Models will be trained when the episod is done, not by every step
			env_kac.train(training_size)

			episode += 1.

			run_times = env_act.run_times()
			env_kac.set_running_times(run_times)

			if run_times % 10 == 0:
				cost1, accuracy, cost2 = env_kac.learning_info()
				printProgress(run_times,
				              total_running,
				              '',
				              ' (%s / %s) EPOCH: %s, EPISOD: %s # COST: %s / %s, ACCURACY: %s / %s %s' % (
					              '{0:9,.0f}'.format(run_times),
					              '{0:9,.0f}'.format(total_running),
					              '{0:2d}'.format(epoch_current),
					              '{0:9,.0f}'.format(episode),
					              '{0:.7f}'.format(cost1),
					              '{0:.7f}'.format(cost2[0] if type(cost2) is list else 0.),
					              '{0:.2f}'.format(accuracy),
					              '{0:.2f}'.format(cost2[1] if type(cost2) is list else 0.),
					              sending_massage
				              ), 4, 10
				)
				print('')

		# [I]. 'action_probability' as a prediction from the ACTOR MODEL is the kind of new Policy
		action_probability, action_type = env_kac.get_action(state_current)
		# Reshape (axis=0:9, axis=1:?) to (axis=0:1, axis=1:9)
		# The output shape from the ACTOR model is already (1, 9), but the outcome from randomly pulled is not...
		action_probability = action_probability.reshape((1, out_dim))

		# Action = action_probability
		state_new, reward, done, real_action_position, sending_massage = env_act.step(action_probability, action_type)
		if state_new is None:
			_, epoch_current = env_act.reset()
			if epoch_current == epoch:
				break

			_ = env_act.get_observation()
			state_new, reward, done, real_action_position, sending_massage = env_act.step(action_probability, action_type)
		state_new = state_new.reshape((1, in_dim))

		# [II]. 'action_error' is right the difference between real action and action_probability from the ACTOR MODEL
		# That means this difference is the Gradient of this STATE(state_current)'s Policy
		# In this case, real action could be 1. in probability
		action_error = .9995
		if not done:
			action_error = 1 - action_probability[0, real_action_position]

		env_kac.keep_history(state_current, action_probability, reward, done, state_new, action_error)

		state_current = state_new



if __name__ == '__main__':
	main()

