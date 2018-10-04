import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import tensorflow as tf
from isin_episod.dataset import Source, Controller
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import RMSprop

_time_sleep_for_work_ = .00001

def seperator(id=None):
    print('%s%s' % ('' if id is None else id, '-' * 100))

class A3CAgent:
    def __init__(self):
        self.MONITOR = Monitor(self)
        self.MODELSAVER = ModelSaver(self)
        self.AGENTS = []
        self.DATA_PREGRESS = []

        self.hold_process = True

        self.current_datasource_name = ''
        self.threads = 7

    def train(self):
        datasource = Source()
        source_size = datasource.source_size()

        self.AGENTS = [Agent(self) for _ in range(self.threads)]

        seperator()
        self.MONITOR.start()
        self.MODELSAVER.start()

        for i in range(source_size):
            if len(datasource.next()) == 0: break

            seperator()
            self.current_datasource_name, dataset, input_dim, output_dim = datasource.get_dateset()
            dataset_info = datasource.dataset_info()
            print('##### [ISIN: %s, SYMBOL NAME: %s, SCALE: %s] started. (%s of %s)' % (dataset_info[1], dataset_info[2], dataset_info[3], i + 1, source_size))

            self.set_training_environment(input_dim, output_dim)

            if i == 0:
                seperator()

            for agent in self.AGENTS:
                agent.set_training_environment(
                    self.current_datasource_name,
                    self.state_size,
                    self.action_size,
                    [self.actor, self.critic],
                    self.sess,
                    self.optimizer,
                    self.discount_factor
                )
                if not agent.ready_to_work:
                    agent.start()

            self.distribute([source_size, i + 1], dataset)
            if i + 1 < source_size:
                self.sess_close()

            seperator()
            print('##### [ISIN: %s, SYMBOL NAME: %s, SCALE: %s] finished. (%s of %s)' % (dataset_info[1], dataset_info[2], dataset_info[3], i + 1, source_size))

        self.stop_working()

    def set_training_environment(self, input_dim, output_dim):
        self.discount_factor = 0.99
        self.actor_lr = .0001
        self.critic_lr = .0001

        self.state_size = input_dim
        self.action_size = output_dim

        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        input_state = Input(shape=(self.state_size,))
        layer1 = Dense(self.state_size, activation='relu')(input_state)
        layer2 = Dense(self.state_size, activation='relu')(layer1)
        layer3 = Dense(self.state_size, activation='relu')(layer2)
        layer4 = Dense(self.state_size, activation='relu')(layer3)
        layer5 = Dense(self.state_size, activation='relu')(layer4)
        layer6 = Dense(self.state_size, activation='relu')(layer5)
        layer7 = Dense(self.state_size, activation='relu')(layer6)

        policy = Dense(self.action_size, activation='softmax')(layer7)
        value = Dense(1, activation='linear')(layer7)

        actor = Model(inputs=input_state, outputs=policy)
        critic = Model(inputs=input_state, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, 1])

        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

        return train

    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None, 1))

        value = self.critic.output
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_prediction], [loss], updates=updates)

        return train

    def distribute(self, source_size, dataset):
        source_controller = Controller(dataset)

        epoch = 3000
        epoch_current = 1

        while epoch_current <= epoch:
            state, y_value, idx = np.array([]), -1, -1
            obervation, rate_applied = source_controller.next()

            if type(obervation) is int:
                epoch_current += 1
            else:
                state = obervation[0:-2].astype('float').reshape((1, self.state_size))
                y_value = obervation[-1]
                idx = source_controller.id()

            for agent in self.AGENTS:
                agent.set_observation(state, y_value, idx)

            self.DATA_PREGRESS = sum([source_size, [epoch, epoch_current, '{0:.4f}'.format(rate_applied)]], [])

            if self.hold_process:
                self.hold_process = False

        self.hold_process = True


    def sess_close(self):
        for agent in self.AGENTS:
            agent.initializing = True

        time.sleep(_time_sleep_for_work_)

        while True:
            accuracys = []

            for agent in self.AGENTS:
                accuracys.append(int(agent.initializing))

            if np.sum(accuracys) == self.threads:
                self.sess.close()
                break

            time.sleep(_time_sleep_for_work_)

    def get_accuracy(self):
        accuracys = []
        for agent in self.AGENTS:
            accuracys.append(agent.get_accuracy())
        return np.mean(accuracys)

    def get_progressing(self):
        return self.DATA_PREGRESS

    def stop_working(self):
        for agent in self.AGENTS: agent.stopped = True
        self.MONITOR.stopped = True
        self.MODELSAVER.stopped = True

        time.sleep(1)

        seperator()
        for agent in self.AGENTS: agent.stop()
        self.MONITOR.stop()
        self.MODELSAVER.stop()


class Agent(threading.Thread):
    def __init__(self, creator):
        threading.Thread.__init__(self)

        self.ready_to_work = False
        self.lock_datasource = False
        self.first_initialized = False
        self.initializing = False
        self.waiting_init = False
        self.modeltraining = False
        self.ready_to_work = False
        self.stopped = False
        self.stop_working = False
        self.work_finished = False

        self.creator = creator
        self._stop_event = threading.Event()

        self.whereami = ''

    def set_training_environment(self, datasource_name, state_size, action_size, model, sess, optimizer, discount_factor):
        self.initializing = True

        if not self.first_initialized:
            self.first_initialized = True
        else:
            while not self.waiting_init:
                time.sleep(_time_sleep_for_work_)

        self._DATASOURCE_NAME_ = datasource_name
        self._DATASOURCE_ = {
            '_id_' : -1,
            '_state_' : np.array([]),
            '_action_' : -1,
            '_interactive_state_' : 'done'
        }
        self._EPSILON_ = 1.
        self._EPSILON_DECAY_ = .99995
        # self._EPSILON_DECAY_ = .995

        self.STACK_HISTORY_HOLDER_ACTOR_LOSS = deque(maxlen=10)
        self.STACK_HISTORY_HOLDER_CRITIC_LOSS = deque(maxlen=10)
        self.STACK_HISTORY_HOLDER_ACCURACY = deque(maxlen=111)
        self.STACK_HISTORY_HOLDER_MODEL_PREDICT = deque(maxlen=111)
        self.STACK_HISTORY_HOLDER_MODEL_ACTION_RANGE = deque(maxlen=111)

        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.states, self.actions, self.rewards, self.dones, self.y_values = [], [], [], [], []

        self.local_actor, self.local_critic = self.creator.build_model()
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

        self.initializing = False

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T

    def onehot(self, x):
        x = x.reshape(-1)
        return np.eye(len(x))[np.argmax(x)]

    def get_action(self, state):
        action_type = 1

        if np.random.random() < self._EPSILON_:
            action_probability = self.softmax(np.random.uniform(0., 1., self.action_size)).reshape((1, self.action_size))
            action_type = 0
        else:
            action_probability = self.local_actor.predict(state)

        self.STACK_HISTORY_HOLDER_MODEL_ACTION_RANGE.append(self.onehot(action_probability).reshape((1, self.action_size)))

        if self._EPSILON_ > 0.0001:
            self._EPSILON_ *= self._EPSILON_DECAY_

        return action_probability, action_type

    def get_accuracy(self):
        return 0 if len(self.STACK_HISTORY_HOLDER_ACCURACY) == 0 else np.mean(self.STACK_HISTORY_HOLDER_ACCURACY)

    def get_progressing(self):
        actor_loss = list(self.STACK_HISTORY_HOLDER_ACTOR_LOSS)
        critic_loss = list(self.STACK_HISTORY_HOLDER_CRITIC_LOSS)
        pridict = list(self.STACK_HISTORY_HOLDER_MODEL_PREDICT)
        action_range = list(self.STACK_HISTORY_HOLDER_MODEL_ACTION_RANGE)
        accuracy = list(self.STACK_HISTORY_HOLDER_ACCURACY)

        if np.sum(actor_loss) == 0 and np.sum(critic_loss) == 0 and np.sum(pridict):
            return 0

        mean_actor = 0 if len(actor_loss) == 0 else np.nan_to_num(np.mean(actor_loss))
        mean_critic = 0 if len(critic_loss) == 0 else np.nan_to_num(np.mean(critic_loss))
        mean_accuracy = 0 if len(accuracy) == 0 else np.nan_to_num(np.mean(accuracy))
        mean_model_predict = 0 if len(pridict) == 0 else np.nan_to_num(np.mean(pridict))
        action_range = np.zeros(7) if len(action_range) == 0 else np.round(np.mean(np.vstack(action_range), axis=0), 2)

        return {
            'LOSS': '%s%s %s%s %s' % (
                ' ' if mean_actor >= 0 else '',
                '{0:3.7f}'.format(mean_actor),
                ' ' if mean_critic >= 0 else '',
                '{0:3.7f}'.format(mean_critic),
                '{0:.4f}'.format(mean_accuracy)
            ),
            'PREDICT': '%s %s' % (
                '{0:.4f}'.format(mean_model_predict),
                action_range
            )
        }

    def append_sample(self, state, action_label, reward, done, y_value):
        self.states.append(state)
        self.actions.append(action_label)
        self.rewards.append(reward)
        self.dones.append(done)
        self.y_values.append(y_value)

    def set_observation(self, state, y_value, idx):
        while True:
            if self.lock_datasource:
                time.sleep(_time_sleep_for_work_)
                continue

            self.lock_datasource = True
            if self._DATASOURCE_['_interactive_state_'] == 'ready':
                self.lock_datasource = False
                time.sleep(_time_sleep_for_work_)
                continue

            self._DATASOURCE_ = {
                '_id_': idx,
                '_state_': state,
                '_action_': y_value,
                '_interactive_state_': 'ready'
            }
            self.lock_datasource = False
            break

    def get_observation(self, idx):
        while True:
            if self.stopped or self.initializing:
                return np.array([]), -1, -1

            if self.lock_datasource:
                time.sleep(_time_sleep_for_work_)
                continue
                
            self.lock_datasource = True

            if self._DATASOURCE_['_id_'] == idx:
                self.lock_datasource = False
                time.sleep(_time_sleep_for_work_)
                continue

            self._DATASOURCE_['_interactive_state_'] = 'done'
            self.lock_datasource = False

            break
            
        return self._DATASOURCE_['_state_'], self._DATASOURCE_['_action_'], self._DATASOURCE_['_id_']

    def step(self, action_probability, y_value):
        action = np.argmax(action_probability)
        done = ~(action == y_value)

        reward = 1.
        if done: reward = .0

        return reward, done

    def run(self):
        print('##### RUNNING : ', self)

        steps_an_episode = 15
        steps_current = 1
        failure_limit = 5
        failure_current = 0
        episode_current = 1
        state_id = -1
        done = False

        self.ready_to_work = True

        while not self.stopped:
            while failure_current <= failure_limit and steps_current <= steps_an_episode and not self.stopped:
                self.check_init()

                state, y_value, state_id = self.get_observation(state_id)
                if state_id == -1 or self.stopped:
                    break

                action_stochastic_policy, action_type = self.get_action(state)
                reward, done = self.step(action_stochastic_policy, y_value)

                self.append_sample(
                    state,
                    self.onehot(action_stochastic_policy).reshape((1, self.action_size)),
                    reward,
                    done,
                    y_value
                )
                self.STACK_HISTORY_HOLDER_MODEL_PREDICT.append(action_type)
                self.STACK_HISTORY_HOLDER_ACCURACY.append(int(~done))

                if done: failure_current += 1
                steps_current += 1

            if state_id != -1:
                self.train_model()
                self.update_local_model()
                
            steps_current = 1
            failure_current = 0
            episode_current += 1

        self.work_finished = True

    def check_init(self):
        self.waiting_init = True

        while self.initializing or self.sess._closed:
            time.sleep(_time_sleep_for_work_)
            if self.stopped: break

        self.waiting_init = False

    def stop(self):
        while not self.work_finished:
            time.sleep(1)

        self._stop_event.set()
        try:
            self._stop()
        except:
            pass

        print('##### TERMINATED : %s' % self)

    def is_stopped(self):
        return self._stop_event.is_set()

    def discounted_prediction(self):
        discounted_rewards = np.zeros_like(self.rewards)
        running_add = self.critic.predict(self.states[-1].reshape((1, self.state_size)))

        for reversed_idx in reversed(range(0, self.rewards.size)):
            if not self.dones[reversed_idx]:
                running_add = running_add * self.discount_factor + self.rewards[reversed_idx]
                discounted_rewards[reversed_idx] = running_add
            else:
                discounted_rewards[reversed_idx] = self.rewards[reversed_idx]

        return discounted_rewards

    def train_model(self):
        self.states = np.vstack(self.states)
        self.actions = np.vstack(self.actions)
        self.rewards = np.vstack(self.rewards)
        self.dones = np.vstack(self.dones)

        discounted_prediction = self.discounted_prediction()
        advantages = discounted_prediction - self.critic.predict(self.states)

        actor_loss = self.optimizer[0]([self.states, self.actions, advantages])
        critic_loss = self.optimizer[1]([self.states, discounted_prediction])

        self.STACK_HISTORY_HOLDER_ACTOR_LOSS.append(actor_loss[0])
        self.STACK_HISTORY_HOLDER_CRITIC_LOSS.append(critic_loss[0])

        self.states, self.actions, self.rewards, self.dones, self.y_values = [], [], [], [], []

    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())


class ModelSaver(threading.Thread):
    def __init__(self, creator):
        threading.Thread.__init__(self)

        self.stopped = False
        self.work_finished = False

        self.creator = creator
        self._stop_event = threading.Event()

    def run(self):
        print('##### RUNNING : ', self)

        time.sleep(60 * 60)

        self.check_for_start()

        while not self.stopped:
            if self.creator.hold_process:
                time.sleep(_time_sleep_for_work_)
                continue

            self.save()
            time.sleep(20)

        self.work_finished = True

    def save(self):
        mean_acc = int(np.mean(self.creator.get_accuracy()) * 100)
        if mean_acc < 90: return

        _MODEL_WEIGHT_DIR_ = 'graduated/%s/' % self.creator.current_datasource_name
        if not os.path.exists(_MODEL_WEIGHT_DIR_): os.makedirs(_MODEL_WEIGHT_DIR_)

        model_actor_located = '%s_MODEL_W_ACTOR_v%s.h5' % (_MODEL_WEIGHT_DIR_, mean_acc)
        model_critic_located = '%s_MODEL_W_CRITIC_v%s.h5' % (_MODEL_WEIGHT_DIR_, mean_acc)

        self.creator.actor.save_weights(model_actor_located)
        self.creator.critic.save_weights(model_critic_located)

        import logging
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(filename='%s_%s_build_log.log' % (_MODEL_WEIGHT_DIR_, datetime.today().strftime('%Y%m%d')), level=logging.DEBUG)
        logging.info(' ### Logging datatime : %s, Percentages of the process : 0.%s' % (
        datetime.today().strftime('%H:%M:%S'), mean_acc))

    def check_for_start(self):
        while True:
            agents_are_ready = []
            for agent in self.creator.AGENTS:
                agents_are_ready.append(int(agent.first_initialized))

            if sum(agents_are_ready) < self.creator.threads:
                time.sleep(_time_sleep_for_work_)
                continue
                
            break

    def stop(self):
        while not self.work_finished:
            continue

        self._stop_event.set()
        try:
            self._stop()
        except:
            pass

        print('##### TERMINATED : %s' % self)

    def is_stopped(self):
        return self._stop_event.is_set()


class Monitor(threading.Thread):
    def __init__(self, creator):
        threading.Thread.__init__(self)

        self.stopped = False
        self.work_finished = False

        self.creator = creator
        self._stop_event = threading.Event()

    def run(self):
        print('##### RUNNING : ', self)

        self.check_for_start()

        while not self.stopped:
            i = 0
            time.sleep(60)

            if self.creator.hold_process:
                continue

            not_available = False
            for agent in self.creator.AGENTS:
                if type(agent.get_progressing()) is int:
                    not_available = True
                    break

            if not not_available:
                seperator()
                for agent in self.creator.AGENTS:
                    print(i, agent.get_progressing())
                    i += 1

                print('##### ACCURACY OF THE CREATOR''s ACOTR: %s, DATA PROGRESS: %s #####' % (
                    '{0:.4f}'.format(self.creator.get_accuracy()),
                    self.creator.get_progressing()
                ))


        self.work_finished = True

    def check_for_start(self):
        while True:
            agents_are_ready = []
            for agent in self.creator.AGENTS:
                agents_are_ready.append(int(agent.first_initialized))

            if sum(agents_are_ready) < self.creator.threads:
                time.sleep(_time_sleep_for_work_)
                continue

            break

    def stop(self):
        while not self.work_finished:
            continue

        self._stop_event.set()
        try:
            self._stop()
        except:
            pass

        print('##### TERMINATED : %s' % self)

    def is_stopped(self):
        return self._stop_event.is_set()



if __name__ == "__main__":
    agent = A3CAgent()
    agent.train()

    print('##### All the trainings are finished. --------------------------------------------------------------')
