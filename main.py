import logging
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from mariq.models import DQN
from mariq.gym_wrappers import mario_wrapper
from nes_py.wrappers import JoypadSpace


# logger
# logging.basicConfig(filename='mariQ.log', format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger('mariQ')
logger.setLevel(level=logging.INFO)
fileHandler = logging.FileHandler('mariQ.log')
logger.addHandler(fileHandler)
logger.info('---- Training started ----')


# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = mario_wrapper(env)

# Uncomment for video snapshots
# env = gym.wrappers.Monitor(env, './video/',video_callable=lambda episode_id : episode_id % 500 == 0, force = True)

dqn = DQN(env, logger)
dqn.train(10)
# dqn.play()

