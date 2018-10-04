'''''''''
### Registering own enviroment to the gym pakage enviroments

## C:\Tools\Anaconda3\Lib\site-packages\gym\envs\innostock\__init__.py
from gym.envs.innostock.market_learing import LearningEnv_v0


## C:\Tools\Anaconda3\Lib\site-packages\gym\envs\__init__.py
# Innostock
# ----------------------------------------
# stock market envs
register(
    id='Innostock-v0',
    entry_point='gym.envs.innostock:InnostockEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)


## C:\Tools\Anaconda3\Lib\site-packages\gym\scoreboard\__init__.py
add_group(
    id='Innostock-v0',
    name='innostock',
    description='Environments to predict the stock market values.'
)


## C:\Tools\Anaconda3\Lib\site-packages\gym\benchmarks\__init__.py
register_benchmark(
    id='InnoStock-v0',
    name='InnoStock',
    view_group="InnoStock",
    description='Stock market pridiction benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'Innostock-v0',
         'trials': 1,
         'max_timesteps': 2000,
        },
    ])
'''
