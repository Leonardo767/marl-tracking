from particle_env.make_env import make_env
import time

env = make_env('simple')
env.render()
time.sleep(0.5)
