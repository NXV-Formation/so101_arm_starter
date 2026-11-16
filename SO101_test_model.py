# +   ++++++++++  ++  ++++++++++  ++  +++++++++++  +
# +    +++++++++  ++++  ++++++  +++++  +++++++++  ++
# +  ++  +++++++  +++++  ++++  +++++++  +++++++  +++
# +  +++   +++++  +++++++    ++++++++++  +++++   +++
# +  +++++  ++++  +++++++    +++++++++++  ++++  ++++
# +  +++++++  ++  +++++   +++  ++++++++++ +++  +++++
# +  ++++++++     ++++  ++++++  ++++++++++    ++++++
# +  ++++++++++   ++  ++++++++++  ++++++++   +++++++
# Copyright 2026 NXV https://nxv.fr
# License: CC BY-NC-SA 4.0
# https://creativecommons.org/licenses/by-nc-sa/4.0/

# Load the gym / mujoco env, as for the training
import gymnasium as gym
import so101_nexus.mujoco  
from so101_nexus import PickConfig, CubeObject

max_steps = 1024
config = PickConfig(objects=CubeObject(color="blue"))
env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array", config=config, max_episode_steps=max_steps)

# To save the video as a gif image
from matplotlib import animation
import matplotlib.pyplot as plt

def save_frames_as_gif(frames, path='./', filename='animation.gif'):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='pillow', fps=60)

# Load the keras model
import keras
import numpy as np

model = keras.saving.load_model("mymodel.keras")
model.compile()
model.summary()

# Main loop to infer 1 scenario and save as gif image
state, info = env.reset()
frames = []

for _ in range(max_steps):
    values = model.predict(state[np.newaxis], verbose=0)
    action = np.argmax(values[0])
    state, reward, done, info = env.step(action)
    if done:
        state, info = env.reset()
    img = env.render()
    frames.append(img)

save_frames_as_gif(frames, filename='animation.gif')
env.close()
