import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model

# Load model hasil training
model = load_model("cartpole_dqn.h5", compile=False)

# Buat environment CartPole
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Jumlah episode untuk testing
test_episodes = 5

for e in range(test_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    time = 0

    while True:
        # Pilih action dari model (argmax Q-value)
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0])

        # Jalankan action di environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        state = next_state
        total_reward += reward
        time += 1

        if done:
            print(f"Test Episode: {e+1}, Score: {time}, Total Reward: {total_reward}")
            break

env.close()
