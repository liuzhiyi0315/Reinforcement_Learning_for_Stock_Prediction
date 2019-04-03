from agent.agent import Agent
from functions import *
import sys

# if len(sys.argv) != 4:
# 	print("Usage: python train.py [stock] [window] [episodes]")
# 	exit()

# stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stock_name, window_size, episode_count = '^GSPC', 20, 600

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
	message = "Episode " + str(e) + "/" + str(episode_count)
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	action_sum = {0: 0, 1: 0, 2: 0}

	pp = PercentagePrinter(message, l)
	for t in range(l):
		action = agent.act(state)
		action_sum[action] += 1

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			# print("Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			# print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

		pp.print(t)

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print(f"buy {action_sum[1]}, sell {action_sum[2]}, hold {action_sum[0]}")
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	pp.final_print()

	# if e % 10 == 0:
	# 	agent.model.save("models/model_ep" + str(e))
