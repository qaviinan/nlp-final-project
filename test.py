# Check if there's anything RL related in types
import tinker.types as t
rl_related = [x for x in dir(t) if any(k in x.lower() for k in ['grpo', 'rl', 'reward', 'policy', 'reinforce'])]
print(rl_related)