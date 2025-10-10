from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Load base model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# PPO config
config = PPOConfig(batch_size=16, learning_rate=1e-5)
ppo_trainer = PPOTrainer(config, model, tokenizer)

# Dummy rollout loop
for step in range(10):
    query = "Explain reinforcement learning in simple terms."
    inputs = tokenizer(query, return_tensors="pt")
    response = model.generate(**inputs, max_new_tokens=50)
    reward = float(len(response[0])) * 0.01  # placeholder reward
    ppo_trainer.step([query], [response], [reward])