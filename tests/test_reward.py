import torch
from transformers import AutoTokenizer, AutoModel
from src.models.reward import RewardModel

def test_reward_forward():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    rm = RewardModel(model, hidden_size=768)

    text = "This is a test."
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    score = rm(enc["input_ids"], enc["attention_mask"])
    assert score.shape == torch.Size([1])

def test_reward_save_load(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    rm = RewardModel(model, hidden_size=768)
    rm.save_pretrained(tmp_path)

    loaded = RewardModel.from_pretrained(tmp_path, base_path_or_name="gpt2")
    assert isinstance(loaded, RewardModel)