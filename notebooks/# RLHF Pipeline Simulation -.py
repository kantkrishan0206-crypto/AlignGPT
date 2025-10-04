# RLHF Pipeline Simulation - corrected and runnable version
# Paste this into a Jupyter cell (or split into cells as you prefer).

# ---------------------------
# Cell 1: Setup, Imports, Config
# ---------------------------
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import functional as F

from collections import Counter

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
sns.set_theme(style="whitegrid")

# Configuration
VOCAB_SIZE = 1000
MAX_SEQ_LEN = 128
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
LR_RM = 1e-4
LR_POLICY = 5e-5
EPOCHS_RM = 3
EPOCHS_PPO = 2
BATCH_SIZE = 8

PPO_CLIP = 0.2
PPO_EPOCHS = 3
KL_COEF = 0.05
TARGET_KL = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Cell 2: Data Loading & Synthetic Data
# ---------------------------
data_filepath = "article_level_data.csv"
try:
    df_raw = pd.read_csv(data_filepath)
    print(f"Loaded {data_filepath} shape={df_raw.shape}")
except FileNotFoundError:
    print("File not found — generating synthetic fallback data.")
    df_raw = pd.DataFrame({
        'article': [
            "NLP is a multidisciplinary field that draws from linguistics and computer science.",
            "The quick brown fox jumps over the lazy dog.",
            "Alignment aims to ensure AI systems follow human values.",
            "Reinforcement Learning from Human Feedback is key to modern LLM safety.",
            "Policy optimization adjusts the LLM's response probabilities.",
            "A reward model learns pairwise human preferences.",
        ] * 10,
        'class': [0, 1, 0, 1, 0, 1] * 10
    })
    df_raw = df_raw.sample(frac=1).reset_index(drop=True).head(50)

# Tokenizer & Vocab
def build_vocab(texts, max_vocab_size=VOCAB_SIZE):
    all_tokens = ' '.join(texts).lower().split()
    counts = Counter(all_tokens)
    most_common = counts.most_common(max_vocab_size - 3)
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2}
    for token, _ in most_common:
        vocab[token] = len(vocab)
    return vocab

vocab = build_vocab(df_raw['article'].tolist())
TOKEN_TO_ID = vocab
ID_TO_TOKEN = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)
print(f"Vocab size: {VOCAB_SIZE}")

def tokenize_text(text, max_len=MAX_SEQ_LEN):
    tokens = text.lower().split()
    ids = [TOKEN_TO_ID.get(t, TOKEN_TO_ID['<unk>']) for t in tokens]
    ids = [TOKEN_TO_ID['<bos>']] + ids
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [TOKEN_TO_ID['<pad>']] * (max_len - len(ids))
    return ids[:max_len]

def generate_synthetic_preference_data(df):
    preference_data = []
    articles = df['article'].tolist()
    classes = df['class'].tolist()
    for article, class_label in zip(articles, classes):
        mid = len(article) // 2
        prompt = article[:mid].strip()
        article_response = article[mid:].strip()

        if class_label == 1:
            chosen = article_response
            rejected = chosen.replace('LLM', 'AI').replace('science', 'magic')
            if chosen == rejected: rejected += ' (corrupted)'
            preference_data.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected, 'label': 1})
        else:
            rejected = article_response
            positive_articles = [a[len(a)//2:].strip() for a, c in zip(articles, classes) if c == 1]
            if positive_articles:
                chosen = random.choice(positive_articles)
                preference_data.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected, 'label': 1})

    tokenized = []
    for item in preference_data:
        prompt_ids = tokenize_text(item['prompt'])
        chosen_ids = tokenize_text(item['chosen'])
        rejected_ids = tokenize_text(item['rejected'])

        chosen_input = prompt_ids + chosen_ids
        rejected_input = prompt_ids + rejected_ids

        # truncate/pad to 2 * MAX_SEQ_LEN
        def fix_len(lst, L=2*MAX_SEQ_LEN):
            if len(lst) > L:
                return lst[:L]
            else:
                return lst + [TOKEN_TO_ID['<pad>']] * (L - len(lst))

        tokenized.append({
            'prompt_ids': torch.tensor(prompt_ids, dtype=torch.long),
            'chosen_input': torch.tensor(fix_len(chosen_input), dtype=torch.long),
            'rejected_input': torch.tensor(fix_len(rejected_input), dtype=torch.long),
            'label': item['label']
        })

    return tokenized

rlhf_data = generate_synthetic_preference_data(df_raw)
train_size = int(0.8 * len(rlhf_data))
rm_train_data = rlhf_data[:train_size]
rm_val_data = rlhf_data[train_size:]

class PreferenceDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

rm_train_loader = DataLoader(PreferenceDataset(rm_train_data), batch_size=BATCH_SIZE, shuffle=True)
rm_val_loader = DataLoader(PreferenceDataset(rm_val_data), batch_size=BATCH_SIZE, shuffle=False)

print(f"Generated {len(rlhf_data)} preference pairs. Train={len(rm_train_data)}, Val={len(rm_val_data)}")

# ---------------------------
# Cell 3: Models
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x, attn_mask=None):
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + attn_output
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output
        return x

class BaseLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len):
        super().__init__()
        max_total_len = max_seq_len * 2
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_total_len, embed_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        # returns hidden states (B, T, D)
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).repeat(B, 1)
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        for block in self.transformer_blocks:
            x = block(x)
        return x

    def get_logits(self, input_ids):
        # convenience to get logits (B, T, V)
        hidden = self.forward(input_ids)
        logits = self.lm_head(hidden)
        return logits

class RewardModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len):
        super().__init__()
        self.transformer_backbone = BaseLanguageModel(vocab_size, embed_dim, num_layers, num_heads, max_seq_len)
        self.reward_head = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        hidden = self.transformer_backbone(input_ids)  # (B, T, D)
        last_hidden = hidden[:, -1, :]                 # (B, D)
        reward = self.reward_head(self.dropout(last_hidden))  # (B,1)
        return reward.squeeze(-1)

reward_model = RewardModel(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN).to(device)
print("Reward model initialized.")

# ---------------------------
# Cell 4: Reward Model Training
# ---------------------------
def preference_loss(chosen_rewards, rejected_rewards):
    diff = chosen_rewards - rejected_rewards
    return -F.logsigmoid(diff).mean()

def evaluate_reward_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            chosen = batch['chosen_input'].to(device)
            rejected = batch['rejected_input'].to(device)
            r_chosen = model(chosen)
            r_rejected = model(rejected)
            total_loss += preference_loss(r_chosen, r_rejected).item()
            correct += (r_chosen > r_rejected).sum().item()
            total += r_chosen.shape[0]
    model.train()
    return total_loss / len(data_loader), correct / total if total>0 else 0.0

def train_reward_model(model, train_loader, val_loader, epochs, lr):
    opt = Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val = float('inf')
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"RM Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            chosen = batch['chosen_input'].to(device)
            rejected = batch['rejected_input'].to(device)
            r_chosen = model(chosen)
            r_rejected = model(rejected)
            loss = preference_loss(r_chosen, r_rejected)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        val_loss, val_acc = evaluate_reward_model(model, val_loader)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}: train={avg_train:.4f} val={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_reward_model.pth")
            print("Saved best RM.")
    return history

rm_history = train_reward_model(reward_model, rm_train_loader, rm_val_loader, EPOCHS_RM, LR_RM)

# quick plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(rm_history['train_loss'], marker='o', label='train')
plt.plot(rm_history['val_loss'], marker='x', label='val')
plt.legend(); plt.title("RM Loss")
plt.subplot(1,2,2)
plt.plot(rm_history['val_acc'], marker='s', label='val_acc')
plt.legend(); plt.title("RM Val Acc")
plt.show()

# ---------------------------
# Cell 5: PPO Trainer Setup
# ---------------------------
policy_model = BaseLanguageModel(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN).to(device)
reference_model = BaseLanguageModel(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN).to(device)
reference_model.load_state_dict(policy_model.state_dict())
for p in reference_model.parameters(): p.requires_grad = False
reference_model.eval()

class PPOTrainer:
    def __init__(self, policy, ref_policy, reward_model, optimizer, kl_coef, max_len):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.max_len = max_len

    @torch.no_grad()
    def generate_and_get_rewards(self, prompt_ids_batch):
        generated = []
        for prompt_ids in prompt_ids_batch:
            # prompt_ids: already a tensor shape (MAX_SEQ_LEN,)
            prompt_ids = prompt_ids.unsqueeze(0).to(device)  # (1, Lp)
            Lp = prompt_ids.shape[1]
            current_ids = prompt_ids.clone()
            chosen_log_probs = []
            ref_log_probs = []

            # Greedy generation step-by-step (cap to max_len)
            for _ in range(self.max_len):
                logits = self.policy.get_logits(current_ids)[:, -1, :]  # (1, V)
                ref_logits = self.ref_policy.get_logits(current_ids)[:, -1, :]
                logp = F.log_softmax(logits, dim=-1)  # (1,V)
                reflogp = F.log_softmax(ref_logits, dim=-1)
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)  # (1,1)
                # record log-probs for chosen token
                lp = torch.gather(logp, 1, next_token).squeeze(0).squeeze(-1)  # scalar tensor
                rlp = torch.gather(reflogp, 1, next_token).squeeze(0).squeeze(-1)
                chosen_log_probs.append(lp)      # list of scalar tensors
                ref_log_probs.append(rlp)
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                # stop if pad token generated
                if next_token.item() == TOKEN_TO_ID['<pad>']:
                    break

            response_ids = current_ids[:, Lp:].squeeze(0)  # (L_resp,)
            # prepare RM input: pad/truncate to 2*MAX_SEQ_LEN
            rm_input = current_ids.squeeze(0).cpu().numpy().tolist()
            L_needed = 2 * MAX_SEQ_LEN
            if len(rm_input) > L_needed:
                rm_input = rm_input[:L_needed]
            else:
                rm_input = rm_input + [TOKEN_TO_ID['<pad>']] * (L_needed - len(rm_input))
            rm_input_tensor = torch.tensor(rm_input[:2*MAX_SEQ_LEN], dtype=torch.long).unsqueeze(0).to(device)
            reward_score = self.reward_model(rm_input_tensor)  # tensor scalar

            if len(chosen_log_probs) == 0:
                # handle no-generation case
                chosen_log_probs = torch.tensor([0.0], device=device)
                ref_log_probs = torch.tensor([0.0], device=device)
            else:
                chosen_log_probs = torch.stack(chosen_log_probs).detach()  # (L_resp,)
                ref_log_probs = torch.stack(ref_log_probs).detach()

            generated.append({
                'prompt_ids': prompt_ids.squeeze(0).detach().cpu(),
                'response_ids': response_ids.detach().cpu(),
                'log_probs': chosen_log_probs.cpu(),      # store on cpu for later conversion
                'ref_log_probs': ref_log_probs.cpu(),
                'reward': reward_score.detach().cpu()
            })
        return generated

    def compute_loss(self, sample):
        # sample fields are tensors on device
        prompt_ids = sample['prompt_ids']  # (Lp,)
        response_ids = sample['response_ids']  # (Lr,)
        old_log_probs = sample['log_probs']  # (Lr,)
        ref_log_probs_old = sample['ref_log_probs']  # (Lr,)
        reward = sample['reward']  # scalar tensor

        # build full sequence and run through policy to get current log-probs
        full_ids = torch.cat([prompt_ids, response_ids], dim=-1).unsqueeze(0).to(device)  # (1, Lp+Lr)
        Lp = prompt_ids.shape[0]
        Lr = response_ids.shape[0]

        logits = self.policy.get_logits(full_ids)  # (1, L, V)
        # The logits that predict tokens at positions of response tokens are at indices [Lp-1 : Lp-1+Lr]
        start = max(0, Lp-1)
        logits_slice = logits[:, start:start+Lr, :]  # (1, Lr, V)
        logp_current = F.log_softmax(logits_slice, dim=-1)
        response_ids = response_ids.unsqueeze(0).to(device)  # (1, Lr)
        current_log_probs = torch.gather(logp_current, 2, response_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (Lr,)

        # make sure shapes match
        old_log_probs = old_log_probs.to(device)
        ref_log_probs_old = ref_log_probs_old.to(device)

        ratio = torch.exp(current_log_probs - old_log_probs.to(device))
        kl_div = (current_log_probs - ref_log_probs_old.to(device)).mean()

        # simplified advantage
        advantage = (reward.to(device) - self.kl_coef * kl_div).detach()
        # reduce advantage to a scalar to broadcast
        advantage_scalar = advantage

        surr1 = ratio * advantage_scalar
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantage_scalar
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss, kl_div.item(), reward.item()

    def ppo_update(self, generated_data):
        self.policy.train()
        total_loss = 0.0
        total_kl = 0.0
        total_reward = 0.0
        n = len(generated_data)
        for _ in range(PPO_EPOCHS):
            for sample in generated_data:
                # convert sample fields to tensors on device
                sample_on_device = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(device))
                    for k, v in sample.items()
                }
                self.optimizer.zero_grad()
                loss, kl, rew = self.compute_loss(sample_on_device)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_kl += kl
                total_reward += rew
        denom = (PPO_EPOCHS * n) if n>0 else 1
        return total_loss/denom, total_kl/denom, total_reward/n if n>0 else 0.0

# initialize trainer
ppo_optimizer = Adam(policy_model.parameters(), lr=LR_POLICY)
ppo_trainer = PPOTrainer(policy_model, reference_model, reward_model, ppo_optimizer, KL_COEF, max_len=20)

# PPO prompts loader (list of prompt tensors)
ppo_prompts = [item['prompt_ids'] for item in rlhf_data]
ppo_loader = DataLoader(ppo_prompts, batch_size=BATCH_SIZE, shuffle=True)
print("PPO trainer ready.")

# ---------------------------
# Cell 6: PPO Fine-tuning Execution
# ---------------------------
def execute_ppo_fine_tuning(trainer, data_loader, epochs):
    history = {'ppo_loss': [], 'kl_div': [], 'avg_reward': []}
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_reward = 0.0
        batches = 0
        for prompt_batch in tqdm(data_loader, desc=f"PPO Epoch {epoch+1}/{epochs}"):
            generated = trainer.generate_and_get_rewards(prompt_batch)
            ppo_loss, kl_div, avg_reward = trainer.ppo_update(generated)
            epoch_loss += ppo_loss
            epoch_kl += kl_div
            epoch_reward += avg_reward
            batches += 1
        if batches == 0: batches = 1
        avg_loss = epoch_loss / batches
        avg_kl = epoch_kl / batches
        avg_reward = epoch_reward / batches

        history['ppo_loss'].append(avg_loss)
        history['kl_div'].append(avg_kl)
        history['avg_reward'].append(avg_reward)

        print(f"PPO Epoch {epoch+1}: loss={avg_loss:.4f}, kl={avg_kl:.4f}, reward={avg_reward:.4f}")

        # adaptive KL adjust (toy)
        if avg_kl > 1.5 * TARGET_KL and trainer.kl_coef < 0.5:
            trainer.kl_coef *= 1.5
            print("Increased KL coef to", trainer.kl_coef)
        elif avg_kl < TARGET_KL / 1.5 and trainer.kl_coef > 0.005:
            trainer.kl_coef /= 1.5
            print("Decreased KL coef to", trainer.kl_coef)

    torch.save(trainer.policy.state_dict(), "final_policy_model.pth")
    return history

ppo_history = execute_ppo_fine_tuning(ppo_trainer, ppo_loader, EPOCHS_PPO)

# quick PPO plots
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(ppo_history['ppo_loss'], marker='o'); plt.title("PPO Loss")
plt.subplot(1,3,2)
plt.plot(ppo_history['kl_div'], marker='x'); plt.title("KL")
plt.subplot(1,3,3)
plt.plot(ppo_history['avg_reward'], marker='s'); plt.title("Avg Reward")
plt.tight_layout(); plt.show()

# ---------------------------
# Cell 7: Ablation Simulation (unchanged logic but safe)
# ---------------------------
def simulate_ablation_metrics(ppo_history, rm_history):
    ablation_results = []
    kl_betas = [0.01, 0.05, 0.1, 0.5]
    base_reward = ppo_history['avg_reward'][-1] if len(ppo_history['avg_reward'])>0 else 0.1
    base_kl = ppo_history['kl_div'][-1] if len(ppo_history['kl_div'])>0 else 0.01
    for beta in kl_betas:
        avg_reward_score = (base_reward * (0.05 / (beta+1e-8))) * 0.95 + 0.1
        robustness_metric = 1.0 / (beta * 10 + 1e-8) + 0.7
        fairness_metric = (1.0 - beta) * 0.9 + 0.05
        avg_reward_score = max(0.1, min(10.0, avg_reward_score * 5))
        robustness_metric = max(0.7, min(10.0, robustness_metric * 1.5))
        fairness_metric = max(0.5, min(10.0, fairness_metric * 1.2))
        ablation_results.append({
            'KL_Beta': beta,
            'Reward_Alignment': avg_reward_score,
            'Robustness_Metric': robustness_metric,
            'Fairness_Metric': fairness_metric
        })
    df = pd.DataFrame(ablation_results)
    df['Safety_Score'] = (df['Robustness_Metric'] + df['Fairness_Metric'])/2
    print(df.round(4))
    return df

ablation_df = simulate_ablation_metrics(ppo_history, rm_history)

# ---------------------------
# Cell 8: Ablation Plots
# ---------------------------
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.scatterplot(data=ablation_df, x='Safety_Score', y='Reward_Alignment', size='KL_Beta', sizes=(50,300), hue='KL_Beta', palette='viridis', legend='brief', alpha=0.8)
for i,row in ablation_df.iterrows():
    plt.text(row['Safety_Score']+0.02, row['Reward_Alignment']-0.02, f"β={row['KL_Beta']}")
plt.title("Alignment vs Safety (sim)")
plt.xlabel("Safety Score")
plt.ylabel("Reward Alignment")

plt.subplot(1,2,2)
melted = ablation_df.melt(id_vars='KL_Beta', value_vars=['Reward_Alignment','Robustness_Metric','Fairness_Metric'], var_name='Metric', value_name='Value')
sns.lineplot(data=melted, x='KL_Beta', y='Value', hue='Metric', marker='o')
plt.xscale('log')
plt.title("Metrics vs KL Beta")
plt.tight_layout(); plt.show()

# ---------------------------
# Cell 9: Inference & RM Comparison
# ---------------------------
def generate_response(model, prompt_text, max_new_tokens=20):
    model.eval()
    prompt_ids = torch.tensor(tokenize_text(prompt_text, max_len=MAX_SEQ_LEN)).unsqueeze(0).to(device)
    current_ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.get_logits(current_ids)[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        current_ids = torch.cat([current_ids, next_token], dim=-1)
        if next_token.item() == TOKEN_TO_ID['<pad>']:
            break
    response_ids = current_ids.squeeze(0).cpu().numpy()[prompt_ids.shape[1]:]
    decoded = [ID_TO_TOKEN.get(int(i), '<unk>') for i in response_ids if int(i) != TOKEN_TO_ID['<pad>']]
    return ' '.join(decoded)

def get_reward_score(rm_model, prompt, response):
    rm_model.eval()
    full_text = prompt + " " + response
    ids = tokenize_text(full_text, max_len=MAX_SEQ_LEN)
    # replicate/pad to length 2*MAX_SEQ_LEN for RM input expectation
    ids = ids + ids
    ids = ids[:2*MAX_SEQ_LEN]
    tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        score = rm_model(tensor).item()
    return score

example_prompt = df_raw.iloc[0]['article'].split('.',1)[0] + '.'
print("PROMPT:", example_prompt)

ref_response = generate_response(reference_model, example_prompt)
ref_score = get_reward_score(reward_model, example_prompt, ref_response)
print("Reference response:", ref_response)
print("RM score (ref):", ref_score)

policy_response = generate_response(policy_model, example_prompt)
policy_score = get_reward_score(reward_model, example_prompt, policy_response)
print("Policy response:", policy_response)
print("RM score (policy):", policy_score)

if policy_score > ref_score:
    print("SUCCESS: policy > reference according to RM.")
else:
    print("NOTE: policy did not get higher RM score in this toy sim.")
