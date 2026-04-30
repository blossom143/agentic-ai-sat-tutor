# Demo - Dynamic Tutor Agent for K-12 Students with ADHD 
## Setup

```bash
pip install -r requirements.txt
```

Set your DeepSeek API key — either:
```bash
export DEEPSEEK_API_KEY="your_key_here"
```
Or create a `.env` file:
```
DEEPSEEK_API_KEY=your_key_here
```

## Run

```bash
cd adhd_tutor
python app.py
```
Then open http://localhost:5000

## Connecting your real question bank

In `app.py`, replace the `QUESTIONS` dict with your actual loader:

```python
# At the top of app.py, add:
import sys
sys.path.append('/path/to/your/project')
from src.utils.processor import load_questions

easy_df, medium_df, hard_df = load_questions()

def get_question(domain, difficulty):
    df = {"Easy": easy_df, "Medium": medium_df, "Hard": hard_df}[difficulty]
    pool = df[df["domain"] == domain]
    if pool.empty: pool = df
    row = pool.sample(1).iloc[0]
    q = row["question"]
    if isinstance(q, str):
        import json
        q = json.loads(q)
    return q
```

## Connecting your trained PPO model

Replace `pick_action()` in `app.py` with:

```python
from stable_baselines3 import PPO

trained_model = PPO.load("path/to/your/model.zip")

def pick_action(state):
    obs = build_obs(state)   # build the 9-dim obs vector
    action, _ = trained_model.predict(obs, deterministic=True)
    action_val = ACTION_MAP[int(action)]
    if action_val == "Break":
        return 12, None, None
    domain, difficulty = action_val
    return int(action), domain, difficulty
```

## Structure

```
adhd_tutor/
├── app.py              # Flask backend + student state logic
├── requirements.txt
├── templates/
│   └── index.html      # Full chat UI
└── static/
    └── owl.png         # Hoot mascot
```
