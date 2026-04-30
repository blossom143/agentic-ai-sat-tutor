from flask import Flask, render_template, request, jsonify, session
import numpy as np
import json
import random
from collections import deque
from openai import OpenAI
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── DeepSeek client ───────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ── Minimal SAT question bank (replace with load_questions() from your project)
QUESTIONS = {
    "Algebra": {
        "Easy": [
            {"question": "If 2x + 3 = 11, what is x?", "choices": {"A": "2", "B": "3", "C": "4", "D": "5"}, "correct_answer": "C"},
            {"question": "What is the slope of y = 3x - 7?", "choices": {"A": "7", "B": "-7", "C": "3", "D": "-3"}, "correct_answer": "C"},
            {"question": "Solve: 5x = 20", "choices": {"A": "2", "B": "4", "C": "5", "D": "100"}, "correct_answer": "B"},
        ],
        "Medium": [
            {"question": "If f(x) = x² - 4x + 3, what are the roots?", "choices": {"A": "1 and 3", "B": "-1 and -3", "C": "2 and 6", "D": "0 and 4"}, "correct_answer": "A"},
            {"question": "Solve the system: x + y = 5, 2x - y = 4", "choices": {"A": "x=2, y=3", "B": "x=3, y=2", "C": "x=1, y=4", "D": "x=4, y=1"}, "correct_answer": "B"},
        ],
        "Hard": [
            {"question": "If x² + bx + 9 = 0 has exactly one solution, what is b?", "choices": {"A": "3", "B": "6", "C": "9", "D": "±6"}, "correct_answer": "D"},
            {"question": "For what value of k does kx² - 4x + k = 0 have real solutions?", "choices": {"A": "k ≤ 2", "B": "k ≥ -2", "C": "-2 ≤ k ≤ 2", "D": "k ≠ 0"}, "correct_answer": "C"},
        ],
    },
    "Advanced Math": {
        "Easy": [
            {"question": "What is log₂(8)?", "choices": {"A": "2", "B": "3", "C": "4", "D": "8"}, "correct_answer": "B"},
            {"question": "Simplify: (x³)(x²)", "choices": {"A": "x⁵", "B": "x⁶", "C": "2x⁵", "D": "x"}, "correct_answer": "A"},
        ],
        "Medium": [
            {"question": "If g(x) = 2^x, what is g(3) - g(1)?", "choices": {"A": "4", "B": "6", "C": "8", "D": "2"}, "correct_answer": "B"},
            {"question": "Which expression is equivalent to (x+2)² - 4?", "choices": {"A": "x²+4", "B": "x(x+4)", "C": "x²+4x", "D": "x²-4"}, "correct_answer": "B"},
        ],
        "Hard": [
            {"question": "If h(x) = x³ - 3x² + 3x - 1, what is h(2)?", "choices": {"A": "0", "B": "1", "C": "2", "D": "8"}, "correct_answer": "B"},
        ],
    },
    "Problem-Solving and Data Analysis": {
        "Easy": [
            {"question": "A dataset has values 2, 4, 6, 8, 10. What is the mean?", "choices": {"A": "5", "B": "6", "C": "7", "D": "8"}, "correct_answer": "B"},
            {"question": "A store marks up items by 20%. An item costs $50. What is the selling price?", "choices": {"A": "$55", "B": "$60", "C": "$65", "D": "$70"}, "correct_answer": "B"},
        ],
        "Medium": [
            {"question": "A survey of 200 students shows 60% prefer math. How many prefer math?", "choices": {"A": "100", "B": "110", "C": "120", "D": "130"}, "correct_answer": "C"},
            {"question": "If y varies directly as x and y=15 when x=3, what is y when x=7?", "choices": {"A": "21", "B": "35", "C": "42", "D": "45"}, "correct_answer": "B"},
        ],
        "Hard": [
            {"question": "A dataset has median 10 and mean 12. Which best describes the distribution?", "choices": {"A": "Symmetric", "B": "Right-skewed", "C": "Left-skewed", "D": "Uniform"}, "correct_answer": "B"},
        ],
    },
    "Geometry and Trigonometry": {
        "Easy": [
            {"question": "What is the area of a circle with radius 4?", "choices": {"A": "8π", "B": "12π", "C": "16π", "D": "4π"}, "correct_answer": "C"},
            {"question": "A right triangle has legs 3 and 4. What is the hypotenuse?", "choices": {"A": "5", "B": "6", "C": "7", "D": "25"}, "correct_answer": "A"},
        ],
        "Medium": [
            {"question": "In a 30-60-90 triangle with hypotenuse 10, what is the shorter leg?", "choices": {"A": "4", "B": "5", "C": "5√2", "D": "5√3"}, "correct_answer": "B"},
            {"question": "What is sin(45°)?", "choices": {"A": "√2/2", "B": "√3/2", "C": "1/2", "D": "1"}, "correct_answer": "A"},
        ],
        "Hard": [
            {"question": "A triangle has sides 7, 24, 25. What type is it?", "choices": {"A": "Acute", "B": "Obtuse", "C": "Right", "D": "Equilateral"}, "correct_answer": "C"},
        ],
    },
}

ACTION_MAP = {
    0:  ("Algebra", "Easy"),
    1:  ("Algebra", "Medium"),
    2:  ("Algebra", "Hard"),
    3:  ("Advanced Math", "Easy"),
    4:  ("Advanced Math", "Medium"),
    5:  ("Advanced Math", "Hard"),
    6:  ("Problem-Solving and Data Analysis", "Easy"),
    7:  ("Problem-Solving and Data Analysis", "Medium"),
    8:  ("Problem-Solving and Data Analysis", "Hard"),
    9:  ("Geometry and Trigonometry", "Easy"),
    10: ("Geometry and Trigonometry", "Medium"),
    11: ("Geometry and Trigonometry", "Hard"),
    12: "Break",
}

DOMAINS = ["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]

# ── Student state helpers ─────────────────────────────────────────────────────

def init_student(levels, special_interest):
    knowledge = {}
    for domain, level in levels.items():
        base = {"Easy": 0.15, "Medium": 0.40, "Hard": 0.65}[level]
        knowledge[domain] = base + random.uniform(-0.05, 0.05)
    return {
        "knowledge": knowledge,
        "focus": random.uniform(0.55, 0.80),
        "frustration": random.uniform(0.05, 0.25),
        "novelty": 1.0,
        "progress_speed": 0.0,
        "cognitive_mode": "Neutral",
        "special_interest": special_interest if special_interest != "None" else None,
        "correct_streak": 0,
        "last_domain": None,
        "last_ten_perf": {d: [] for d in DOMAINS},
        "history": [],
        "step": 0,
    }

def get_cognitive_mode(state, domain=None):
    is_special = (domain == state["special_interest"]) if domain else False
    f, fr, n = state["focus"], state["frustration"], state["novelty"]
    if fr > 0.70:
        return "Overwhelmed"
    if is_special or (f > 0.65 and n > 0.55):
        return "Hyperfocused"
    if n < 0.35 or f < 0.35:
        return "Understimulated"
    return "Neutral"

def get_prob(state, domain, difficulty):
    k = state["knowledge"][domain]
    diff_factor = {"Easy": 0.75, "Medium": 0.45, "Hard": 0.20}[difficulty]
    base_p = 0.15 + (0.55 * k * diff_factor) + (0.18 * diff_factor)
    focus_mult = 0.75 + (state["focus"] * 0.50)
    calm_mult  = 0.85 + ((1 - state["frustration"]) * 0.30)
    p = base_p * focus_mult * calm_mult
    mode_mult = {"Hyperfocused": 1.25, "Neutral": 1.00,
                 "Understimulated": 0.80, "Overwhelmed": 0.40}[state["cognitive_mode"]]
    p *= mode_mult
    if domain == state["special_interest"]:
        p *= 1.15
    return float(np.clip(p + random.uniform(-0.05, 0.05), 0.08, 0.95))

def pick_action(state):
    """Simple heuristic policy: target weakest domain at appropriate difficulty."""
    knowledge = state["knowledge"]
    weakest = min(knowledge, key=knowledge.get)
    k = knowledge[weakest]
    if k < 0.35:
        diff = "Easy"
    elif k < 0.65:
        diff = "Medium"
    else:
        diff = "Hard"
    # Map to action int
    for action, val in ACTION_MAP.items():
        if val != "Break" and val[0] == weakest and val[1] == diff:
            return action, weakest, diff
    return 0, "Algebra", "Easy"

def update_state_after_answer(state, domain, difficulty, correct):
    k_gain = 0.06 if state["cognitive_mode"] == "Hyperfocused" else 0.025
    if state["cognitive_mode"] == "Overwhelmed":
        k_gain *= 0.05
    if correct:
        state["knowledge"][domain] = min(1.0, state["knowledge"][domain] + k_gain)
        state["frustration"] = max(0.0, state["frustration"] - 0.05)
        state["focus"] = min(1.0, state["focus"] + 0.05)
        state["correct_streak"] += 1
    else:
        if state["knowledge"][domain] > 0.05:
            state["knowledge"][domain] = max(0.0, state["knowledge"][domain] - 0.005)
        state["frustration"] = min(1.0, state["frustration"] + {"Easy": 0.08, "Medium": 0.12, "Hard": 0.20}[difficulty])
        state["focus"] = max(0.0, state["focus"] - 0.05)
        state["correct_streak"] = 0

    # Novelty
    if state["last_domain"] == domain:
        state["novelty"] = max(0.0, state["novelty"] - 0.08)
    else:
        state["novelty"] = min(1.0, state["novelty"] + 0.25)
    state["last_domain"] = domain

    # Rolling accuracy
    state["last_ten_perf"][domain].append(1.0 if correct else 0.0)
    if len(state["last_ten_perf"][domain]) > 10:
        state["last_ten_perf"][domain].pop(0)
    all_perf = [v for vals in state["last_ten_perf"].values() for v in vals]
    state["progress_speed"] = float(np.mean(all_perf)) if all_perf else 0.0

    # Update mode
    state["cognitive_mode"] = get_cognitive_mode(state, domain)
    state["step"] += 1

    # Log history
    state["history"].append({
        "step": state["step"],
        "domain": domain,
        "difficulty": difficulty,
        "correct": correct,
        "mode": state["cognitive_mode"],
        "knowledge": {d: round(v, 3) for d, v in state["knowledge"].items()},
        "focus": round(state["focus"], 3),
        "frustration": round(state["frustration"], 3),
        "novelty": round(state["novelty"], 3),
        "progress_speed": round(state["progress_speed"], 3),
    })
    return state

# ── DeepSeek CoT ──────────────────────────────────────────────────────────────

def generate_cot(state, domain, difficulty, question_text):
    k = state["knowledge"]
    prompt = f"""You are Hoot, a warm and encouraging AI tutor for a student with ADHD.

Student's current state:
- Cognitive Mode: {state['cognitive_mode']}
- Focus: {state['focus']:.2f}/1.0
- Frustration: {state['frustration']:.2f}/1.0  
- Novelty/Engagement: {state['novelty']:.2f}/1.0
- Progress Speed: {state['progress_speed']:.2f}/1.0
- Special Interest Domain: {state['special_interest'] or 'None'}
- Knowledge levels: {', '.join(f"{d}: {v:.0%}" for d, v in k.items())}

I've selected a {difficulty} {domain} question for this student.

Question: {question_text}

Think step by step:
1. Why is this question appropriate for this student's current ADHD state and knowledge level?
2. What strategy should this student use to approach it?
3. What encouraging message will help them focus?

Keep your response warm, concise (3-4 sentences max), and ADHD-friendly. 
Format as:
🧠 **Tutor thinking:** [your reasoning about why this question suits them]
💡 **Approach tip:** [brief strategy hint without giving the answer]
✨ **Encouragement:** [short motivating message]"""

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"🧠 **Tutor thinking:** I picked this {difficulty} {domain} question because it matches your current focus level.\n💡 **Approach tip:** Take it one step at a time.\n✨ **Encouragement:** You've got this!"

def get_question(domain, difficulty):
    pool = QUESTIONS.get(domain, {}).get(difficulty, [])
    if not pool:
        # Fallback
        return {"question": f"Sample {difficulty} {domain} question.", 
                "choices": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}, 
                "correct_answer": "A"}
    return random.choice(pool)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start():
    data = request.json
    levels = data.get("levels", {d: "Easy" for d in DOMAINS})
    special_interest = data.get("special_interest", "None")
    state = init_student(levels, special_interest)
    session["state"] = state
    # Pick first question
    action, domain, difficulty = pick_action(state)
    q = get_question(domain, difficulty)
    state["current_question"] = q
    state["current_domain"] = domain
    state["current_difficulty"] = difficulty
    session["state"] = state
    cot = generate_cot(state, domain, difficulty, q["question"])
    return jsonify({
        "cot": cot,
        "question": q["question"],
        "choices": q["choices"],
        "domain": domain,
        "difficulty": difficulty,
        "state": {
            "cognitive_mode": state["cognitive_mode"],
            "focus": round(state["focus"], 2),
            "frustration": round(state["frustration"], 2),
            "novelty": round(state["novelty"], 2),
            "knowledge": {d: round(v, 2) for d, v in state["knowledge"].items()},
        }
    })

@app.route("/api/answer", methods=["POST"])
def answer():
    data = request.json
    user_answer = data.get("answer", "").upper()
    state = session.get("state")
    if not state:
        return jsonify({"error": "Session expired"}), 400

    q = state["current_question"]
    domain = state["current_domain"]
    difficulty = state["current_difficulty"]
    correct = (user_answer == q["correct_answer"])

    state = update_state_after_answer(state, domain, difficulty, correct)

    # Pick next question
    action, next_domain, next_difficulty = pick_action(state)
    next_q = get_question(next_domain, next_difficulty)
    state["current_question"] = next_q
    state["current_domain"] = next_domain
    state["current_difficulty"] = next_difficulty
    session["state"] = state

    cot = generate_cot(state, next_domain, next_difficulty, next_q["question"])

    return jsonify({
        "correct": correct,
        "correct_answer": q["correct_answer"],
        "feedback": "Great job! 🎉" if correct else f"The correct answer was {q['correct_answer']}.",
        "cot": cot,
        "question": next_q["question"],
        "choices": next_q["choices"],
        "domain": next_domain,
        "difficulty": next_difficulty,
        "state": {
            "cognitive_mode": state["cognitive_mode"],
            "focus": round(state["focus"], 2),
            "frustration": round(state["frustration"], 2),
            "novelty": round(state["novelty"], 2),
            "progress_speed": round(state["progress_speed"], 2),
            "knowledge": {d: round(v, 2) for d, v in state["knowledge"].items()},
        }
    })

@app.route("/api/progress", methods=["GET"])
def progress():
    state = session.get("state")
    if not state:
        return jsonify({"error": "No session"}), 400
    history = state.get("history", [])
    mode_counts = {}
    for h in history:
        m = h["mode"]
        mode_counts[m] = mode_counts.get(m, 0) + 1
    correct_count = sum(1 for h in history if h["correct"])
    return jsonify({
        "total_questions": len(history),
        "correct": correct_count,
        "accuracy": round(correct_count / len(history), 2) if history else 0,
        "knowledge": {d: round(v, 2) for d, v in state["knowledge"].items()},
        "focus": round(state["focus"], 2),
        "frustration": round(state["frustration"], 2),
        "novelty": round(state["novelty"], 2),
        "progress_speed": round(state["progress_speed"], 2),
        "cognitive_mode": state["cognitive_mode"],
        "mode_distribution": mode_counts,
        "correct_streak": state["correct_streak"],
        "special_interest": state["special_interest"],
        "history": history[-20:],  # last 20 steps
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
