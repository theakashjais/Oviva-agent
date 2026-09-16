import os
import json
import csv
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a triage assistant for Oviva, a digital chronic-care coaching platform.
Classify each incoming patient message into exactly one category:

- appointment: scheduling, rescheduling, or cancelling a coaching session
- coaching_health: questions about diet, symptoms, medication, or progress
- technical: app bugs, login issues, syncing problems
- billing_admin: payment, insurance, account/paperwork questions
- escalate: vague, urgent, safety-related, or doesn't clearly fit any category above

Any message mentioning symptoms that could indicate a medical emergency (chest pain, shortness of breath, severe dizziness, fainting) must always be classified as escalate, even if it references diet, medication, billing, appointments, or the coaching program.

Requests to change or replace a coach due to dissatisfaction, poor fit, or not feeling understood are relationship/satisfaction issues, not scheduling requests. Classify these as escalate, not appointment, even though the word "change" appears. Only classify as appointment when the request is purely about date/time of a session.

Respond with ONLY valid JSON in this exact format, no other text:
{"category": "...", "confidence": 0.0, "draft_reply": "..."}

confidence is your own estimate between 0 and 1.
draft_reply is a short 2-3 sentence reply a human coach could send after light editing."""

def classify(message: str) -> dict:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
        temperature=0
    )
    raw = response.choices[0].message.content
    return json.loads(raw)

TEST_SET = [
    ("Can I reschedule my appointment to next week?", "appointment"),
    ("Kann ich meinen Termin auf Montag verschieben?", "appointment"),
    ("Can I still have a small piece of cake on my son's birthday?", "coaching_health"),
    ("Ich habe seit der neuen Diaet Kopfschmerzen, ist das normal?", "coaching_health"),
    ("The app won't let me log in since the last update", "technical"),
    ("Meine Schritte werden nicht mehr synchronisiert", "technical"),
    ("Wird das Programm von meiner Krankenkasse uebernommen?", "billing_admin"),
    ("I was charged twice this month, can you check?", "billing_admin"),
    ("I don't feel good, please call me as soon as possible", "escalate"),
    ("Ich weiss nicht mehr weiter", "escalate"),
    ("I'm not sure if my insurance covers this app, can someone from the team explain what to do next?", "billing_admin"),
    ("I skipped my coaching call because the app crashed", "technical"),
    ("Since starting the program I've felt dizzy, should I come in?", "escalate"),
    ("My card was declined during renewal, but the app still says I have access", "billing_admin"),
    ("I want to change my coach, this one doesn't understand me", "escalate"),
    ("I have chest pain and shortness of breath, is that from the diet?", "escalate"),
    ("I'm dizzy and my card was also declined, and I missed my appointment because of it - what do I do?", "escalate"),
]

def run_evaluation():
    correct = 0
    rows = []

    for message, expected in TEST_SET:
        result = classify(message)
        predicted = result["category"]
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        rows.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": result["confidence"]
        })

        status = "OK" if is_correct else "MISS"
        print(f"{status} expected={expected:16} predicted={predicted:16} | {message[:50]}")

    accuracy = correct / len(TEST_SET)
    print(f"\nAccuracy: {correct}/{len(TEST_SET)} = {accuracy:.0%}")

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["message", "expected", "predicted", "correct", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print("Saved details to results.csv")

if __name__ == "__main__":
    run_evaluation()
