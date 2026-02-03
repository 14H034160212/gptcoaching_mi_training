
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8080/api"
USER_ID = "demo_vip_user"

def print_step(msg):
    print(f"\n{'='*60}")
    print(f"STEP: {msg}")
    print(f"{'='*60}")

def check_status(expected_stage=None):
    rep = requests.get(f"{BASE_URL}/journey/{USER_ID}")
    if rep.status_code != 200:
        print(f"Error checking status: {rep.text}")
        return {}
    data = rep.json()
    print(f"  -> Current Stage: {data['current_stage']}")
    print(f"  -> Can Advance: {data['can_advance']}")
    if expected_stage and data['current_stage'] != expected_stage:
        print(f"  !! WARNING: Expected {expected_stage}, got {data['current_stage']}")
    return data

def chat(msg):
    print(f"\nUser: {msg}")
    payload = {"user_id": USER_ID, "user_msg": msg}
    rep = requests.post(f"{BASE_URL}/chat", json=payload)
    if rep.status_code != 200:
        print(f"Error: {rep.text}")
        return
    data = rep.json()
    print(f"Kerrio: {data['reply'][:150]}...")
    return data

def main():
    print("Starting Kerrio Journey Walkthrough...")
    time.sleep(2) # Give server a moment
    
    # 0. Reset
    try:
        requests.post(f"{BASE_URL}/reset", json={"user_id": USER_ID})
    except Exception as e:
        print(f"Server not up yet? {e}")
        return

    # 1. Registration
    print_step("1. Registration & Validation")
    status = check_status("registration")
    
    print("  -> Attempting to advance without validation...")
    rep = requests.post(f"{BASE_URL}/journey/advance", json={"user_id": USER_ID})
    print(f"  -> Advance result: {rep.json()['success']} (Expected: False)")

    print("  -> Validating invite code 'KERRIO-VIP'...")
    rep = requests.post(f"{BASE_URL}/journey/validate", json={"user_id": USER_ID, "invite_code": "KERRIO-VIP"})
    print(f"  -> Validation result: {rep.json()['success']}")

    print("  -> Advancing to History Collection...")
    rep = requests.post(f"{BASE_URL}/journey/advance", json={"user_id": USER_ID})
    print(f"  -> Advance result: {rep.json()['success']}")
    check_status("history_collection")

    # 2. History Collection
    print_step("2. History Collection (The Three Pillars)")
    
    # Send some history data
    chat("Hi, I want to optimize my life.")
    chat("When I was a child, my parents were very demanding. I felt I had to be perfect.")
    chat("I believe that if I'm not productive, I'm worthless. That is my core belief.")
    chat("I sleep poorly, maybe 5 hours a night. I'm always stressed and tired.")

    # Check if we can advance
    print("  -> Checking if we gathered enough history...")
    status = check_status()
    if status.get('can_advance'):
        print("  -> Advancing to Consultation...")
        requests.post(f"{BASE_URL}/journey/advance", json={"user_id": USER_ID})
    else:
        print("  !! Failed to gather enough history.")
        # Force it for demo/testing if logic is strict? 
        # But our regex in kerrio_journey.py should catch "child", "parents", "believe", "sleep", "stressed"
        pass

    # 3. Consultation
    print_step("3. Consultation (Clinician's Notes)")
    check_status("consultation")
    
    # Chat to generate insights
    chat("I avoid starting big projects because I'm afraid they won't be good enough.")
    chat("I don't think that's true, I just like quality work.") # Resistance
    chat("Anyway, let's talk about something else.") # Deflection/Blind spot

    print("  -> Checking clinician insights...")
    rep = requests.get(f"{BASE_URL}/journey/notes/{USER_ID}")
    notes = rep.json()
    print(f"  -> Insights found: {len(notes.get('session_insights', []))}")
    for insight in notes.get('session_insights', []):
        print(f"    - {insight['category']}: {insight['observation']}")

    # Advance
    print("  -> Advancing to Diagnosis...")
    requests.post(f"{BASE_URL}/journey/advance", json={"user_id": USER_ID})

    # 4. Diagnosis
    print_step("4. Diagnosis & Cognitive Wiring Map")
    check_status("diagnosis")
    
    # Trigger diagnosis generation (happens on get_diagnosis or next chat)
    rep = requests.get(f"{BASE_URL}/journey/diagnosis/{USER_ID}")
    diag = rep.json().get('diagnosis', {})
    
    print("\n  === DIAGNOSIS GENERATED ===")
    print(f"  Explanation: {diag.get('explanation')}")
    print(f"  Core Constraints: {diag.get('core_constraints')}")
    print(f"  Root Causes: {diag.get('root_causes')}")
    print(f"  Recommended Videos: {len(diag.get('recommended_videos', []))}")

    print("\n  -> Confirming understanding...")
    requests.post(f"{BASE_URL}/journey/diagnosis/confirm/{USER_ID}")
    
    print("  -> Advancing to Proposal...")
    requests.post(f"{BASE_URL}/journey/advance", json={"user_id": USER_ID})

    # 5. Proposal
    print_step("5. Treatment Proposal")
    check_status("proposal")
    
    print("  -> Accepting Proposal...")
    rep = requests.post(f"{BASE_URL}/journey/proposal/accept", json={"user_id": USER_ID})
    print(f"  -> Proposal accepted: {rep.json().get('success')}")

    print("  -> Advancing to Treatment...")
    requests.post(f"{BASE_URL}/journey/advance", json={"user_id": USER_ID})

    # 6. Treatment
    print_step("6. Treatment & Monitoring")
    status = check_status("treatment")
    
    print("\nWalkthrough Complete! The user is now in Treatment phase.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
