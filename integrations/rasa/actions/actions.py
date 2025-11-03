# actions/actions.py
import os, requests, json
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000/chat")

class ActionLlmCoach(Action):
    def name(self) -> Text:
        return "action_llm_coach"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Build history from the tracker
        events = tracker.events
        history = []
        last_user = None
        last_bot = None
        for e in events:
            if e.get("event") == "user":
                last_user = e.get("text", "")
                if last_bot is not None:
                    history.append({"user": last_user, "coach": last_bot})
                    last_user, last_bot = None, None
            elif e.get("event") == "bot":
                last_bot = e.get("text", "")
        # Take current message
        user_msg = tracker.latest_message.get("text","")
        payload = {"history": history[-6:], "user_msg": user_msg}
        try:
            r = requests.post(FASTAPI_URL, json=payload, timeout=30)
            r.raise_for_status()
            reply = r.json().get("reply","")
        except Exception as ex:
            reply = f"(backend error: {ex})"
        dispatcher.utter_message(text=reply)
        return []
