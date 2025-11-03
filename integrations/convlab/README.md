# ConvLab-3 Bridge (Template)

This folder contains a minimal `policy_bridge.py` showing how to treat the MI LLM as a *policy* that reads a DST state + history and returns an action/utterance.

To integrate with ConvLab-3:
1. Install ConvLab-3 (per their docs) and identify the dialogue policy interface (e.g., a class with `.predict(state)`).
2. Wrap the logic in `policy_bridge.py` into a policy class that:
   - builds the MI prompt from DST `state` and `history`,
   - calls the LLM for a response,
   - (optional) parses the response into a template action or NLG output.
3. Insert your policy into ConvLab-3's pipeline and run their evaluation harness.
