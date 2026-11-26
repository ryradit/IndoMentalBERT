"""
Custom actions for Rasa integrating IndoMentalBERT.
Queries the trained PyTorch checkpoint to perform dynamic semantic QA matching on data.csv.
"""

import os
import sys
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Setup path to project root
ACTION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ACTION_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ActionSemanticQARetrieval(Action):
    def __init__(self):
        super(ActionSemanticQARetrieval, self).__init__()
        self.model = None
        self.tokenizer = None
        self.candidate_pool = None
        self.device = None

    def name(self) -> Text:
        return "action_semantic_qa_retrieval"

    def _lazy_load_model(self):
        if self.model is None:
            try:
                from demo import load_model_and_pool
                ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "indomentalbert", "indomentalbert_best.pt")
                csv_path = os.path.join(PROJECT_ROOT, "data.csv")
                self.model, self.tokenizer, self.candidate_pool, self.device = load_model_and_pool(
                    model_checkpoint=ckpt_path,
                    data_path=csv_path,
                    sample_pool_size=100
                )
            except Exception as e:
                print(f"[!] Warning: Could not lazy-load IndoMentalBERT: {e}")

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get("text", "")
        self._lazy_load_model()

        if self.model is not None and self.candidate_pool:
            from demo import query_counselor
            top_matches = query_counselor(
                user_message,
                self.model,
                self.tokenizer,
                self.candidate_pool,
                self.device,
                top_k=1
            )
            if top_matches:
                best_response, score = top_matches[0]
                dispatcher.utter_message(text=best_response)
                return []

        dispatcher.utter_message(
            text="Saya memahami ini situasi yang tidak mudah bagi Anda. Silakan ceritakan lebih lanjut, atau pertimbangkan untuk berkonsultasi langsung dengan psikolog berlisensi."
        )
        return []
