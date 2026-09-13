import os
import json
import logging
import asyncio
from google import genai
from google.genai import types
import PIL.Image
from pydantic import BaseModel, Field
from typing import Literal, List

class FinancialDecision(BaseModel):
    request_id: str = Field(description="The unique ID of the request being evaluated.")
    amount_safe_to_pay: float = Field(description="The maximum amount the user can safely pay on request_date.")
    affordability_status: Literal['affordable_now', 'affordable_with_plan', 'affordable_later', 'not_affordable']
    recommended_payment_method: Literal['full_payment', 'partial_payment', 'installments', 'wait', 'not_recommended']
    payment_plan: str = Field(description="Format exactly: YYYY-MM-DD:amount|YYYY-MM-DD:amount. Use 'none' if empty.")
    earliest_date_for_full_payment: str = Field(description="YYYY-MM-DD format. Leave empty string '' if not safe within forecast. Must equal request_date if affordable_now.")
    spending_changes_needed: str = Field(description="Format exactly: stop:<event_id>|reduce_to:<event_id>:<new_amount>. Max 3 changes. Use 'none' if empty.")
    decision_explanation: str = Field(description="Short explanation of the recommendation and financial facts. MUST BE IN ENGLISH.")

class BatchFinancialDecisions(BaseModel):
    scratchpad: str = Field(description="MANDATORY: 90-day daily balance simulation for ALL requests. Extract missing event amounts from attached images. Rank plans strictly using the 6 tie-breakers.")
    decisions: List[FinancialDecision] = Field(description="Exactly one decision for every request provided in the batch.")

class TokenTracker:
    def __init__(self):
        self.metrics = {"gemini-3.6-flash": {"prompt": 0, "completion": 0, "calls": 0, "cost_in": 0.0, "cost_out": 0.0}}
        self.lock = asyncio.Lock()
        
    async def add(self, model, prompt, completion):
        async with self.lock:
            self.add_sync(model, prompt, completion)

    def add_sync(self, model, prompt, completion):
        if model not in self.metrics:
            self.metrics[model] = {"prompt": 0, "completion": 0, "calls": 0, "cost_in": 0.0, "cost_out": 0.0}
        self.metrics[model]["prompt"] += int(prompt or 0)
        self.metrics[model]["completion"] += int(completion or 0)
        self.metrics[model]["calls"] += 1

    def generate_report(self, num_requests, output_path):
        total_in, total_out, total_calls = 0, 0, 0
        report = f"# AI Token Usage and Cost Analysis\n* **Total Requests Evaluated:** {num_requests}\n\n"
        for m, data in self.metrics.items():
            if data['calls'] > 0:
                total_in += data['prompt']; total_out += data['completion']; total_calls += data['calls']
                report += f"### Model: {m}\n- **Calls:** {data['calls']} | **In Tokens:** {data['prompt']:,} | **Out Tokens:** {data['completion']:,} | **Cost:** $0.0000 (Free Tier)\n\n"
        report += f"### Overall Totals\n- **Total Calls:** {total_calls}\n- **Total Tokens:** {total_in + total_out:,}\n- **Avg Tokens/Req:** {(total_in + total_out)/max(1, num_requests):,.0f}\n"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f: f.write(report)

class FinancialAgent:
    def __init__(self, tracker: TokenTracker, api_key: str, agent_id: int):
        self.client = genai.Client(api_key=api_key)
        self.tracker = tracker
        self.agent_id = agent_id

    async def async_evaluate_batch(self, batch_contexts: list) -> dict:
        model = "gemini-3.6-flash"
        
        system_instruction = """You are a strict, objective AI financial agent evaluating a BATCH of requests.
CRITICAL RULES FOR EACH REQUEST:
1. 90-DAY CHECK: Balance must NEVER fall below minimum_balance_to_keep.
2. MISSING AMOUNTS: If an event is missing an amount, inspect the attached images to deduce it. DO NOT treat missing amounts as zero.
3. CURRENCY: Convert foreign currencies using provided fixed rates.
4. TIE-BREAKERS: 1) Complete by deadline 2) No spending changes 3) Min amount paid 4) Start earlier 5) Fewer payments 6) Lowest payment_option_id."""

        payload = [system_instruction]
        
        for ctx in batch_contexts:
            images = ctx.pop("Attached_Images", [])
            for img_data in images:
                try:
                    payload.append(f"\nImage for Event {img_data['event_id']}:")
                    payload.append(PIL.Image.open(img_data["image_path"]))
                except Exception as e:
                    logging.error(f"Failed to load image: {e}")

        payload.append(f"\nBATCH DATA:\n{json.dumps(batch_contexts, indent=2, default=str)}")

        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=payload,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchFinancialDecisions,
                    temperature=0.0
                )
            )
            
            if getattr(response, "usage_metadata", None):
                await self.tracker.add(model, getattr(response.usage_metadata, "prompt_token_count", 0), getattr(response.usage_metadata, "candidates_token_count", 0))
            
            return json.loads(response.text)
        except Exception as e:
            raise e
