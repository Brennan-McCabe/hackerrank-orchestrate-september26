import os
import json
import logging
import asyncio
import copy
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
        self.metrics = {}
        self.lock = asyncio.Lock()
        
    async def add(self, model, prompt, completion):
        async with self.lock:
            if model not in self.metrics:
                self.metrics[model] = {"prompt": 0, "completion": 0, "calls": 0}
            self.metrics[model]["prompt"] += int(prompt or 0)
            self.metrics[model]["completion"] += int(completion or 0)
            self.metrics[model]["calls"] += 1

    def generate_report(self, num_requests, output_path):
        total_in = 0
        total_out = 0
        total_calls = 0
        total_cost = 0.0
        
        # Base pricing per 1M tokens 
        pricing = {
            "gemini-3.6-flash": {"in": 0.075 / 1_000_000, "out": 0.30 / 1_000_000},
            "gemini-3.6-pro": {"in": 1.25 / 1_000_000, "out": 5.00 / 1_000_000}
        }
        
        report = f"# AI Token Usage and Cost Analysis\n* **Total Requests Evaluated:** {num_requests}\n\n"
        
        for m, data in self.metrics.items():
            if data['calls'] > 0:
                total_in += data['prompt']
                total_out += data['completion']
                total_calls += data['calls']
                
                rate_in = pricing.get(m, pricing["gemini-3.6-flash"])["in"]
                rate_out = pricing.get(m, pricing["gemini-3.6-flash"])["out"]
                
                cost_in = data['prompt'] * rate_in
                cost_out = data['completion'] * rate_out
                m_cost = cost_in + cost_out
                total_cost += m_cost
                
                report += f"### Model: {m}\n"
                report += f"- **Calls:** {data['calls']}\n"
                report += f"- **In Tokens:** {data['prompt']:,} (${cost_in:.4f})\n"
                report += f"- **Out Tokens:** {data['completion']:,} (${cost_out:.4f})\n"
                report += f"- **Estimated Cost:** ${m_cost:.4f}\n\n"
                
        avg_tokens = (total_in + total_out) / max(1, num_requests)
        avg_cost = total_cost / max(1, num_requests)
        
        report += f"### Overall Totals\n"
        report += f"- **Total Calls:** {total_calls}\n"
        report += f"- **Total Tokens:** {total_in + total_out:,}\n"
        report += f"- **Avg Tokens/Req:** {avg_tokens:,.0f}\n"
        report += f"- **Total Cost:** ${total_cost:.4f}\n"
        report += f"- **Avg Cost/Req:** ${avg_cost:.4f}\n"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(report)

class FinancialAgent:
    def __init__(self, tracker: TokenTracker, api_key: str, agent_id: int):
        self.client = genai.Client(api_key=api_key)
        self.tracker = tracker
        self.agent_id = agent_id

    async def async_evaluate_batch(self, batch_contexts: list, model: str = "gemini-3.6-flash") -> dict:
        system_instruction = """You are a strict, objective AI financial agent evaluating a BATCH of requests.
CRITICAL RULES FOR EACH REQUEST:
1. 90-DAY CHECK: Balance must NEVER fall below minimum_balance_to_keep.
2. MISSING AMOUNTS: If an event is missing an amount, inspect the attached images to deduce it. DO NOT treat missing amounts as zero.
3. CURRENCY: Convert foreign currencies using provided fixed rates.
4. TIE-BREAKERS: 1) Complete by deadline 2) No spending changes 3) Min amount paid 4) Start earlier 5) Fewer payments 6) Lowest payment_option_id."""

        payload = [system_instruction]
        
        # Prevent mutating the context dicts when requeuing and splitting
        contexts_copy = copy.deepcopy(batch_contexts)
        
        for ctx in contexts_copy:
            images = ctx.pop("Attached_Images", [])
            for img_data in images:
                try:
                    payload.append(f"\nImage for Event {img_data['event_id']}:")
                    payload.append(PIL.Image.open(img_data["image_path"]))
                except Exception as e:
                    logging.error(f"Failed to load image: {e}")

        payload.append(f"\nBATCH DATA:\n{json.dumps(contexts_copy, indent=2, default=str)}")

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
