import os
import json
import base64
import logging
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. Strict Schema Enforcement matching the problem statement
class FinancialDecision(BaseModel):
    # Hidden Chain-of-Thought field to force the LLM to do the math first
    cash_flow_reasoning: str = Field(description="Step-by-step month-by-month cash flow calculations. Deduct recurring/pending expenses from income. Ensure the minimum preferred balance is NEVER breached.")
    
    amount_safe_to_pay: float = Field(description="The maximum amount the user can safely pay today.")
    affordability_status: str = Field(description="Whether the request is affordable now, with a plan, later, or not at all.")
    recommended_payment_method: str = Field(description="Pay in full, pay partially, use installments, wait, or not proceed.")
    payment_plan: str = Field(description="The dates and amounts of recommended payments, or 'N/A'.")
    earliest_date_for_full_payment: str = Field(description="The earliest safe date for paying the full amount (YYYY-MM-DD), or 'N/A'.")
    spending_changes_needed: str = Field(description="Flexible expenses that must be stopped or reduced, or 'None'.")
    decision_explanation: str = Field(description="A short explanation supporting the recommendation.")

class FinancialAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def encode_image(self, image_path: str) -> str:
        """Encodes local media files to Base64 for the Vision model."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def evaluate_request(self, row_data: dict, dataset_dir: str) -> dict:
        system_prompt = """You are a strict, safety-first AI financial agent.
Your job is to decide whether a user can safely afford a requested expense.

CRITICAL RULES:
1. A recommendation is ONLY SAFE if the user can complete the payment plan, cover all essential/recurring expenses, and maintain their preferred minimum balance throughout the forecast period.
2. Consider confirmed income and available payment options.
3. Personalize the recommendation: Factor in their financial history and willingness to adjust flexible expenses.
4. Calculate the cash flow strictly in `cash_flow_reasoning` before making your final recommendations."""

        # Build text payload
        content = [{"type": "text", "text": f"User Financial Request:\n{json.dumps(row_data, indent=2)}"}]
        
        # Build image payload dynamically (Look for common image column names)
        image_keys = ['image_path', 'image', 'receipt', 'media']
        for key in image_keys:
            if key in row_data and pd.notna(row_data[key]) and str(row_data[key]).strip():
                img_path = os.path.join(dataset_dir, str(row_data[key]).strip())
                if os.path.exists(img_path):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img_path)}"}
                    })
                break

        try:
            # .parse() guarantees the output matches our Pydantic schema perfectly
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                response_format=FinancialDecision,
                temperature=0.0 # Strict determinism for financial decisions
            )
            return response.choices[0].message.parsed.model_dump()
            
        except Exception as e:
            logging.error(f"Agent Error: {e}")
            # Failsafe escalation: If the API times out or hits a safety filter, default to denying the expense safely
            return {
                "cash_flow_reasoning": "Error occurred.",
                "amount_safe_to_pay": 0.0,
                "affordability_status": "not at all",
                "recommended_payment_method": "not proceed",
                "payment_plan": "N/A",
                "earliest_date_for_full_payment": "N/A",
                "spending_changes_needed": "None",
                "decision_explanation": "System error or safety violation detected. Defaulting to safe decline."
            }
