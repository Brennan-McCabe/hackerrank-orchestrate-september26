import os
import pandas as pd
import logging
import asyncio
from dotenv import load_dotenv
from agent import FinancialAgent, TokenTracker
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def process_single_request(row, loader, agent, semaphore, cache_file):
    req_id = row['request_id']
    
    # 1. Skip if already processed in cache
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file)
            if req_id in cached_df['request_id'].values:
                logging.info(f"Skipping {req_id} (Found in cache)")
                return cached_df[cached_df['request_id'] == req_id].iloc[0].to_dict()
        except pd.errors.EmptyDataError:
            pass

    async with semaphore:
        logging.info(f"Evaluating {req_id}...")
        context = loader.get_unified_context(row)
        decision = await agent.async_evaluate_request(context)
        
        # Strip scratchpad for final output schema
        decision.pop("scratchpad", None)

        if decision.get("earliest_date_for_full_payment") in ["none", "None", "N/A", "null"]:
            decision["earliest_date_for_full_payment"] = ""
            
        final_row = {"request_id": req_id}
        final_row.update(decision)
        
        # Auto-save to cache incrementally
        pd.DataFrame([final_row]).to_csv(cache_file, mode='a', header=not os.path.exists(cache_file), index=False)
        return final_row

async def main():
    load_dotenv()
    dataset_dir = os.path.join(os.path.dirname(__file__), "../dataset")
    output_path = os.path.join(os.path.dirname(__file__), "../output.csv")
    report_path = os.path.join(os.path.dirname(__file__), "evaluation/usage_report.md")
    
    tracker = TokenTracker()
    agent = FinancialAgent(tracker)
    loader = DataLoader(dataset_dir, agent)
    
    # Limit concurrency to avoid API rate limits (HTTP 429)
    semaphore = asyncio.Semaphore(10)
    
    tasks = [process_single_request(row, loader, agent, semaphore, output_path) for _, row in loader.requests.iterrows()]
    results = await asyncio.gather(*tasks)

    # Sort results to perfectly match original requests.csv order
    order_map = {req_id: idx for idx, req_id in enumerate(loader.requests['request_id'])}
    results.sort(key=lambda x: order_map.get(x['request_id'], 999999))

    # Final rewrite to ensure strict column ordering
    out_cols = [
        "request_id", "amount_safe_to_pay", "affordability_status", 
        "recommended_payment_method", "payment_plan", 
        "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"
    ]
    pd.DataFrame(results)[out_cols].to_csv(output_path, index=False)
    
    tracker.generate_report(len(results), report_path)
    logging.info(f"✅ Pipeline Complete! Created {output_path} and {report_path}.")

if __name__ == "__main__":
    asyncio.run(main())
