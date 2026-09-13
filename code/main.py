import os
import pandas as pd
import logging
import asyncio
from dotenv import load_dotenv
from agent import FinancialAgent, TokenTracker
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

csv_lock = asyncio.Lock()

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def worker(agent, queue, cache_file):
    max_retries = 8
    base_delay = 15.0

    while True:
        try:
            batch_rows, batch_contexts = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        batch_ids = [r['request_id'] for r in batch_rows]
        logging.info(f"🚀 [Agent {agent.agent_id}] Evaluating Batch of {len(batch_rows)}: {batch_ids[0]} to {batch_ids[-1]}...")
        
        final_decisions = []
        
        for attempt in range(max_retries):
            try:
                response_data = await agent.async_evaluate_batch(batch_contexts)
                decisions_list = response_data.get('decisions', [])
                decision_map = {d.get("request_id"): d for d in decisions_list if "request_id" in d}
                
                for r_id in batch_ids:
                    if r_id in decision_map:
                        decision = decision_map[r_id]
                        if decision.get("earliest_date_for_full_payment") in ["none", "None", "N/A", "null"]:
                            decision["earliest_date_for_full_payment"] = ""
                        final_decisions.append(decision)
                    else:
                        logging.warning(f"[Agent {agent.agent_id}] AI omitted request {r_id}. Applying fallback.")
                        final_decisions.append({
                            "request_id": r_id, "amount_safe_to_pay": 0.0, "affordability_status": "not_affordable",
                            "recommended_payment_method": "not_recommended", "payment_plan": "none",
                            "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
                            "decision_explanation": "Model omission fallback."
                        })
                break 
                
            except Exception as e:
                error_str = str(e)
                if any(err in error_str for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                    wait_time = base_delay * (1.5 ** attempt)
                    logging.warning(f"[Agent {agent.agent_id}] API Busy. Cooling down for {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"[Agent {agent.agent_id}] Batch Failed unrecoverably: {e}")
                    for r_id in batch_ids:
                        final_decisions.append({
                            "request_id": r_id, "amount_safe_to_pay": 0.0, "affordability_status": "not_affordable",
                            "recommended_payment_method": "not_recommended", "payment_plan": "none",
                            "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
                            "decision_explanation": "Error fallback."
                        })
                    break
        else:
            logging.error(f"[Agent {agent.agent_id}] Exhausted retries for batch {batch_ids[0]}.")

        if final_decisions:
            good_decisions = [d for d in final_decisions if "fallback" not in str(d.get("decision_explanation", "")).lower()]
            if good_decisions:
                async with csv_lock:
                    pd.DataFrame(good_decisions).to_csv(cache_file, mode='a', header=not os.path.exists(cache_file), index=False)
        
        logging.info(f"✅ [Agent {agent.agent_id}] Batch processed. Resting for 20 seconds before next batch...")
        await asyncio.sleep(20.0)
        queue.task_done()

async def main():
    load_dotenv()
    
    code_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(code_dir, ".."))
    
    dataset_dir = os.path.join(repo_root, "dataset")
    output_path = os.path.join(repo_root, "output.csv")
    report_path = os.path.join(code_dir, "evaluation", "usage_report.md")
    
    if not os.path.exists(dataset_dir):
        logging.error(f"FATAL: Dataset not found at {dataset_dir}")
        return

    tracker = TokenTracker()
    loader = DataLoader(dataset_dir) 
    
    if loader.requests.empty:
        logging.error("FATAL: requests.csv loaded empty data.")
        return

    # Gather API keys from .env
    api_keys = []
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY") and v.strip():
            api_keys.append(v.strip())
            
    if not api_keys:
        logging.error("FATAL: No GEMINI_API_KEY found in .env.")
        return
        
    logging.info(f"🚀 Spawning {len(api_keys)} parallel workers based on available API keys.")
    
    # Process caching BEFORE queueing
    cached_ids = set()
    if os.path.exists(output_path):
        try:
            cached_df = pd.read_csv(output_path)
            for _, r in cached_df.iterrows():
                if r.get('amount_safe_to_pay') == 0.0 and "fallback" in str(r.get('decision_explanation', '')).lower():
                    continue
                cached_ids.add(r['request_id'])
        except pd.errors.EmptyDataError:
            pass

    pending_rows_list = [row for _, row in loader.requests.iterrows() if row['request_id'] not in cached_ids]
    
    if not pending_rows_list:
        logging.info("All requests processed!")
    else:
        queue = asyncio.Queue()
        batches = list(chunk_list(pending_rows_list, 10))
        for batch in batches:
            batch_contexts = [loader.get_unified_context(r) for r in batch]
            queue.put_nowait((batch, batch_contexts))

        # Spawn workers
        worker_tasks = []
        for i, key in enumerate(api_keys):
            agent = FinancialAgent(tracker, api_key=key, agent_id=i+1)
            task = asyncio.create_task(worker(agent, queue, output_path))
            worker_tasks.append(task)
            
        await asyncio.gather(*worker_tasks)

    # Compile final results
    if os.path.exists(output_path):
        final_df = pd.read_csv(output_path)
        final_df.drop_duplicates(subset=['request_id'], keep='last', inplace=True)
        
        order_map = {req_id: idx for idx, req_id in enumerate(loader.requests['request_id'])}
        final_df['sort_idx'] = final_df['request_id'].map(order_map)
        final_df = final_df.sort_values('sort_idx').drop('sort_idx', axis=1)
        
        out_cols = [
            "request_id", "amount_safe_to_pay", "affordability_status", 
            "recommended_payment_method", "payment_plan", 
            "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"
        ]
        final_df[out_cols].to_csv(output_path, index=False)
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        tracker.generate_report(len(final_df), report_path)
        logging.info(f"✅ Pipeline Complete! Created {output_path} and {report_path}.")

if __name__ == "__main__":
    asyncio.run(main())
