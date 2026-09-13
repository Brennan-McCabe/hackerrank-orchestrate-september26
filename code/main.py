import os
import pandas as pd
import logging
import asyncio
import random
import time
from dotenv import load_dotenv
from agent import FinancialAgent, TokenTracker
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

csv_lock = asyncio.Lock()

# 1. PROACTIVE PACING: Enforce a strict delay BEFORE hitting the API
class GlobalRateLimiter:
    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_call = time.monotonic()

# Cap the entire system at 2 Requests Per Minute (TPM) to avoid hitting the API's rate limits. This is a global limit across all workers.
rate_limiter = GlobalRateLimiter(calls_per_minute=2)
# Cap the number of simultaneous network connections at 5
api_semaphore = asyncio.Semaphore(5)

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def worker(agent, queue, cache_file):
    base_delay = 10.0
    max_retries = 5

    while True:
        try:
            item = await queue.get()
        except asyncio.CancelledError:
            break

        if item is None:
            queue.task_done()
            break

        batch_rows, batch_contexts, model, attempt = item
        batch_ids = [r['request_id'] for r in batch_rows]
        
        final_decisions = []
        success = False
        
        try:
            # 2. WAIT FOR THE GREEN LIGHT: Prevents the bottleneck before it happens
            await rate_limiter.wait()
            
            async with api_semaphore:
                logging.info(f"🚀 [Agent {agent.agent_id}] Evaluating Batch of {len(batch_rows)} ({model}) (Attempt {attempt+1}): {batch_ids[0]} to {batch_ids[-1]}...")
                
                response_data = await asyncio.wait_for(
                    agent.async_evaluate_batch(batch_contexts, model=model),
                    timeout=180.0
                )
            
            decisions_list = response_data.get('decisions', [])
            decision_map = {d.get("request_id"): d for d in decisions_list if isinstance(d, dict) and "request_id" in d}
            
            for r_id in batch_ids:
                if r_id in decision_map:
                    decision = decision_map[r_id]
                    if str(decision.get("earliest_date_for_full_payment")).lower() in ["none", "n/a", "null"]:
                        decision["earliest_date_for_full_payment"] = ""
                    final_decisions.append(decision)
                else:
                    logging.warning(f"⚠️ [Agent {agent.agent_id}] AI omitted request {r_id}. Applying fallback.")
                    final_decisions.append({
                        "request_id": r_id, "amount_safe_to_pay": 0.0, "affordability_status": "not_affordable",
                        "recommended_payment_method": "not_recommended", "payment_plan": "none",
                        "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
                        "decision_explanation": "Model omission fallback."
                    })
            success = True
            
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(err in error_str for err in ["429", "resource_exhausted", "503", "unavailable", "quota"])
            is_timeout = isinstance(e, asyncio.TimeoutError)
            
            if is_rate_limit:
                wait_time = (base_delay * (1.5 ** attempt)) * random.uniform(0.8, 1.2)
                logging.warning(f"⏳ [Agent {agent.agent_id}] API Busy. Re-queueing and cooling down for {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                await queue.put((batch_rows, batch_contexts, model, attempt + 1))
                queue.task_done()
                continue
            else:
                logging.warning(f"⚠️ [Agent {agent.agent_id}] Batch Failed ({'Timeout' if is_timeout else 'Error: ' + str(e)[:100]}).")
                
                if attempt >= 1 and len(batch_rows) > 1:
                    logging.info(f"✂️ [Agent {agent.agent_id}] Splitting batch of {len(batch_rows)} to recover.")
                    mid = len(batch_rows) // 2
                    await queue.put((batch_rows[:mid], batch_contexts[:mid], model, attempt + 1))
                    await queue.put((batch_rows[mid:], batch_contexts[mid:], model, attempt + 1))
                elif attempt < max_retries:
                    await queue.put((batch_rows, batch_contexts, model, attempt + 1))
                else:
                    logging.error(f"💀 [Agent {agent.agent_id}] Exhausted retries for {batch_ids[0]}. Applying fallback.")
                    for r_id in batch_ids:
                        final_decisions.append({
                            "request_id": r_id, "amount_safe_to_pay": 0.0, "affordability_status": "not_affordable",
                            "recommended_payment_method": "not_recommended", "payment_plan": "none",
                            "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
                            "decision_explanation": "Error fallback."
                        })
                    success = True

        if success and final_decisions:
            good_decisions = [d for d in final_decisions if "fallback" not in str(d.get("decision_explanation", "")).lower()]
            if good_decisions:
                async with csv_lock:
                    pd.DataFrame(good_decisions).to_csv(cache_file, mode='a', header=not os.path.exists(cache_file), index=False)
            
            logging.info(f"✅ [Agent {agent.agent_id}] Processed batch starting at {batch_ids[0]}.")
            
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

    api_keys = [v.strip() for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY") and v.strip()]
            
    if not api_keys:
        logging.error("FATAL: No GEMINI_API_KEY found in .env.")
        return
        
    # 3. REDUCE PARALLELISM: We do not need 25 workers. 5 is plenty and prevents quota burn.
    num_workers = min(len(api_keys), 5)
    logging.info(f"🚀 Spawning {num_workers} parallel workers to protect quota and rate limits.")
    
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
        # Batch size changed to 15 due to TPM limits
        batches = list(chunk_list(pending_rows_list, 15))
        for batch in batches:
            batch_contexts = [loader.get_unified_context(r) for r in batch]
            queue.put_nowait((batch, batch_contexts, "gemini-3.8-flash", 0))

        worker_tasks = []
        for i in range(num_workers):
            key = api_keys[i]
            agent = FinancialAgent(tracker, api_key=key, agent_id=i+1)
            task = asyncio.create_task(worker(agent, queue, output_path))
            worker_tasks.append(task)
            
        await queue.join()
        
        for _ in worker_tasks:
            queue.put_nowait(None)
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    if os.path.exists(output_path):
        final_df = pd.read_csv(output_path)
        final_df.drop_duplicates(subset=['request_id'], keep='last', inplace=True)
        
        missing_ids = set(loader.requests['request_id']) - set(final_df['request_id'])
        if missing_ids:
            logging.warning(f"Adding fallbacks for {len(missing_ids)} missing requests.")
            missing_rows = [{
                "request_id": m_id, "amount_safe_to_pay": 0.0, "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended", "payment_plan": "none",
                "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
                "decision_explanation": "Final safety fallback."
            } for m_id in missing_ids]
            final_df = pd.concat([final_df, pd.DataFrame(missing_rows)], ignore_index=True)

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
