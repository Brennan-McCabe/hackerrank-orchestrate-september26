import os
import pandas as pd

class DataLoader:
    def __init__(self, dataset_dir, agent):
        self.dir = dataset_dir
        self.agent = agent
        def load(name): 
            path = os.path.join(dataset_dir, name)
            return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
            
        self.requests = load("requests.csv")
        self.profiles = load("financial_profiles.csv")
        self.events = load("financial_events.csv")
        self.rates = load("exchange_rates.csv")
        self.options = load("request_payment_options.csv")
        self.messages = load("messages.csv")
        self.images = load("images.csv")

        # FIX: Ensure allows_partial_payment is a strict boolean, not the string "false"/"true"
        if not self.requests.empty and 'allows_partial_payment' in self.requests.columns:
            if self.requests['allows_partial_payment'].dtype == object:
                self.requests['allows_partial_payment'] = self.requests['allows_partial_payment'].astype(str).str.lower().map({'true': True, 'false': False})

    def get_unified_context(self, row) -> dict:
        r_id, u_id = row['request_id'], row['user_id']
        prof = self.profiles[self.profiles['user_id'] == u_id].iloc[0].to_dict() if not self.profiles.empty and u_id in self.profiles['user_id'].values else {}
        events = self.events[self.events['user_id'] == u_id].copy() if not self.events.empty else pd.DataFrame()

        # Handle Missing Amounts via Vision OCR
        if not events.empty:
            for idx, ev in events.iterrows():
                if pd.isna(ev.get('amount')) or str(ev.get('amount')).strip() == "":
                    img_row = self.images[self.images['related_event_id'] == ev['event_id']]
                    if not img_row.empty:
                        path = os.path.join(self.dir, "media", "images", f"{img_row.iloc[0]['image_id']}.png")
                        if os.path.exists(path): 
                            events.at[idx, 'amount'] = self.agent.extract_missing_amount(path)
        
        target_cur = prof.get('home_currency', 'USD')
        rel_rates = self.rates[self.rates['target_currency'] == target_cur] if not self.rates.empty else pd.DataFrame()
        opts = self.options[self.options['request_id'] == r_id] if not self.options.empty else pd.DataFrame()
        
        e_ids = events['event_id'].tolist() if not events.empty else []
        msgs = self.messages[(self.messages['user_id'] == u_id) | (self.messages['request_id'] == r_id) | (self.messages['related_event_id'].isin(e_ids))] if not self.messages.empty else pd.DataFrame()

        return {
            "Request": row.dropna().to_dict(), 
            "Profile": prof, 
            "Events": events.dropna(axis=1, how='all').to_dict('records') if not events.empty else [],
            "Options": opts.dropna(axis=1, how='all').to_dict('records') if not opts.empty else [], 
            "Messages": msgs.dropna(axis=1, how='all').to_dict('records') if not msgs.empty else [],
            "Rates": rel_rates.dropna(axis=1, how='all').to_dict('records') if not rel_rates.empty else []
        }
