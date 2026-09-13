import os
import pandas as pd
import math

class DataLoader:
    def __init__(self, dataset_dir):
        self.dir = dataset_dir
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

        if not self.requests.empty and 'allows_partial_payment' in self.requests.columns:
            if self.requests['allows_partial_payment'].dtype == object:
                self.requests['allows_partial_payment'] = self.requests['allows_partial_payment'].astype(str).str.lower().map({'true': True, 'false': False})

    def _clean_dict(self, d):
        if isinstance(d, dict):
            return {k: self._clean_dict(v) for k, v in d.items() if v is not None and not (isinstance(v, float) and math.isnan(v)) and v != "" and v != []}
        elif isinstance(d, list):
            return [self._clean_dict(v) for v in d if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return d

    def get_unified_context(self, row) -> dict:
        r_id, u_id = row['request_id'], row['user_id']
        prof = self.profiles[self.profiles['user_id'] == u_id].iloc[0].to_dict() if not self.profiles.empty and u_id in self.profiles['user_id'].values else {}
        events = self.events[self.events['user_id'] == u_id].copy() if not self.events.empty else pd.DataFrame()

        attached_images = []

        if not events.empty:
            for idx, ev in events.iterrows():
                if pd.isna(ev.get('amount')) or str(ev.get('amount')).strip() == "":
                    img_row = self.images[self.images['related_event_id'] == ev['event_id']]
                    if not img_row.empty:
                        path = os.path.join(self.dir, "media", "images", f"{img_row.iloc[0]['image_id']}.png")
                        if os.path.exists(path): 
                            try:
                                attached_images.append({"event_id": ev['event_id'], "image_path": path})
                                events.at[idx, 'amount'] = f"IMAGE_ATTACHED"
                            except Exception:
                                pass
            
            # DROP TOKEN-WASTING COLUMNS
            cols_to_drop = ['user_id', 'created_at', 'updated_at', 'description']
            events = events.drop(columns=[c for c in cols_to_drop if c in events.columns], errors='ignore')
        
        target_cur = prof.get('home_currency', 'USD')
        rel_rates = self.rates[self.rates['to_currency'] == target_cur] if not self.rates.empty else pd.DataFrame()
        
        opts = self.options[self.options['request_id'] == r_id] if not self.options.empty else pd.DataFrame()
        if not opts.empty:
            opts = opts.drop(columns=[c for c in ['request_id', 'provider_name'] if c in opts.columns], errors='ignore')
        
        e_ids = events['event_id'].tolist() if not events.empty else []
        msgs = self.messages[(self.messages['user_id'] == u_id) | (self.messages['request_id'] == r_id) | (self.messages['related_event_id'].isin(e_ids))] if not self.messages.empty else pd.DataFrame()
        if not msgs.empty:
            msgs = msgs.drop(columns=[c for c in ['user_id', 'message_id', 'timestamp'] if c in msgs.columns], errors='ignore')

        raw_context = {
            "Request": row.dropna().to_dict(), 
            "Profile": prof, 
            "Events": events.dropna(axis=1, how='all').to_dict('records') if not events.empty else [],
            "Attached_Images": attached_images,
            "Options": opts.dropna(axis=1, how='all').to_dict('records') if not opts.empty else [], 
            "Messages": msgs.dropna(axis=1, how='all').to_dict('records') if not msgs.empty else [],
            "Rates": rel_rates.dropna(axis=1, how='all').to_dict('records') if not rel_rates.empty else []
        }
        return self._clean_dict(raw_context)
