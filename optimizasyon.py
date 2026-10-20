import pandas as pd
import numpy as np
import networkx as nx
import joblib
from datetime import datetime, timedelta
import os

# --- KARBON VE ELEKTRİKLİ POMPA PARAMETRELERİ ---
CO2_SAVED_PER_TON_BASE = 1.20   
CO2_EMISSION_PER_TON_KM = 0.025 

SECTOR_CARBON_MULTIPLIER = {
    'Kimya': 1.6,
    'Metal': 1.3,
    'Tekstil': 1.1,
    'Gıda': 0.9
}

def prepare_forecaster_features(df_history, target_time, expected_features, nodes_df):
    target_dt = pd.to_datetime(target_time)
    is_weekend = 1 if target_dt.weekday() >= 5 else 0
    
    features_list = []
    
    for fid in nodes_df['Fabrika_ID'].unique():
        f_data = df_history[df_history['Fabrika_ID'] == fid].sort_values('Tarih')
        last_3 = f_data.tail(3)
        
        t1 = last_3.iloc[-1] if len(last_3) >= 1 else pd.Series(dtype=float)
        t2 = last_3.iloc[-2] if len(last_3) >= 2 else pd.Series(dtype=float)
        t3 = last_3.iloc[-3] if len(last_3) >= 3 else pd.Series(dtype=float)
        
        sektor = nodes_df[nodes_df['Fabrika_ID'] == fid]['Sektor'].iloc[0]
        
        row = {
            'Fabrika_ID': fid,
            'Is_Weekend': is_weekend,
            'Lag_1_Talep': t1.get('Temiz_Su_Talebi_m3', 0),
            'Lag_1_Desarj': t1.get('Atik_Su_Desarji_m3', 0),
            'Lag_2_Talep': t2.get('Temiz_Su_Talebi_m3', 0),
            'Lag_3_Talep': t3.get('Temiz_Su_Talebi_m3', 0),
            'Veren_Sektor': sektor, 
            'Last_pH': t1.get('pH', 7.0),
            'Last_Sicaklik_C': t1.get('Sicaklik_C', 25.0),
            'Last_KOI_mgL': t1.get('KOI_mgL', 100.0)
        }
        features_list.append(row)
        
    df_features = pd.DataFrame(features_list)
    df_encoded = pd.get_dummies(df_features, columns=['Veren_Sektor'])
    
    for col in df_encoded.columns:
        if col.startswith('Veren_Sektor_'):
            df_encoded.rename(columns={col: col.replace('Veren_Sektor_', 'Sektor_')}, inplace=True)
            
    df_aligned = df_encoded.reindex(columns=['Fabrika_ID'] + list(expected_features), fill_value=0)
    return df_aligned, df_features

def run_daily_simulation(hedef_gun_str, forecaster_data, scorer_data):
    print(f"[{hedef_gun_str}] Günlük Optimizasyon Motoru Tetiklendi...")
    
    nodes_df = pd.read_csv('gercek_nodes_factories.csv')
    edges_df = pd.read_csv('gercek_edges_network.csv')
    su_verisi = pd.read_csv('gunluk_su_verisi_2027.csv')
    su_verisi['Tarih'] = pd.to_datetime(su_verisi['Tarih'])
    
    hedef_gun = pd.to_datetime(hedef_gun_str)
    
    gecmis_veri = su_verisi[su_verisi['Tarih'] < hedef_gun]
    
    forecaster_model = forecaster_data['model']
    forecaster_features = forecaster_data['features']
    
    df_model_input, df_raw_features = prepare_forecaster_features(gecmis_veri, hedef_gun, forecaster_features, nodes_df)
    
    X_pred = df_model_input.drop(columns=['Fabrika_ID'])
    predictions = forecaster_model.predict(X_pred) 
    
    durum_dict = {}
    for i, row in df_raw_features.iterrows():
        fid = row['Fabrika_ID']
        node_info = nodes_df[nodes_df['Fabrika_ID'] == fid].iloc[0]
        
        pred_talep = max(0, predictions[i][0])
        pred_desarj = max(0, predictions[i][1])
        
        durum_dict[fid] = {
            'Gercek_Isim': node_info['Gercek_Isim'],
            'Sektor': node_info['Sektor'],
            'Enlem': node_info['Enlem'],
            'Boylam': node_info['Boylam'],
            'Temiz_Su_Talebi_m3': pred_talep,
            'Atik_Su_Desarji_m3': pred_desarj,
            'pH': row['Last_pH'],
            'Sicaklik_C': row['Last_Sicaklik_C'],
            'KOI_mgL': row['Last_KOI_mgL']
        }
        
    scorer_model = scorer_data['model']
    scorer_features = scorer_data['features']
    
    req_list = []
    for _, edge in edges_df.iterrows():
        kaynak = edge['Kaynak_Fabrika']
        hedef = edge['Hedef_Fabrika']
        if kaynak in durum_dict and hedef in durum_dict:
            k_data = durum_dict[kaynak]
            h_data = durum_dict[hedef]
            req_list.append({
                'Kaynak': kaynak,
                'Hedef': hedef,
                'Mesafe_km': edge['Mesafe_km'],
                'Veren_Sektor': k_data['Sektor'],
                'Alan_Sektor': h_data['Sektor'],
                'Veren_pH': k_data['pH'],
                'Alan_Hedef_pH': h_data['pH'],
                'Sicaklik_Farki': k_data['Sicaklik_C'] - h_data['Sicaklik_C'],
                'KOI_Farki': k_data['KOI_mgL'] - h_data['KOI_mgL']
            })

    df_req = pd.DataFrame(req_list)
    df_encoded = pd.get_dummies(df_req, columns=['Veren_Sektor', 'Alan_Sektor'])
    df_aligned = df_encoded.reindex(columns=scorer_features, fill_value=0)
    df_req['uyum_skoru'] = scorer_model.predict(df_aligned)
    
    G = nx.DiGraph()
    G.add_node('S')
    G.add_node('T')

    for fid, data in durum_dict.items():
        G.add_edge('S', fid, capacity=int(data['Atik_Su_Desarji_m3']), weight=0)
        G.add_edge(fid, 'T', capacity=int(data['Temiz_Su_Talebi_m3']), weight=0)

    for _, row in df_req.iterrows():
        kaynak = row['Kaynak']
        hedef = row['Hedef']
        mesafe = row['Mesafe_km']
        uyum_skoru = row['uyum_skoru']
        k_data = durum_dict[kaynak]
        
        sektor_carpan = SECTOR_CARBON_MULTIPLIER.get(k_data['Sektor'], 1.0)
        birim_co2_kazanc = CO2_SAVED_PER_TON_BASE * sektor_carpan
        birim_co2_maliyet = mesafe * CO2_EMISSION_PER_TON_KM
        net_co2_kazanc_birim = birim_co2_kazanc - birim_co2_maliyet
        
        if net_co2_kazanc_birim <= 0:
            continue
            
        tasima_maliyeti = mesafe * 2.0
        mikro_aritma_maliyeti = 65 * (1 - (uyum_skoru / 100))**2 * 4
        toplam_maliyet = tasima_maliyeti + mikro_aritma_maliyeti
        
        if toplam_maliyet < 65.0:
            G.add_edge(kaynak, hedef, capacity=99999, weight=int(toplam_maliyet), net_co2_birim=net_co2_kazanc_birim)

    try:
        flow_dict = nx.max_flow_min_cost(G, 'S', 'T')
    except nx.NetworkXUnfeasible:
        print("Optimizasyon çözülemedi (Unfeasible).")
        return []

    acilan_vanalar = []
    
    for kaynak, hedefler in flow_dict.items():
        if kaynak in ['S', 'T']: continue
        for hedef, miktar in hedefler.items():
            # YENİ: 50 Ton'dan az günlük su için devasa vanayı açmaya değmez (Fizibilite kısıtı)
            if hedef in ['S', 'T'] or miktar <= 50.0: continue
            
            edge_data = G.get_edge_data(kaynak, hedef)
            maliyet = edge_data['weight']
            co2_birim = edge_data.get('net_co2_birim', 0)
            
            acilan_vanalar.append({
                'Veren_Fabrika': kaynak,
                'Alan_Fabrika': hedef,
                'Veren_Isim': durum_dict[kaynak]['Gercek_Isim'],
                'Alan_Isim': durum_dict[hedef]['Gercek_Isim'],
                'Veren_Enlem': durum_dict[kaynak]['Enlem'],
                'Veren_Boylam': durum_dict[kaynak]['Boylam'],
                'Alan_Enlem': durum_dict[hedef]['Enlem'],
                'Alan_Boylam': durum_dict[hedef]['Boylam'],
                'Aktarilan_Su_Ton': miktar,
                'Birim_Maliyet_TL': maliyet,
                'Kazanilan_Tasarruf_TL': (65.0 - maliyet) * miktar,
                'Net_CO2_Kazanc_kg': co2_birim * miktar,
                'Tarih': hedef_gun_str
            })
            
    sonuc_df = pd.DataFrame(acilan_vanalar)
    sonuc_df.to_csv('optimizasyon_sonucu_live_edges.csv', index=False)
    
    return acilan_vanalar