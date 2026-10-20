from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List, Dict, Any
import uvicorn
import os
import joblib
from optimizasyon import run_daily_simulation

app = FastAPI(title="Gebze Water Matchmaker API", version="1.0.0")

ai_models = {}

RESULTS_FILE_PATH = "optimizasyon_sonucu_live_edges.csv"

@app.get("/api/get_aggregated_metrics")
def get_aggregated_metrics(target_time: str, period: str = Query("aylik")):
    try:
        # 1. AI'nin ürettiği o günkü sonuçları al
        df_live = pd.read_csv("optimizasyon_sonucu_live_edges.csv")
        gunluk_kurtarilan = float(df_live['Aktarilan_Su_Ton'].sum())
        gunluk_tasarruf = float(df_live['Kazanilan_Tasarruf_TL'].sum())
        
        # 2. Tüm yılın olduğu ana veri dosyasını oku
        csv_path = "gunluk_su_verisi_2027_2.csv" if os.path.exists("gunluk_su_verisi_2027_2.csv") else "gunluk_su_verisi_2027.csv"
        df_raw = pd.read_csv(csv_path)
        df_raw['Tarih'] = pd.to_datetime(df_raw['Tarih'])
        target_date = pd.to_datetime(target_time)

        # 3. O günün gerçek su talebini bul
        gunluk_talep = df_raw[df_raw['Tarih'] == target_date]['Temiz_Su_Talebi_m3'].sum()
        if gunluk_talep == 0: gunluk_talep = 1 # Sıfıra bölme hatasını engelle

        # 4. Seçilen periyodun (Ay/Yıl) GERÇEK toplam su talebini hesapla
        if period == "aylik":
            mask = (df_raw['Tarih'].dt.year == target_date.year) & (df_raw['Tarih'].dt.month == target_date.month)
        elif period == "yillik":
            mask = df_raw['Tarih'].dt.year == target_date.year
        else:
            mask = df_raw['Tarih'] == target_date

        periyot_toplam_talep = df_raw[mask]['Temiz_Su_Talebi_m3'].sum()

        # 5. CSV'deki gerçek dalgalanmalara göre oranla
        carpan = periyot_toplam_talep / gunluk_talep

        return {
            "toplam_kurtarilan_su_ton": round(gunluk_kurtarilan * carpan, 2),
            "toplam_tasarruf_tl": round(gunluk_tasarruf * carpan, 2),
            "karbon_ayak_izi_kredisi_euro": round((gunluk_kurtarilan * carpan) * 1.55, 2)
        }
    except Exception as e:
        print(f"Hata: {e}")
        return {"toplam_kurtarilan_su_ton": 0, "toplam_tasarruf_tl": 0, "karbon_ayak_izi_kredisi_euro": 0}


@app.get("/api/get_aggregated_edges")
def get_aggregated_edges(target_time: str, period: str = Query("aylik")):
    try:
        # Transfer listesini (aktif vanaları) de aynı gerçek veri çarpanıyla büyüteceğiz
        df_live = pd.read_csv("optimizasyon_sonucu_live_edges.csv")
        
        csv_path = "gunluk_su_verisi_2027_2.csv" if os.path.exists("gunluk_su_verisi_2027_2.csv") else "gunluk_su_verisi_2027.csv"
        df_raw = pd.read_csv(csv_path)
        df_raw['Tarih'] = pd.to_datetime(df_raw['Tarih'])
        target_date = pd.to_datetime(target_time)

        gunluk_talep = df_raw[df_raw['Tarih'] == target_date]['Temiz_Su_Talebi_m3'].sum()
        if gunluk_talep == 0: gunluk_talep = 1

        if period == "aylik":
            mask = (df_raw['Tarih'].dt.year == target_date.year) & (df_raw['Tarih'].dt.month == target_date.month)
        elif period == "yillik":
            mask = df_raw['Tarih'].dt.year == target_date.year
        else:
            mask = df_raw['Tarih'] == target_date

        periyot_toplam_talep = df_raw[mask]['Temiz_Su_Talebi_m3'].sum()
        carpan = periyot_toplam_talep / gunluk_talep

        df_nodes = load_nodes()
        result = []

        for _, row in df_live.iterrows():
            try:
                kaynak_isim = df_nodes[df_nodes['Fabrika_ID'] == row['Veren_Fabrika']].iloc[0]['Gercek_Isim']
                hedef_isim = df_nodes[df_nodes['Fabrika_ID'] == row['Alan_Fabrika']].iloc[0]['Gercek_Isim']
            except:
                kaynak_isim = row.get('Veren_Isim', str(row['Veren_Fabrika']))
                hedef_isim = row.get('Alan_Isim', str(row['Alan_Fabrika']))

            result.append({
                "Veren_Fabrika": row['Veren_Fabrika'],
                "Alan_Fabrika": row['Alan_Fabrika'],
                "Veren_Isim": kaynak_isim,
                "Alan_Isim": hedef_isim,
                "Aktarilan_Su_Ton": round(row['Aktarilan_Su_Ton'] * carpan, 2),
                "Kazanilan_Tasarruf_TL": round(row['Kazanilan_Tasarruf_TL'] * carpan, 2)
            })
        return result
    except Exception as e:
        print(f"Hata: {e}")
        return []
    
@app.on_event("startup")
def load_ai_models():
    print("Yapay Zeka modelleri RAM'e yükleniyor...")
    try:
        ai_models['forecaster'] = joblib.load('model_b_forecaster.joblib')
        ai_models['scorer'] = joblib.load('model_a_scorer_skky.joblib')
        print("Modeller başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_nodes() -> pd.DataFrame:
    if not os.path.exists('gercek_nodes_factories.csv'):
        raise HTTPException(status_code=404, detail="Nodes dosyasi bulunamadi.")
    return pd.read_csv('gercek_nodes_factories.csv')

def load_live_edges() -> pd.DataFrame:
    if not os.path.exists('optimizasyon_sonucu_live_edges.csv'):
        return pd.DataFrame() 
    return pd.read_csv('optimizasyon_sonucu_live_edges.csv')

def load_physical_edges() -> pd.DataFrame:
    if not os.path.exists('gercek_edges_network.csv'):
        raise HTTPException(status_code=404, detail="Fiziksel baglanti dosyasi bulunamadi.")
    return pd.read_csv('gercek_edges_network.csv')

@app.get("/api/simulate", tags=["Simulation"])
def simulate_day(target_time: str = Query("2027-01-01", description="Simüle edilecek tarih (YYYY-MM-DD)")):
    '''
    Belirtilen gün için forecaster modelinden tahminleri alır ve optimizasyon motorunu çalıştırır.
    '''
    if 'forecaster' not in ai_models or 'scorer' not in ai_models:
        raise HTTPException(status_code=500, detail="Yapay Zeka modelleri yüklenemedi.")
    
    try:
        acilan_vanalar = run_daily_simulation(target_time, ai_models['forecaster'], ai_models['scorer'])
        return {
            "status": "success", 
            "target_time": target_time, 
            "acilan_vana_sayisi": len(acilan_vanalar),
            "vanalar": acilan_vanalar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "success", "message": "Gebze Water Matchmaker Backend Calisiyor."}

@app.get("/api/get_nodes", response_model=List[Dict[str, Any]], tags=["Map Data"])
def get_nodes():
    df = load_nodes()
    return df.to_dict(orient="records")

@app.get("/api/get_live_edges", response_model=List[Dict[str, Any]], tags=["Map Data"])
def get_live_edges():
    df_live = load_live_edges()
    if df_live.empty:
        return []

    df_nodes = load_nodes()
    result = []
    for _, row in df_live.iterrows():
        kaynak_data = df_nodes[df_nodes['Fabrika_ID'] == row['Veren_Fabrika']].iloc[0]
        hedef_data = df_nodes[df_nodes['Fabrika_ID'] == row['Alan_Fabrika']].iloc[0]

        result.append({
            "Veren_ID": row['Veren_Fabrika'],
            "Veren_Isim": kaynak_data['Gercek_Isim'],
            "Veren_Enlem": kaynak_data['Enlem'],
            "Veren_Boylam": kaynak_data['Boylam'],
            "Alan_ID": row['Alan_Fabrika'],
            "Alan_Isim": hedef_data['Gercek_Isim'],
            "Alan_Enlem": hedef_data['Enlem'],
            "Alan_Boylam": hedef_data['Boylam'],
            "Aktarilan_Su_Ton": float(row['Aktarilan_Su_Ton']),
            "Kazanilan_Tasarruf_TL": float(row['Kazanilan_Tasarruf_TL'])
        })

    return result

@app.get("/api/get_physical_network", response_model=List[Dict[str, Any]], tags=["Map Data"])
def get_physical_network():
    df_edges = load_physical_edges()
    df_nodes = load_nodes()

    result = []
    for _, row in df_edges.iterrows():
        kaynak_data = df_nodes[df_nodes['Fabrika_ID'] == row['Kaynak_Fabrika']].iloc[0]
        hedef_data = df_nodes[df_nodes['Fabrika_ID'] == row['Hedef_Fabrika']].iloc[0]

        result.append({
            "Kaynak_ID": row['Kaynak_Fabrika'],
            "Hedef_ID": row['Hedef_Fabrika'],
            "Kaynak_Enlem": kaynak_data['Enlem'],
            "Kaynak_Boylam": kaynak_data['Boylam'],
            "Hedef_Enlem": hedef_data['Enlem'],
            "Hedef_Boylam": hedef_data['Boylam'],
            "Mesafe_km": float(row['Mesafe_km'])
        })

    return result

@app.get("/api/get_dashboard_metrics", tags=["Dashboard"])
def get_dashboard_metrics():
    df_live = load_live_edges()
    if df_live.empty:
        return {
            "toplam_kurtarilan_su_ton": 0,
            "toplam_tasarruf_tl": 0,
            "karbon_ayak_izi_kredisi_euro": 0
        }

    toplam_su = float(df_live['Aktarilan_Su_Ton'].sum())
    toplam_tasarruf = float(df_live['Kazanilan_Tasarruf_TL'].sum())
    
    if 'Net_CO2_Kazanc_kg' in df_live.columns:
        toplam_co2 = float(df_live['Net_CO2_Kazanc_kg'].sum())
    else:
        toplam_co2 = toplam_su * 0.5 
        
    karbon_kredisi = round(toplam_co2 * 0.02, 2)

    return {
        "toplam_kurtarilan_su_ton": round(toplam_su, 2),
        "toplam_tasarruf_tl": round(toplam_tasarruf, 2),
        "karbon_ayak_izi_kredisi_euro": karbon_kredisi
    }

if __name__ == "__main__":
    print("FastAPI Backend hazirlaniyor...")