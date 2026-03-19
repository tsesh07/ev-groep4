import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title="Dashboard Laadinfrastructuur NL",
    page_icon="⚡",
    layout="wide"
)

# --- 2. API CONFIGURATIE & DATA FUNCTIE ---


API_KEY = "952e5ef9-7c54-437e-9539-da81ab498118"

@st.cache_data(show_spinner="Bezig met voorladen van alle 5000 laadpalen (dit kan even duren)...")
def load_initial_data(limit=5000):
    url = f"https://api.openchargemap.io/v3/poi/?output=json&key={API_KEY}&countrycode=NL&maxresults={limit}"
    try:
        # TIMEOUT FIX: Verhoogd naar 90 seconden
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"⚠️ De OpenChargeMap API is momenteel te traag of onbereikbaar. Foutmelding: {e}")
        return [] # Return lege lijst zodat de app niet crasht

@st.cache_data
def process_data_for_insights(raw_data):
    if not raw_data: 
        return pd.DataFrame(columns=['Town', 'KW', 'Category', 'Title', 'IsFast'])
    
    data = []
    for site in raw_data:
        addr = site.get('AddressInfo') or {}
        conns = site.get('Connections') or []
        for conn in conns:
            kw = conn.get('PowerKW', 0) or 0
            cat = 'Traag (<11kW)'
            if kw >= 150: cat = 'Ultra-Snel (>150kW)'
            elif kw >= 43: cat = 'Snel (43-150kW)'
            elif kw >= 11: cat = 'Publiek (11-22kW)'
            data.append({
                'Town': addr.get('Town', 'Onbekend'),
                'KW': kw,
                'Category': cat,
                'Title': addr.get('Title'),
                'IsFast': kw > 40
            })
    return pd.DataFrame(data)

@st.cache_data
def laad_en_clean_cbs_data(bestandsnaam):
    try:
        df_cbs = pd.read_csv(bestandsnaam, sep=';')
        kolom_regio = "Regio's"
        kolom_inwoners = "Bevolking/Bevolkingssamenstelling op 1 januari/Totale bevolking (aantal)"
        df_cbs = df_cbs[[kolom_regio, kolom_inwoners]].copy()
        df_cbs.columns = ['Regio', 'Inwoners']
        df_cbs['Stad_Clean'] = df_cbs['Regio'].str.replace(r'\s*\(.*?\)\s*', '', regex=True).str.strip()
        df_cbs['Stad_Clean'] = df_cbs['Stad_Clean'].replace({
            "'s-Gravenhage": "Den Haag",
            "'s-Hertogenbosch": "Den Bosch"
        })
        df_cbs = df_cbs.dropna(subset=['Inwoners'])
        return df_cbs[['Stad_Clean', 'Inwoners']]
    except Exception as e:
        st.error(f"Fout bij inladen CBS data. Zorg dat '{bestandsnaam}' in de map staat. Fout: {e}")
        return pd.DataFrame()

# Direct laden bij opstarten
all_data_raw = load_initial_data(5000)
df = process_data_for_insights(all_data_raw)
cbs_data = laad_en_clean_cbs_data('regionaal_2025.csv')

PROVINCIE_COORDS = {
    "Alle": [52.1326, 5.2913], "Noord-Holland": [52.5206, 4.8889],
    "Zuid-Holland": [52.0202, 4.4938], "Utrecht": [52.0907, 5.1214],
    "Gelderland": [52.1182, 5.9224], "Noord-Brabant": [51.4827, 5.2321],
    "Overijssel": [52.4388, 6.5016], "Flevoland": [52.5270, 5.5942],
    "Groningen": [53.2194, 6.5665], "Friesland": [53.1143, 5.8508],
    "Drenthe": [52.8574, 6.6231], "Zeeland": [51.4940, 3.8497],
    "Limburg": [51.2111, 5.9382]
}

# --- 3. SIDEBAR NAVIGATIE ---
st.sidebar.title("⚡ EV Dashboard")
st.sidebar.markdown("Navigeer door de analyses over de Nederlandse laadinfrastructuur.")

page = st.sidebar.radio(
    "Selecteer een pagina:",
    ["🏠 Landingspagina", "🗺️ Interactieve Kaart", "📈 KPI Dashboard", "🔮 Voorspellend Model"]
)

st.sidebar.divider()
st.sidebar.info("Gemaakt door Juri van der Ster, Micha Bakker, Sai Hong Tse, Stijn Kooistra")

# --- 4. PAGINA LOGICA ---

if page == "🏠 Home":
    st.title("⚡ Is de Nederlandse laadinfrastructuur eerlijk verdeeld?")
    
    st.markdown("""
    Welkom bij ons Data Science Dashboard! Dit onderzoek focust op de **geografische spreiding** en **laadsnelheid** van laadpalen in Nederland. 
    Niet elke stad heeft evenveel behoefte aan snelladers, maar een structureel gebrek aan publieke laders in bepaalde regio's kan de transitie naar elektrisch rijden remmen.
    """)
    
    st.divider()

    if not df.empty:
        st.subheader("📊 De dataset in vogelvlucht")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔌 Totaal Punten", f"{len(df):,}".replace(',', '.'))
        c2.metric("⚡ Gem. Snelheid", f"{df['KW'].mean():.1f} kW")
        c3.metric("🏙️ Aantal Steden", df['Town'].nunique())
    
    st.divider()

    col_links, col_rechts = st.columns(2)
    
    with col_links:
        st.markdown("### 🧭 Wat vind je in dit dashboard?")
        st.markdown("""
        Gebruik het menu aan de linkerkant om door onze analyses te navigeren:
        * **🗺️ Kaart:** Een geografische weergave van de laadpalen, inclusief een filter voor snelladers.
        * **📈 KPI Dashboard:** Onze diepte-analyse waarbij we laadpalen afzetten tegen het aantal inwoners. Hier beantwoorden we de vraag over 'eerlijke verdeling'.
        * **📊 Inzichten & Vergelijking:** Grafieken die de kwaliteit (snelheid) afzetten tegen de kwantiteit (aantal) per stad.
        """)
        
    with col_rechts:
        st.markdown("### 🗄️ Onze Databronnen")
        st.info("""
        Voor deze case hebben we meerdere databronnen gecombineerd:
        1. **OpenChargeMap API:** Live data van beschikbare laadpalen (capaciteit, locatie, netwerk).
        2. **CBS StatLine (2025):** Regionale kerncijfers voor de demografische context (inwonersaantallen per gemeente).
        
        *Data inspectie en opschoning is in Python uitgevoerd om deze bronnen betrouwbaar te kunnen koppelen.*
        """)

    st.markdown("---")
    st.markdown("**Gemaakt door:** *Juri van der Ster, Stijn Kooistra, Sai-Hong Tse & Micha Bakker* | Minor Data Science")

elif page == "🗺️ Interactieve Kaart":
    st.title("🗺️ Interactieve Kaart")

    @st.cache_data
    def load_cbs_and_geo():
        import geopandas as gpd
        try:
            df_kaart = pd.read_csv('regionaal_2025.csv', sep=';')
            df_kaart = df_kaart[["Regio's", "Bevolking/Bevolkingssamenstelling op 1 januari/Bevolkingsdichtheid (aantal inwoners per km²)"]].copy()
            df_kaart.columns = ['Gemeente', 'Bevolkingsdichtheid']
            df_kaart['Gemeente'] = df_kaart['Gemeente'].str.replace(r'\s*\(.*?\)\s*', '', regex=True).str.strip()
            df_kaart['Bevolkingsdichtheid'] = pd.to_numeric(df_kaart['Bevolkingsdichtheid'], errors='coerce')
            df_kaart = df_kaart.dropna(subset=['Bevolkingsdichtheid'])
            
            gdf_kaart = gpd.read_file('https://cartomap.github.io/nl/wgs84/gemeente_2025.geojson')
            return df_kaart, gdf_kaart
        except Exception as e:
            return pd.DataFrame(), None

    df_cbs_kaart, gdf = load_cbs_and_geo()

    providers = sorted(list(set(s['OperatorInfo'].get('Title', 'Onbekend') for s in all_data_raw if s and s.get('OperatorInfo')))) if all_data_raw else []

    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        provincie_keuze = st.selectbox("1. Selecteer Provincie", options=list(PROVINCIE_COORDS.keys()))
    with col_f2:
        selected_providers = st.multiselect("2. Kies Providers", options=providers)
    with col_f3:
        type_filter = st.multiselect("3. Type Lader", options=["Normaal", "Snellader"], default=["Normaal", "Snellader"])
        
    aantal_tonen = st.slider("4. Maximaal aantal palen op kaart", 10, 5000, 1000)

    st.divider()

    col_map, col_text = st.columns([3, 1])
    
    with col_map:
        start_coords = PROVINCIE_COORDS.get(provincie_keuze, [52.1326, 5.2913])
        m = folium.Map(location=start_coords, zoom_start=7 if provincie_keuze == "Alle" else 9, tiles='cartodbpositron')
        
        if not df_cbs_kaart.empty and gdf is not None:
            folium.Choropleth(
                geo_data=gdf,
                name='👥 Bevolkingsdichtheid (CBS)',
                data=df_cbs_kaart,
                columns=['Gemeente', 'Bevolkingsdichtheid'],
                key_on='feature.properties.statnaam',
                fill_color='Blues',
                fill_opacity=0.6,
                line_opacity=0.2,
                legend_name='Bevolkingsdichtheid (inwoners per km²)',
                nan_fill_color='lightgrey',
                show=False 
            ).add_to(m)

        fg_clusters = folium.FeatureGroup(name="📍 Laadpaal Clusters")
        marker_cluster = MarkerCluster().add_to(fg_clusters)
        fg_buffers = folium.FeatureGroup(name="⭕ Afstand (500m)", show=False)
        
        filtered_points = []
        if all_data_raw:
            for station in all_data_raw:
                if not station: continue
                addr = station.get('AddressInfo', {})
                op_info = station.get('OperatorInfo')
                operator = op_info.get('Title', 'Onbekend') if op_info else 'Onbekend'
                title = addr.get('Title', '')

                if provincie_keuze != "Alle" and provincie_keuze.lower() not in str(addr.get('StateOrProvince', '')).lower():
                    continue
                if selected_providers and operator not in selected_providers:
                    continue
                
                is_fast = False
                max_pwr = 0
                for conn in station.get('Connections', []):
                    pwr = conn.get('PowerKW')
                    if pwr:
                        max_pwr = max(max_pwr, pwr)
                        if pwr > 40: is_fast = True
                
                cat = "Snellader" if is_fast else "Normaal"
                if cat in type_filter:
                    filtered_points.append((station, cat, max_pwr, operator, is_fast))

            for station, cat, pwr, op_name, is_fast in filtered_points[:aantal_tonen]:
                addr = station.get('AddressInfo', {})
                lat, lon = addr.get('Latitude'), addr.get('Longitude')
                
                if lat and lon:
                    color = "green" if is_fast else "blue"
                    icon_type = "bolt" if is_fast else "plug"
                    
                    folium.Marker(
                        location=[lat, lon],
                        popup=f"<b>{addr.get('Title')}</b><br>Provider: {op_name}<br>Vermogen: {pwr}kW",
                        tooltip=f"{op_name} ({cat})",
                        icon=folium.Icon(color=color, icon=icon_type, prefix="fa")
                    ).add_to(marker_cluster)
                    
                    folium.Circle(
                        location=[lat, lon],
                        radius=500,
                        color='crimson',
                        fill=True,
                        fill_color='crimson',
                        fill_opacity=0.1,
                        popup=f"Loopafstand: {addr.get('Title')}"
                    ).add_to(fg_buffers)

        fg_clusters.add_to(m)
        fg_buffers.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width="100%", height=600, returned_objects=[])

    with col_text:
        st.subheader("Legenda")
        st.success(f"Resultaten gevonden: {len(filtered_points)}")
        st.markdown("""
        **Lagen (Rechtsboven):**
        Je kunt de verschillende lagen aan- en uitzetten in het menu rechtsboven op de kaart. 
        Vink **👥 Bevolkingsdichtheid** aan om te zien of palen in dichtbevolkte gebieden staan!
        
        **Markers:**
        * 🟢 = Snellader (>40kW)
        * 🔵 = Normale lader
        * ⭕ = 500m Loopafstand
        """)

elif page == "📈 KPI Dashboard":
    st.title("📈 KPI Dashboard: Is het eerlijk verdeeld?")
    st.markdown("Dit dashboard combineert onze live laadpaal-data met CBS-inwonersaantallen. Zo zien we pas écht of de laadinfrastructuur eerlijk is verdeeld onder de bevolking.")

    if df.empty or cbs_data.empty:
        st.warning("Data is nog aan het laden of de CBS data ontbreekt (controleer de bestandsnaam).")
    else:
        city_stats = df.groupby('Town').agg({'KW': ['count', 'mean']}).reset_index()
        city_stats.columns = ['Stad', 'Aantal_Punten', 'Gem_Snelheid']

        merged_df = pd.merge(city_stats, cbs_data, left_on='Stad', right_on='Stad_Clean', how='inner')
        
        if not merged_df.empty:
            merged_df['Palen_per_10k'] = (merged_df['Aantal_Punten'] / merged_df['Inwoners']) * 10000
            merged_df = merged_df.sort_values(by='Palen_per_10k', ascending=False).reset_index(drop=True)

            st.subheader("Landelijk Overzicht (Top & Flop)")
            col1, col2, col3 = st.columns(3)
            
            beste_stad = merged_df.iloc[0]
            slechtste_stad = merged_df.iloc[-1]
            gemiddeld_nl = merged_df['Palen_per_10k'].mean()

            with col1:
                st.metric("🥇 Beste dekking", beste_stad['Stad'], f"{beste_stad['Palen_per_10k']:.1f} / 10k inw.")
            with col2:
                st.metric("⚠️ Slechtste dekking", slechtste_stad['Stad'], f"{slechtste_stad['Palen_per_10k']:.1f} / 10k inw.", delta_color="inverse")
            with col3:
                st.metric("📊 Gemiddeld (Dataset)", f"{gemiddeld_nl:.1f}", "palen per 10k inw.", delta_color="off")

            st.divider()

            st.subheader("🔍 Interactieve Stads-Analyse")
            st.markdown("Kies een specifieke stad om de lokale prestaties te bekijken, of bekijk de landelijke Top 20.")
            
            opties = ["Landelijke Top 20 Grafiek"] + sorted(merged_df['Stad'].tolist())
            gekozen_stad = st.selectbox("Selecteer weergave:", opties)

            if gekozen_stad == "Landelijke Top 20 Grafiek":
                fig_kpi = px.bar(
                    merged_df.head(20), 
                    x='Stad', 
                    y='Palen_per_10k',
                    color='Palen_per_10k',
                    color_continuous_scale='RdYlGn',
                    labels={'Palen_per_10k': 'Palen per 10.000 inw.'},
                    title="Top 20 steden met de meeste laadpunten per inwoner"
                )
                st.plotly_chart(fig_kpi, use_container_width=True)
                
            else:
                stad_data = merged_df[merged_df['Stad'] == gekozen_stad].iloc[0]
                
                st.markdown(f"#### Detailgegevens voor: **{gekozen_stad}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("👥 Inwoners (CBS)", f"{int(stad_data['Inwoners']):,}".replace(',', '.'))
                c2.metric("🔌 Publieke Palen (API)", int(stad_data['Aantal_Punten']))
                c3.metric("⚖️ KPI (Palen per 10k inw.)", f"{stad_data['Palen_per_10k']:.1f}")
                
                stad_api_data = df[df['Town'] == gekozen_stad]
                if not stad_api_data.empty:
                    st.markdown("**Verdeling van Laadsnelheden in deze stad:**")
                    cat_counts = stad_api_data['Category'].value_counts()
                    st.bar_chart(cat_counts)
        else:
             st.warning("Kon stadsnamen niet aan elkaar koppelen.")

elif page == "🔮 Voorspellend Model":
    st.title("🔮 Voorspellend Model: Wie blijft achter?")
    st.markdown("""
    Dit model gebruikt **lineaire regressie** om op basis van het inwonersaantal te voorspellen hoeveel laadpalen 
    een gemeente *zou moeten* hebben. Het verschil tussen de voorspelling en de werkelijkheid noemen we het **residu**.
    
    * Een **negatief residu** betekent: deze gemeente heeft *minder* palen dan verwacht → mogelijk onderbedeeld.
    * Een **positief residu** betekent: deze gemeente heeft *meer* palen dan verwacht → relatief goed bedeeld.
    """)

    if df.empty or cbs_data.empty:
        st.warning("Data is nog aan het laden of de CBS data ontbreekt.")
    else:
        city_stats = df.groupby('Town').agg({'KW': ['count']}).reset_index()
        city_stats.columns = ['Stad', 'Aantal_Punten']
        merged_df = pd.merge(city_stats, cbs_data, left_on='Stad', right_on='Stad_Clean', how='inner')
        merged_df = merged_df.dropna(subset=['Inwoners', 'Aantal_Punten'])

        if not merged_df.empty:
            X = merged_df[['Inwoners']].values
            y = merged_df['Aantal_Punten'].values
            model = LinearRegression()
            model.fit(X, y)

            merged_df['Voorspeld'] = model.predict(X)
            merged_df['Residu'] = merged_df['Aantal_Punten'] - merged_df['Voorspeld']
            merged_df = merged_df.sort_values('Residu')

            r2 = model.score(X, y)
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("📐 Model R²", f"{r2:.2f}", "verklaringskracht")
            col2.metric("📉 Grootste achterblijver", merged_df.iloc[0]['Stad'], f"{merged_df.iloc[0]['Residu']:.0f} palen tekort")
            col3.metric("📈 Grootste voorloper", merged_df.iloc[-1]['Stad'], f"+{merged_df.iloc[-1]['Residu']:.0f} palen extra")

            st.divider()

            st.subheader("1. Regressielijn: Inwoners vs. Laadpalen")
            st.markdown("_Punten ver onder de lijn zijn potentieel onderbedeeld._")

            fig_reg = px.scatter(
                merged_df, x='Inwoners', y='Aantal_Punten',
                hover_name='Stad',
                labels={'Inwoners': 'Aantal Inwoners (CBS)', 'Aantal_Punten': 'Aantal Laadpunten (API)'},
                title="Lineaire regressie: verwacht vs. werkelijk aantal laadpunten"
            )
            x_line = np.linspace(merged_df['Inwoners'].min(), merged_df['Inwoners'].max(), 100)
            y_line = model.predict(x_line.reshape(-1, 1))
            fig_reg.add_scatter(x=x_line, y=y_line, mode='lines', name='Verwachting (model)',
                                line=dict(color='red', dash='dash'))
            st.plotly_chart(fig_reg, use_container_width=True)

            st.subheader("2. Residuenanalyse: Wie zit boven of onder de verwachting?")
            st.markdown("_Rood = minder palen dan verwacht (achterblijvers). Groen = meer dan verwacht (voorlopers)._")

            top_bottom = pd.concat([merged_df.head(15), merged_df.tail(15)]).drop_duplicates()
            top_bottom['Kleur'] = top_bottom['Residu'].apply(lambda x: 'Achterblijver' if x < 0 else 'Voorloper')

            fig_residu = px.bar(
                top_bottom.sort_values('Residu'),
                x='Residu', y='Stad',
                color='Kleur',
                orientation='h',
                color_discrete_map={'Achterblijver': '#EF553B', 'Voorloper': '#00CC96'},
                labels={'Residu': 'Residu (werkelijk − voorspeld)', 'Stad': ''},
                title="Top 15 achterblijvers en voorlopers"
            )
            st.plotly_chart(fig_residu, use_container_width=True)

            st.subheader("3. Volledige Residuentabel")
            st.markdown("Bekijk voor elke gemeente het werkelijke aantal, de voorspelling en het verschil.")
            tabel = merged_df[['Stad', 'Inwoners', 'Aantal_Punten', 'Voorspeld', 'Residu']].copy()
            tabel['Voorspeld'] = tabel['Voorspeld'].round(1)
            tabel['Residu'] = tabel['Residu'].round(1)
            tabel['Inwoners'] = tabel['Inwoners'].astype(int)
            tabel['Aantal_Punten'] = tabel['Aantal_Punten'].astype(int)
            tabel = tabel.sort_values('Residu').reset_index(drop=True)
            st.dataframe(tabel, use_container_width=True)