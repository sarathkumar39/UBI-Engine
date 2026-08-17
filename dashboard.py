"""
Streamlit dashboard for the UBI POC.
Professional insurer-style layout for demo presentation.
"""
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
import uuid
from datetime import datetime
import base64

st.set_page_config(page_title='UBI Pricing & Billing Engine', layout='wide')

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f3f7fb 0%, #eef4fa 100%); }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .header-shell {
        background: linear-gradient(135deg, #0d3b66 0%, #1d5d9c 100%);
        padding: 1.5rem 1.5rem 1.0rem 1.5rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(13, 59, 102, 0.18);
        margin-bottom: 1.2rem;
    }
    .header-shell h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .header-sub {
        color: rgba(255,255,255,0.8);
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #dfeaf4;
        border-radius: 14px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 3px 10px rgba(18, 38, 63, 0.04);
        height: 100%;
    }
    .metric-card.compact {
        padding: 0.45rem 0.6rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(18,38,63,0.03);
    }
    .metric-label {
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #5d7386;
        font-weight: 700;
    }
    .metric-label.compact {
        font-size: 0.68rem;
        font-weight: 700;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #123a62;
        margin-top: 0.3rem;
    }
    .metric-value.compact {
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 0.15rem;
    }
    /* KPI color variants */
    .kpi-green { background: #e9f7ef; border-color: #d1eedb; }
    .kpi-warning { background: #fff6e6; border-color: #ffe7b8; }
    .kpi-danger { background: #fff0f0; border-color: #f5c6c6; }

    .section-card {
        background: white;
        border: 1px solid #e2eaf3;
        border-radius: 16px;
        padding: 1rem 1rem 1.2rem 1rem;
        box-shadow: 0 3px 12px rgba(18, 38, 63, 0.04);
        margin-top: 1rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #143a5b;
        margin-bottom: 0.8rem;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: #dff3e7;
        color: #0d5a38;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .warning-badge {
        background: #fff3d9;
        color: #7b4a00;
    }
    .danger-badge {
        background: #fde5e5;
        color: #8d2d2d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = os.path.join('artifacts', 'model.joblib')
SAMPLE_PRED = os.path.join('artifacts', 'sample_predictions.csv')
DEMO_DRIVERS_PATH = os.path.join('data', 'demo_drivers.csv')
FEATURE_COLS = ['duration_sec','distance_km','avg_speed_kmh','hard_brakes','rapid_accels','start_hour','weekday','speed_per_10km','brake_accel_sum','is_night']


# UI helpers
def format_currency(amount, symbol='₹'):
    try:
        return f"{symbol}{float(amount):,.2f}"
    except Exception:
        return str(amount)


def format_pct(pct):
    try:
        return f"{float(pct):.1f}%"
    except Exception:
        return str(pct)


def render_card(label, value, status=None, compact=False):
    # status -> 'green' | 'warning' | 'danger' or None
    classes = ['metric-card']
    if compact:
        classes.append('compact')
    if status:
        classes.append(f'kpi-{status}')
    class_str = ' '.join(classes)
    label_cls = 'metric-label compact' if compact else 'metric-label'
    value_cls = 'metric-value compact' if compact else 'metric-value'
    html = f"<div class=\"{class_str}\"><div class=\"{label_cls}\">{label}</div><div class=\"{value_cls}\">{value}</div></div>"
    st.markdown(html, unsafe_allow_html=True)


def prepare_features_from_row(row):
    df = pd.DataFrame([row])
    df['speed_per_10km'] = df['avg_speed_kmh'] / 10.0
    df['brake_accel_sum'] = df['hard_brakes'] + df['rapid_accels']
    df['is_night'] = df['start_hour'].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)
    return df[FEATURE_COLS]


def score_trip_with_model(model, row):
    """Return raw probability and a UI risk score mapped to 0-100.

    The ML model gives probability where higher = riskier. For the demo dashboard and
    pricing logic, we invert that into a risk score where higher = safer/lower risk so
    the premium adjustments are easier to explain to business users.
    """
    X = prepare_features_from_row(row)
    prob = float(model.predict_proba(X)[0, 1])
    risk_score = round((1.0 - prob) * 100.0, 2)
    return prob, risk_score


st.markdown(
    """
    <div class="header-shell">
        <h1>UBI Pricing & Billing Engine</h1>
        <div class="header-sub">Usage-based insurance portfolio risk, premium, invoice, and fraud monitoring</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error(f'Model not found at {MODEL_PATH}. Run training first.')
else:
    model = joblib.load(MODEL_PATH)

    # --- Add Driver / Trip panel (no login) ---
    st.markdown('<div class="section-card"><div class="section-title">Add Driver / Trip</div></div>', unsafe_allow_html=True)
    with st.form('add_driver_form'):
        driver_id = st.text_input('Driver ID', value='driver_001', key='driver_id')
        # device id default uses the entered driver id from session_state
        device_default = f"dev_{st.session_state.get('driver_id','driver_001')}_01"
        device_id = st.text_input('Device ID', value=device_default, key='device_id')
        trip_type = st.selectbox('Trip type', ['commute','long_haul'], key='trip_type')
        start_hour_u = st.number_input('Start hour', min_value=0, max_value=23, value=9, key='start_hour_u')
        duration_u = st.number_input('Duration (sec)', min_value=60, max_value=6*60*60, value=900, key='duration_u')
        avg_speed_u = st.number_input('Avg speed (km/h)', min_value=0.0, max_value=300.0, value=60.0, format="%.1f", key='avg_speed_u')
        hard_brakes_u = st.number_input('Hard brakes', min_value=0, max_value=20, value=0, key='hard_brakes_u')
        rapid_accels_u = st.number_input('Rapid accels', min_value=0, max_value=20, value=0, key='rapid_accels_u')
        monthly_mileage_u = st.number_input('Monthly mileage', min_value=0, max_value=50000, value=1200, key='monthly_mileage_u')
        is_ev_u = st.checkbox('Is EV', value=False, key='is_ev_u')
        safe_driver_u = st.checkbox('Safe driver (no events)', value=False, key='safe_driver_u')
        persist = st.checkbox('Persist trip to data/trips.csv', value=False, key='persist')
        submit_trip = st.form_submit_button('Add trip, compute premium & optionally invoice')

    # Portfolio KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Portfolio risk</div><div class="metric-value">72.4</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-label">Avg premium</div><div class="metric-value">₹32,880</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-label">Fraud alerts</div><div class="metric-value">3</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-label">Active drivers</div><div class="metric-value">100</div></div>', unsafe_allow_html=True)

    # User form submission handling with confirmations and focus improvements
    if submit_trip:
        # build trip dict for scoring
        distance_u = round(avg_speed_u * (duration_u / 3600.0), 2)
        trip_row = {
            'duration_sec': int(duration_u),
            'distance_km': float(distance_u),
            'avg_speed_kmh': float(avg_speed_u),
            'hard_brakes': int(hard_brakes_u),
            'rapid_accels': int(rapid_accels_u),
            'start_hour': int(start_hour_u),
            'weekday': int(datetime.now().weekday()),
        }
        prob_u, risk_score_u = score_trip_with_model(model, trip_row)
        st.markdown(
            f"**Risk probability:** {prob_u:.4f} ({prob_u * 100:.1f}%) — "
            f"**Safety score:** {prob_u * 100:.2f}"
        )
        # compute premium
        from pricing import compute_premium
        breakdown_u = compute_premium(risk_score=risk_score_u, monthly_mileage=float(monthly_mileage_u), is_ev=bool(is_ev_u), safe_driver=bool(safe_driver_u))
        st.subheader('Premium breakdown')
        with st.expander('Premium breakdown (details)', expanded=False):
            col_left, col_right = st.columns(2)
            with col_left:
                render_card('Base premium', format_currency(breakdown_u.get('base_premium', 0)))
                render_card('Risk adjustment', format_pct(breakdown_u.get('risk_adj_pct', 0)))
                render_card('Mileage adjustment', format_pct(breakdown_u.get('mileage_adj_pct', 0)))
                render_card('EV adjustment', format_pct(breakdown_u.get('ev_adj_pct', 0)))
            with col_right:
                render_card('Safe driver adjustment', format_pct(breakdown_u.get('safe_adj_pct', 0)))
                render_card('Premium before tax', format_currency(breakdown_u.get('premium_before_tax', 0)))
                render_card('Tax', f"{format_pct(breakdown_u.get('tax_pct', 0))}  ({format_currency(breakdown_u.get('tax', 0))})")
        st.metric('Total premium due (incl. tax)', format_currency(breakdown_u.get('total_due', 0)))

        # Offer invoice generation with confirmation
        if st.button('Prepare Invoice for this driver'):
            st.session_state['pending_invoice'] = {'driver_id': driver_id, 'period': datetime.utcnow().strftime('%Y-%m'), 'breakdown': breakdown_u}
            st.warning('Invoice prepared. Click Confirm to generate.')
        if st.session_state.get('pending_invoice'):
            if st.button('Confirm generate invoice'):
                try:
                    from billing import generate_invoice
                    pending = st.session_state.pop('pending_invoice')
                    inv = generate_invoice(driver_id=pending['driver_id'], period=pending['period'], premium_breakdown=pending['breakdown'])
                    st.success(f"Invoice {inv.get('invoice_id')} generated")
                    # show concise invoice summary instead of raw JSON
                    try:
                        st.markdown(f"**Invoice ID:** {inv.get('invoice_id')}  \n**Total Due:** ₹{inv.get('total_due')}  \n**Due date:** {inv.get('due_date')}")
                    except Exception:
                        pass
                    # If a PDF was generated, offer download and inline preview
                    pdf_path = inv.get('pdf_path')
                    if pdf_path and os.path.exists(pdf_path):
                        try:
                            with open(pdf_path, 'rb') as pf:
                                pdf_bytes = pf.read()
                            st.download_button('Download invoice PDF', data=pdf_bytes, file_name=os.path.basename(pdf_path), mime='application/pdf')
                            # embed inline via base64
                            b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                            pdf_display = f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='600px'></iframe>"
                            components.html(pdf_display, height=650)
                        except Exception as _e:
                            st.error('Failed to read or display PDF: ' + str(_e))
                    elif inv.get('pdf_error'):
                        st.warning(inv.get('pdf_error'))
                except Exception as _e:
                    st.error('Failed to generate invoice: ' + str(_e))

        # Prepare row to persist (but require explicit confirmation before writing)
        csv_path = os.path.join('data', 'trips.csv')
        start_ts_str = datetime.utcnow().isoformat()
        end_ts_str = (datetime.utcnow() + pd.to_timedelta(int(duration_u), unit='s')).isoformat()
        start_lat = 12.971598
        start_lon = 77.594566
        end_lat = start_lat
        end_lon = start_lon
        row_to_write = {
            'driver_id': driver_id,
            'device_id': device_id,
            'trip_id': str(uuid.uuid4()),
            'trip_type': trip_type,
            'start_ts': start_ts_str,
            'end_ts': end_ts_str,
            'duration_sec': int(duration_u),
            'distance_km': float(distance_u),
            'avg_speed_kmh': float(avg_speed_u),
            'hard_brakes': int(hard_brakes_u),
            'rapid_accels': int(rapid_accels_u),
            'start_hour': int(start_hour_u),
            'weekday': int(datetime.now().weekday()),
            'start_lat': round(start_lat, 6),
            'start_lon': round(start_lon, 6),
            'end_lat': round(end_lat, 6),
            'end_lon': round(end_lon, 6),
            'risk_score': round(1.0 - prob_u, 4),
            'label': 0,
        }

        if persist:
            st.warning('Persist selected — please Confirm to save this trip to data/trips.csv')
            if st.button('Confirm save trip'):
                import csv
                fieldnames = ['driver_id','device_id','trip_id','trip_type','start_ts','end_ts','duration_sec','distance_km','avg_speed_kmh','hard_brakes','rapid_accels','start_hour','weekday','start_lat','start_lon','end_lat','end_lon','risk_score','label']
                file_exists = os.path.exists(csv_path)
                try:
                    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row_to_write)
                    st.success(f'Appended trip to {csv_path}')
                except Exception as e:
                    st.error('Failed to append trip: ' + str(e))

        # Auto-focus driver id input after submit to speed repeated demo entries
        focus_script = '<script>const el=document.querySelector("input[aria-label=\'Driver ID\']"); if(el){el.focus(); el.select();}</script>'
        try:
            components.html(focus_script, height=0)
        except Exception:
            # components may not be allowed in some older streamlit versions; ignore silently
            pass

    # --- Per-driver persisted trip summary (visual) ---
    st.markdown('<div class="section-card"><div class="section-title">Per-driver trip summary</div></div>', unsafe_allow_html=True)
    trips_csv_path = os.path.join('data', 'trips.csv')
    if os.path.exists(trips_csv_path):
        trips_df = pd.read_csv(trips_csv_path)
        # Basic KPIs
        total_trips = len(trips_df)
        avg_risk = trips_df['risk_score'].mean() if 'risk_score' in trips_df.columns else None
        avg_speed = trips_df['avg_speed_kmh'].mean() if 'avg_speed_kmh' in trips_df.columns else None
        drivers = sorted(trips_df['driver_id'].dropna().unique().tolist())
        st.write(f"Total saved trips: {total_trips}")
        kcols = st.columns(3)
        with kcols[0]:
            render_card('Total trips', total_trips)
        with kcols[1]:
            render_card('Avg risk score', f"{avg_risk:.2f}" if avg_risk is not None else 'N/A')
        with kcols[2]:
            render_card('Avg speed (km/h)', f"{avg_speed:.1f}" if avg_speed is not None else 'N/A')

        # Compact KPI grid — conserve space while showing key metrics
        def render_kpi_grid(items, cols=3, compact=True):
            rows = [items[i:i+cols] for i in range(0, len(items), cols)]
            for row in rows:
                cols_ui = st.columns(len(row))
                for c, item in zip(cols_ui, row):
                    with c:
                        # item can be (label, value) or (label, value, status)
                        if len(item) == 2:
                            label, value = item
                            status = None
                        else:
                            label, value, status = item
                        render_card(label, value, status=status, compact=compact)

        total_mileage = trips_df['distance_km'].sum() if 'distance_km' in trips_df.columns else None
        avg_trip_duration_min = (trips_df['duration_sec'].mean() / 60.0) if 'duration_sec' in trips_df.columns else None
        unique_drivers = trips_df['driver_id'].nunique()
        percent_ev = None
        if 'is_ev' in trips_df.columns:
            try:
                percent_ev = (trips_df['is_ev'].astype(bool).mean() * 100.0)
            except Exception:
                percent_ev = None

        # Load KPI thresholds from fraud config if available
        try:
            from fraud import load_config as load_fraud_config
            cfg_kpi = load_fraud_config()
            high_risk_thr = float(cfg_kpi.get('high_risk_threshold', 40.0))
            medium_risk_thr = float(cfg_kpi.get('medium_risk_threshold', 70.0))
            night_start = int(cfg_kpi.get('night_start_hour', 22))
            night_end = int(cfg_kpi.get('night_end_hour', 5))
        except Exception:
            high_risk_thr = 40.0
            medium_risk_thr = 70.0
            night_start = 22
            night_end = 5

        # percent high-risk trips
        percent_high_risk = None
        if 'risk_score' in trips_df.columns:
            try:
                percent_high_risk = (trips_df['risk_score'] < high_risk_thr).mean() * 100.0
            except Exception:
                percent_high_risk = None

        # Average monthly trips per driver
        avg_monthly_trips_per_driver = None
        try:
            if 'start_ts' in trips_df.columns:
                trips_df['start_ts_dt'] = pd.to_datetime(trips_df['start_ts'], errors='coerce')
                trips_df['year_month'] = trips_df['start_ts_dt'].dt.to_period('M')
                per_driver_month = trips_df.groupby(['driver_id', 'year_month']).size().groupby('driver_id').mean()
                avg_monthly_trips_per_driver = per_driver_month.mean()
        except Exception:
            avg_monthly_trips_per_driver = None

        median_trip_distance = trips_df['distance_km'].median() if 'distance_km' in trips_df.columns else None

        # percent trips during night hours
        percent_night = None
        try:
            if 'start_hour' in trips_df.columns:
                if night_start <= night_end:
                    is_night = trips_df['start_hour'].between(night_start, night_end)
                else:
                    is_night = (~trips_df['start_hour'].between(night_end+1, night_start-1))
                percent_night = is_night.mean() * 100.0
            elif 'start_ts' in trips_df.columns:
                trips_df['start_ts_dt'] = pd.to_datetime(trips_df['start_ts'], errors='coerce')
                hrs = trips_df['start_ts_dt'].dt.hour.dropna().astype(int)
                if night_start <= night_end:
                    percent_night = ((hrs >= night_start) & (hrs <= night_end)).mean() * 100.0
                else:
                    percent_night = ((hrs >= night_start) | (hrs <= night_end)).mean() * 100.0
        except Exception:
            percent_night = None

        # Fraud alert count and percent flagged
        fraud_count = None
        fraud_pct = None
        try:
            from fraud import find_fraud_from_csv
            alerts = find_fraud_from_csv(trips_csv_path)
            fraud_count = len(alerts)
            fraud_trip_ids = set(a.get('trip_id') for a in alerts if a.get('trip_id'))
            if len(trips_df) > 0:
                fraud_pct = (len(fraud_trip_ids) / len(trips_df)) * 100.0
        except Exception:
            fraud_count = None
            fraud_pct = None

        # Avg premium per driver (if premium_total exists)
        avg_premium_per_driver = None
        try:
            if 'premium_total' in trips_df.columns:
                per_driver_prem = trips_df.groupby('driver_id')['premium_total'].mean()
                avg_premium_per_driver = per_driver_prem.mean()
        except Exception:
            avg_premium_per_driver = None

        # Build compact KPI list and render as a grid (aim for ~15 KPIs)
        avg_events_per_trip = None
        if 'hard_brakes' in trips_df.columns and 'rapid_accels' in trips_df.columns:
            try:
                avg_events_per_trip = (trips_df['hard_brakes'] + trips_df['rapid_accels']).mean()
            except Exception:
                avg_events_per_trip = None

        median_risk_score = None
        max_risk_score = None
        avg_speed_overall = None
        if 'risk_score' in trips_df.columns:
            try:
                median_risk_score = trips_df['risk_score'].median()
                max_risk_score = trips_df['risk_score'].max()
            except Exception:
                pass
        if 'avg_speed_kmh' in trips_df.columns:
            try:
                avg_speed_overall = trips_df['avg_speed_kmh'].mean()
            except Exception:
                avg_speed_overall = None

        kpi_items = [
            ('Total mileage', format_currency(total_mileage) if total_mileage is not None else 'N/A'),
            ('Avg trip duration (min)', f"{avg_trip_duration_min:.1f}" if avg_trip_duration_min is not None else 'N/A'),
            ('Avg monthly trips/driver', f"{avg_monthly_trips_per_driver:.1f}" if avg_monthly_trips_per_driver is not None else 'N/A'),
            ('Unique drivers', unique_drivers),
            ('Median trip dist (km)', f"{median_trip_distance:.1f}" if median_trip_distance is not None else 'N/A'),
            ('% trips at night', f"{percent_night:.1f}%" if percent_night is not None else 'N/A'),
            ('% EV trips', f"{percent_ev:.1f}%" if percent_ev is not None else 'N/A'),
            # add statuses for attention-grabbing KPIs
            ('High-risk trips (%)', f"{percent_high_risk:.1f}%" if percent_high_risk is not None else 'N/A', 'danger' if (percent_high_risk is not None and percent_high_risk > 20) else ('warning' if (percent_high_risk is not None and percent_high_risk > 5) else 'green')),
            ('Fraud alerts (count)', fraud_count if fraud_count is not None else 'N/A', 'danger' if (fraud_count is not None and fraud_count > 5) else ('warning' if (fraud_count is not None and fraud_count > 0) else 'green')),
            ('Fraud alerts (%)', f"{fraud_pct:.1f}%" if fraud_pct is not None else 'N/A', 'danger' if (fraud_pct is not None and fraud_pct > 5) else ('warning' if (fraud_pct is not None and fraud_pct > 0.5) else 'green')),
            ('Avg premium / driver', format_currency(avg_premium_per_driver) if avg_premium_per_driver is not None else 'N/A'),
            ('Median risk score', f"{median_risk_score:.1f}" if median_risk_score is not None else 'N/A'),
            ('Max risk score', f"{max_risk_score:.1f}" if max_risk_score is not None else 'N/A'),
            ('Avg events / trip', f"{avg_events_per_trip:.2f}" if avg_events_per_trip is not None else 'N/A', 'warning' if (avg_events_per_trip is not None and avg_events_per_trip > 1.5) else 'green'),
            ('Avg speed (km/h)', f"{avg_speed_overall:.1f}" if avg_speed_overall is not None else 'N/A')
        ]

        render_kpi_grid(kpi_items, cols=3)

        # Risk insights in a compact expander
        with st.expander('Risk insights (distribution & trend)', expanded=False):
            if 'risk_score' in trips_df.columns:
                try:
                    bins = [0, high_risk_thr, medium_risk_thr, 100]
                    labels = [f'High risk (<{int(high_risk_thr)})', f'Medium ({int(high_risk_thr)}-{int(medium_risk_thr)})', f'Low risk (>{int(medium_risk_thr)})']
                    trips_df['risk_band'] = pd.cut(trips_df['risk_score'], bins=bins, labels=labels, include_lowest=True)
                    band_counts = trips_df['risk_band'].value_counts().reindex(labels).fillna(0)
                    st.markdown('**Risk band distribution**')
                    st.bar_chart(band_counts)
                except Exception:
                    pass

            if 'start_ts' in trips_df.columns and 'risk_score' in trips_df.columns:
                try:
                    trips_df['start_ts_dt'] = pd.to_datetime(trips_df['start_ts'], errors='coerce')
                    daily = trips_df.dropna(subset=['start_ts_dt']).set_index('start_ts_dt').resample('D')['risk_score'].mean()
                    daily_df = daily.reset_index()
                    daily_df['date_str'] = daily_df['start_ts_dt'].dt.strftime('%Y-%m-%d')
                    daily_df = daily_df.set_index('date_str')
                    st.markdown('**Daily avg risk score (recent)**')
                    st.line_chart(daily_df['risk_score'])
                except Exception:
                    pass

        # Driver selector for focused view (no raw table)
        if len(drivers) > 0:
            selected_driver = st.selectbox('Select driver to focus (visuals update)', options=drivers)
            ddf = trips_df[trips_df['driver_id'] == selected_driver]
            render_card('Trips for selected driver', len(ddf))
            if 'risk_score' in ddf.columns and 'start_ts' in ddf.columns:
                try:
                    ddf = ddf.copy()
                    ddf['start_ts_dt'] = pd.to_datetime(ddf['start_ts'], errors='coerce')
                    ddf['date_str'] = ddf['start_ts_dt'].dt.strftime('%Y-%m-%d')
                    recent = ddf.sort_values('start_ts_dt').groupby('date_str')['risk_score'].mean()
                    st.markdown('**Recent risk scores (selected driver)**')
                    st.line_chart(recent)
                except Exception:
                    pass
    else:
        st.info('No data/trips.csv found — save a trip using the Add Driver / Trip form to populate it')

    st.markdown('<div class="section-card"><div class="section-title">Risk & premium demo: before vs after</div></div>', unsafe_allow_html=True)
    if os.path.exists(DEMO_DRIVERS_PATH):
        demo_df = pd.read_csv(DEMO_DRIVERS_PATH)
        rows = []
        for _, row in demo_df.iterrows():
            prob, risk_score = score_trip_with_model(model, row)
            from pricing import compute_premium
            premium = compute_premium(risk_score=risk_score, monthly_mileage=float(row['monthly_mileage']), is_ev=bool(row['is_ev']), safe_driver=bool(row['safe_driver']))
            rows.append({
                'driver_id': row['driver_id'],
                'name': row['name'],
                'scenario': row['scenario'],
                'risk_probability': round(prob, 4),
                'risk_score': risk_score,
                'premium_total': round(float(premium['total_due']), 2),
                'monthly_mileage': row['monthly_mileage'],
            })

        comparison = pd.DataFrame(rows)
        pivot = comparison.pivot_table(index=['driver_id', 'name'], columns='scenario', values=['risk_probability', 'risk_score', 'premium_total'], aggfunc='first').reset_index()
        pivot.columns = [
            col if col in ['driver_id', 'name'] else f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in pivot.columns
        ]
        if 'risk_probability_before' in pivot.columns and 'risk_probability_after' in pivot.columns:
            pivot['delta_risk_probability'] = (pivot['risk_probability_after'] - pivot['risk_probability_before']).round(4)
            pivot['delta_premium_total'] = (pivot['premium_total_after'] - pivot['premium_total_before']).round(2)
        st.dataframe(pivot, use_container_width=True)
    else:
        st.info('No data/demo_drivers.csv found.')

    st.markdown('<div class="section-card"><div class="section-title">Portfolio behavior analytics</div></div>', unsafe_allow_html=True)
    if os.path.exists(SAMPLE_PRED):
        sample_df = pd.read_csv(SAMPLE_PRED)
        # Show risk distribution and premium distribution if available
        risk_col = None
        if 'risk_score' in sample_df.columns:
            risk_col = 'risk_score'
        elif 'risk_probability' in sample_df.columns:
            # convert probability to score
            sample_df['risk_score'] = (1.0 - sample_df['risk_probability']) * 100.0
            risk_col = 'risk_score'
        if risk_col:
            st.markdown('**Risk score distribution**')
            # histogram by bucket
            buckets = pd.cut(sample_df[risk_col].dropna(), bins=10)
            dist = sample_df.groupby(buckets).size()
            st.bar_chart(dist)
            st.markdown('**Risk score over sample**')
            st.line_chart(sample_df[risk_col].fillna(method='ffill').reset_index(drop=True))
        # premium column optional
        if 'premium_total' in sample_df.columns:
            st.markdown('**Premium distribution (sample)**')
            st.bar_chart(sample_df['premium_total'].dropna())
        # Show quick KPIs
        pcols = st.columns(3)
        with pcols[0]:
            render_card('Sample size', len(sample_df))
        with pcols[1]:
            if risk_col:
                render_card('Avg risk score', f"{sample_df[risk_col].mean():.2f}")
            else:
                    render_card('Avg risk score', 'N/A')
        with pcols[2]:
            if 'premium_total' in sample_df.columns:
                render_card('Avg premium', format_currency(sample_df['premium_total'].mean()))
            else:
                render_card('Avg premium', 'N/A')
    else:
        st.info('No sample_predictions.csv found in artifacts/. Run training to generate it.')

    st.markdown('<div class="section-card"><div class="section-title">Interactive driver scoring</div></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        duration_sec = st.slider('Duration (sec)', min_value=60, max_value=3600, value=600)
        distance_km = st.slider('Distance (km)', min_value=0.1, max_value=200.0, value=10.0)
        avg_speed_kmh = st.slider('Avg speed (km/h)', min_value=5.0, max_value=200.0, value=60.0)
    with col2:
        hard_brakes = st.slider('Hard brakes', 0, 10, 0)
        rapid_accels = st.slider('Rapid accels', 0, 10, 0)
        start_hour = st.slider('Start hour', 0, 23, 14)

    df_in = pd.DataFrame([{
        'duration_sec': duration_sec,
        'distance_km': distance_km,
        'avg_speed_kmh': avg_speed_kmh,
        'hard_brakes': hard_brakes,
        'rapid_accels': rapid_accels,
        'start_hour': start_hour,
        'weekday': 1,
        'monthly_mileage': 1200,
        'is_ev': False,
        'safe_driver': False,
    }])
    df_in['speed_per_10km'] = df_in['avg_speed_kmh'] / 10.0
    df_in['brake_accel_sum'] = df_in['hard_brakes'] + df_in['rapid_accels']
    df_in['is_night'] = df_in['start_hour'].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)
    X = df_in[FEATURE_COLS]
    prob = model.predict_proba(X)[0, 1]
    risk_score = round((1.0 - float(prob)) * 100.0, 2)

    a, b = st.columns(2)
    with a:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Risk probability</div><div class="metric-value">{prob:.4f}</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Safety score</div><div class="metric-value">{risk_score:.2f}</div></div>', unsafe_allow_html=True)

    st.write('Feature inputs')
    st.table(X.T)

    st.markdown('<div class="section-card"><div class="section-title">Pricing engine</div></div>', unsafe_allow_html=True)
    monthly_mileage = st.slider('Estimated monthly mileage (km)', min_value=0, max_value=5000, value=1000)
    is_ev = st.checkbox('Electric vehicle', value=False)
    safe_driver = st.checkbox('Safe-driver discount', value=False)

    try:
        from pricing import compute_premium
        from billing import generate_invoice
        from fraud import find_fraud_from_csv, load_config as load_fraud_config

        # Load fraud config and provide quick tuning UI
        cfg = load_fraud_config()
        with st.expander('Fraud detection tuning (thresholds)', expanded=False):
            hs = st.number_input('High speed threshold (km/h)', value=int(cfg.get('high_speed_threshold_kmh', 160)), min_value=50, max_value=400)
            short_dur = st.number_input('Short trip duration (sec)', value=int(cfg.get('short_trip_duration_sec', 120)), min_value=10, max_value=3600)
            short_ev = st.number_input('Short trip events (brakes+accels)', value=int(cfg.get('short_trip_events', 3)), min_value=1, max_value=20)
            rapid_gap = st.number_input('Rapid consecutive gap (sec)', value=int(cfg.get('rapid_consecutive_gap_sec', 60)), min_value=1, max_value=3600)
            rapid_min_dist = st.number_input('Rapid consecutive min distance (km)', value=float(cfg.get('rapid_consecutive_min_distance_km', 1.0)), min_value=0.0, max_value=100.0)
            max_implied = st.number_input('Max implied speed for impossible journey (km/h)', value=int(cfg.get('max_implied_speed_kmh', 220)), min_value=50, max_value=1000)
            # KPI thresholds
            high_risk_threshold = st.number_input('High-risk threshold (risk score)', value=int(cfg.get('high_risk_threshold', 40)), min_value=0, max_value=100)
            medium_risk_threshold = st.number_input('Medium-risk threshold (risk score)', value=int(cfg.get('medium_risk_threshold', 70)), min_value=0, max_value=100)
            night_start = st.number_input('Night hour start (0-23)', value=int(cfg.get('night_start_hour', 22)), min_value=0, max_value=23)
            night_end = st.number_input('Night hour end (0-23)', value=int(cfg.get('night_end_hour', 5)), min_value=0, max_value=23)
            if st.button('Save fraud tuning'):
                # write back to fraud_config.json
                try:
                    new_cfg = {
                        'high_speed_threshold_kmh': int(hs),
                        'short_trip_duration_sec': int(short_dur),
                        'short_trip_events': int(short_ev),
                        'rapid_consecutive_gap_sec': int(rapid_gap),
                        'rapid_consecutive_min_distance_km': float(rapid_min_dist),
                        'max_implied_speed_kmh': int(max_implied),
                        'high_risk_threshold': int(high_risk_threshold),
                        'medium_risk_threshold': int(medium_risk_threshold),
                        'night_start_hour': int(night_start),
                        'night_end_hour': int(night_end)
                    }
                    cfg_path = os.path.join(os.path.dirname(__file__), 'fraud_config.json')
                    with open(cfg_path, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(new_cfg, f, indent=2)
                    st.success('Fraud config saved')
                except Exception as _e:
                    st.error('Error saving config: ' + str(_e))

        breakdown = compute_premium(risk_score=risk_score, monthly_mileage=monthly_mileage, is_ev=is_ev, safe_driver=safe_driver)
        premium_col1, premium_col2 = st.columns(2)
        with premium_col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Premium before tax</div><div class="metric-value">₹{breakdown["premium_before_tax"]:,.2f}</div></div>', unsafe_allow_html=True)
        with premium_col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total due</div><div class="metric-value">₹{breakdown["total_due"]:,.2f}</div></div>', unsafe_allow_html=True)

        # present breakdown as polished cards inside an expander
        with st.expander('Premium breakdown (details)', expanded=False):
            col_left, col_right = st.columns(2)
            with col_left:
                render_card('Base premium', format_currency(breakdown.get('base_premium', 0)))
                render_card('Risk adjustment', format_pct(breakdown.get('risk_adj_pct', 0)))
                render_card('Mileage adjustment', format_pct(breakdown.get('mileage_adj_pct', 0)))
                render_card('EV adjustment', format_pct(breakdown.get('ev_adj_pct', 0)))
            with col_right:
                render_card('Safe driver adjustment', format_pct(breakdown.get('safe_adj_pct', 0)))
                render_card('Premium before tax', format_currency(breakdown.get('premium_before_tax', 0)))
                render_card('Tax', f"{format_pct(breakdown.get('tax_pct', 0))}  ({format_currency(breakdown.get('tax', 0))})")
        st.metric('Total premium due (incl. tax)', format_currency(breakdown.get('total_due', 0)))

        st.markdown('<div class="section-card"><div class="section-title">Invoice generation</div></div>', unsafe_allow_html=True)
        driver_id_input = st.text_input('Driver ID for invoice', value='driver_001')
        period_input = st.text_input('Invoice period (YYYY-MM)', value='2026-08')
        if st.button('Generate invoice', use_container_width=True):
            inv = generate_invoice(driver_id=driver_id_input, period=period_input, premium_breakdown=breakdown)
            st.success(f"Invoice {inv['invoice_id']} generated successfully")
            try:
                st.markdown(f"**Invoice ID:** {inv.get('invoice_id')}  \n**Total Due:** ₹{inv.get('total_due')}  \n**Due date:** {inv.get('due_date')}")
            except Exception:
                pass
            pdf_path = inv.get('pdf_path')
            if pdf_path and os.path.exists(pdf_path):
                try:
                    with open(pdf_path, 'rb') as pf:
                        pdf_bytes = pf.read()
                    st.download_button('Download invoice PDF', data=pdf_bytes, file_name=os.path.basename(pdf_path), mime='application/pdf')
                    b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                    pdf_display = f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='600px'></iframe>"
                    components.html(pdf_display, height=650)
                except Exception as _e:
                    st.error('Failed to read or display PDF: ' + str(_e))
            elif inv.get('pdf_error'):
                st.warning(inv.get('pdf_error'))

        st.markdown('<div class="section-card"><div class="section-title">Fraud monitoring</div></div>', unsafe_allow_html=True)
        if st.button('Run fraud scan', use_container_width=True):
            trips_path = os.path.join('data', 'trips.csv')
            if os.path.exists(trips_path):
                alerts = find_fraud_from_csv(trips_path)
                st.write(f'Found {len(alerts)} alerts')
                if len(alerts) > 0:
                    try:
                        df_alerts = pd.DataFrame(alerts)
                        # show a concise table
                        display_cols = ['trip_id','driver_id','fraud_score','severity']
                        st.dataframe(df_alerts[display_cols].sort_values('fraud_score', ascending=False).reset_index(drop=True))
                        # expand first 10 alerts with reasons
                        for a in alerts[:10]:
                            with st.expander(f"Alert: {a['trip_id']} ({a['severity'].upper()} - {a['fraud_score']})"):
                                st.json(a)
                    except Exception:
                        st.table(alerts[:20])
                else:
                    st.markdown('<span class="badge">No suspicious activity detected</span>', unsafe_allow_html=True)
            else:
                st.info('No data/trips.csv found')
    except Exception as e:
        st.error('Pricing engine error: ' + str(e))
