# UBI-Engine — AI-Powered Usage-Based Insurance PoC

## Overview

UBI-Engine is a Proof of Concept for **Usage-Based Insurance (UBI)** that demonstrates how vehicle and driving-behavior data can be used to assess risk, detect suspicious activity, calculate personalized insurance premiums, and generate billing outputs.

Traditional motor insurance often relies on relatively static pricing models that may not adequately differentiate drivers based on their actual driving behavior. This can contribute to pricing leakage, adverse selection, limited personalization, and dissatisfaction among lower-risk customers.

This PoC demonstrates an alternative approach:

**Driving behavior → AI risk assessment → fraud detection → personalized pricing → billing**

The solution is designed as a foundation for future UBI scenarios involving connected vehicles, EVs, OEM insurance programs, fleets, and mobility platforms.

---

## Business Problem

Motor insurance pricing can be improved when insurers have better visibility into how a vehicle is actually being driven.

Key challenges include:

* Limited use of individual driving behavior in pricing decisions
* Difficulty differentiating lower-risk and higher-risk drivers
* Potential pricing leakage from broad/static risk assumptions
* Fraud and suspicious journey patterns
* Limited transparency into how driving behavior affects premiums
* Increasing availability of connected-vehicle and telematics data

A UBI approach allows insurers to use behavioral and usage information to support more risk-aligned pricing.

---

## PoC Objectives

This project demonstrates how an insurer could:

1. Generate or ingest telematics-style trip data
2. Engineer behavioral driving features
3. Apply an ML model to estimate driving risk
4. Detect suspicious or potentially fraudulent trip patterns
5. Convert risk and usage information into a personalized premium
6. Apply EV and safe-driver pricing adjustments
7. Generate an invoice/billing output
8. Expose the capabilities through APIs and a dashboard

---

## Solution Architecture

```text
              Vehicle / Telematics Data
                        │
                        ▼
              ┌─────────────────────┐
              │  Trip Data Layer    │
              │  Simulator / CSV    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Feature Engineering │
              │ Speed / Distance    │
              │ Braking / Accel.    │
              │ Time / Usage        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    ML Risk Engine   │
              │ Driving Risk Score  │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
     ┌─────────────────┐   ┌─────────────────┐
     │  Fraud Engine   │   │ Pricing Engine  │
     │                 │   │                 │
     │ High Speed      │   │ Risk Adjustment │
     │ Aggressive Trip │   │ Mileage         │
     │ Rapid Trips     │   │ EV Discount     │
     │ Impossible      │   │ Safe Driver     │
     │ Journey         │   │                 │
     └────────┬────────┘   └────────┬────────┘
              │                     │
              └──────────┬──────────┘
                         ▼
                ┌──────────────────┐
                │ Billing / Invoice│
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ API / Dashboard  │
                └──────────────────┘
```

---

## Key Capabilities

### 1. Telematics-Style Data Generation

The PoC includes a synthetic trip-data simulator representing connected-vehicle information such as:

* Driver ID
* Vehicle/device ID
* Trip ID
* Trip duration
* Distance
* Average speed
* Hard braking
* Rapid acceleration
* Start time and weekday
* Geographic coordinates
* Driving risk label

This allows the complete UBI workflow to be demonstrated without requiring a live vehicle data source.

---

### 2. Driving Behavior Feature Engineering

Trip data is transformed into model-ready behavioral features.

Current derived features include:

* Speed normalization
* Combined braking and acceleration events
* Night-driving indicator
* Trip duration
* Distance
* Average speed
* Hard braking
* Rapid acceleration
* Start hour
* Weekday

These features are used by the ML scoring layer.

---

### 3. AI/ML Risk Scoring

The PoC uses a trained ML model to estimate the probability associated with the trip's risk classification.

The scoring API accepts trip information and returns a probability for each trip.

For pricing, the model probability is converted into a 0–100 risk score where a higher score represents lower risk behavior.

This enables driving behavior to become an input into insurance pricing rather than relying solely on static assumptions.

---

### 4. Fraud and Anomaly Detection

The PoC includes configurable fraud-detection rules.

Current rules include:

* **High-speed detection**
* **Short aggressive trips**
* **Rapid consecutive trips**
* **Impossible journey detection**

The impossible-journey detector uses GPS coordinates and timestamps to calculate the distance between consecutive journeys and estimate the implied travel speed.

An alert is generated when the implied speed exceeds the configured threshold.

---

### 5. Usage-Based Pricing

The pricing engine calculates a premium using multiple factors:

```text
Base Premium
     ↓
Risk Adjustment
     ↓
Mileage Adjustment
     ↓
EV Adjustment
     ↓
Safe-Driver Adjustment
     ↓
Tax
     ↓
Final Premium
```

The pricing engine returns a transparent breakdown so the impact of each adjustment can be demonstrated.

---

### 6. EV and Safe-Driver Differentiation

The PoC supports pricing adjustments based on:

* Electric vehicle status
* Safe-driver eligibility
* Monthly mileage
* Driving risk score

This demonstrates how UBI pricing can incorporate both behavioral and vehicle/usage characteristics.

---

### 7. Billing

The solution includes an API endpoint for generating an invoice from the calculated premium.

The billing flow can consume either:

* A precomputed risk score, or
* Trip data that is scored by the ML model before premium calculation

This demonstrates the end-to-end concept from driving behavior to a billable insurance amount.

---

### 8. API Layer

The PoC exposes the core capabilities through FastAPI endpoints.

Examples include:

```text
GET  /health

POST /score

POST /pricing/calculate

POST /pricing/from_trips

POST /billing/generate

GET  /fraud-alerts
```

This provides a simple integration point for a future insurer, OEM, fleet, or mobility application.

---

## End-to-End Business Flow

```text
Driver uses vehicle
        ↓
Trip / telematics data generated
        ↓
Driving behavior features calculated
        ↓
ML model evaluates risk
        ↓
Fraud rules identify suspicious patterns
        ↓
Driver risk score determined
        ↓
UBI premium calculated
        ↓
Premium breakdown generated
        ↓
Invoice / billing output
```

### Example

A safer driver with:

* Lower-risk driving behavior
* Fewer hard-braking events
* Lower aggressive-driving activity
* Lower monthly mileage
* EV vehicle
* Safe-driver qualification

can receive more favorable pricing.

A higher-risk driver with:

* Excessive speed
* Aggressive driving
* Higher mileage
* Suspicious journey patterns

can receive a higher risk-adjusted premium and/or fraud alert.

---

## Business Value

The PoC demonstrates the potential to support:

### More risk-aligned pricing

Use actual driving behavior and usage information to differentiate customers.

### Better customer experience

Provide customers with a more transparent connection between their behavior and their premium.

### Fraud reduction

Identify suspicious driving and journey patterns using behavioral and geographic signals.

### Improved operational efficiency

Automate risk scoring, fraud checks, pricing calculations and billing preparation.

### New UBI distribution opportunities

The architecture provides a foundation for future integrations with:

* Connected-car platforms
* OEM insurance programs
* Fleet operators
* Mobility platforms
* Commercial vehicle ecosystems

These integrations are **future extensions of the PoC**, not currently implemented integrations.

---

## Technology Stack

* **Python**
* **Pandas**
* **Scikit-learn / ML model**
* **FastAPI**
* **Pydantic**
* **Joblib**
* **Streamlit**
* **JSON-based configurable pricing and fraud rules**
* **Synthetic telematics data**

---

## Project Structure

```text
UBI-Engine/
│
├── api_server.py          # FastAPI scoring, pricing, billing and fraud APIs
├── billing.py             # Billing / invoice generation
├── data_simulator.py      # Synthetic telematics data generation
├── demo_run_fraud.py      # Fraud detection demonstration
├── features.py            # Feature engineering
├── fraud.py               # Fraud and anomaly detection
├── pricing.py             # UBI premium calculation
├── dashboard.py           # Streamlit dashboard
│
├── fraud_config.json      # Fraud detection thresholds
├── pricing_rules.json     # Pricing rules and adjustments
│
├── data/
│   └── ...                # Synthetic trip/demo data
│
├── test_scoring.py        # Scoring tests
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Running the PoC

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic trip data

```bash
python data_simulator.py --output data/trips.csv --n-drivers 50 --n-trips 2000 --seed 42
```

### 3. Run fraud detection demo

```bash
python demo_run_fraud.py data/demo_trips_with_geo.csv
```

The demo includes an impossible-journey scenario to demonstrate the geospatial fraud rule.

### 4. Run the API

```bash
uvicorn api_server:app --reload
```

The API provides health, scoring, pricing, billing and fraud endpoints.

### 5. Run the dashboard

```bash
streamlit run dashboard.py
```

---

## POC Scope and Limitations

This project is intentionally a **Proof of Concept**, not a production insurance platform.

The current implementation uses synthetic telematics data and demonstrates the core business workflow.

Future production capabilities could include:

* Live telematics/OEM integrations
* Real-time event streaming
* Production-grade ML model lifecycle management
* Fleet and commercial-vehicle hierarchies
* Policy administration integration
* Payment processing
* Production authentication and authorization
* Multi-tenant insurer architecture
* Actuarial model calibration using real claims data
* Regulatory and compliance controls

These are outside the scope of the current capstone PoC.

---

## GitHub Copilot and AI-Assisted Development

GitHub Copilot was used as an AI-assisted development tool during the implementation lifecycle.

Relevant activities include:

* Code generation and implementation assistance
* Feature engineering support
* API development
* Debugging and refactoring
* Test development
* Documentation assistance

The final implementation was reviewed and adapted as part of the development process.

For the capstone submission, the specific prompts and examples used during development should be included separately as supporting evidence.

---

## Key Takeaway

The UBI-Engine PoC demonstrates how insurers can move from:

**Static premium assumptions**

to:

**Data-driven, behavior-aware insurance pricing**

by connecting:

**Telematics → AI Risk → Fraud Detection → UBI Pricing → Billing**

The goal is not to build a complete insurance platform, but to demonstrate the business feasibility and value of an AI-assisted UBI operating model.
