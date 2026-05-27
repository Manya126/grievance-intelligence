"""
synthetic_generator.py
-----------------------
Generates realistic synthetic grievance data using templates.
Use this to:
  - Bootstrap the project on Day 1 before API keys are ready
  - Test your pipeline without hitting API rate limits
  - Supplement real data when quota runs out

The output schema is identical to real data so your full pipeline
works without any changes.

Run:   python synthetic_generator.py --records 2000
"""

import random
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from db_writer import save_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Domain data ───────────────────────────────────────────────────────────────
DEPARTMENTS = [
    "Ministry of Railways", "Ministry of Finance", "Ministry of Health",
    "Ministry of Education", "Ministry of Agriculture", "Ministry of Home Affairs",
    "Department of Posts", "Ministry of Petroleum", "Ministry of Labour",
    "Ministry of Housing", "Ministry of Water Resources", "Ministry of Power",
    "Ministry of Telecommunications", "Ministry of Transport", "Ministry of Commerce",
]

STATES = [
    "Uttar Pradesh", "Maharashtra", "Bihar", "West Bengal", "Madhya Pradesh",
    "Rajasthan", "Tamil Nadu", "Karnataka", "Gujarat", "Andhra Pradesh",
    "Odisha", "Telangana", "Punjab", "Haryana", "Delhi", "Assam",
    "Jharkhand", "Chhattisgarh", "Kerala", "Uttarakhand",
]

STATUS_OPTIONS  = ["pending", "in_progress", "resolved", "closed", "escalated"]
STATUS_WEIGHTS  = [0.30, 0.20, 0.35, 0.10, 0.05]

FEEDBACK_RATINGS = ["Excellent", "Good", "Average", "Poor", ""]
FEEDBACK_WEIGHTS = [0.15, 0.25, 0.20, 0.30, 0.10]

SOURCES = ["datagov_cpgrams", "twitter", "reddit", "cpgrams_portal"]

# Grievance templates per department

GRIEVANCE_TEMPLATES = {

    "Ministry of Railways": [
        "Train {train} from {city1} to {city2} was delayed by {hours} hours without any announcement, causing inconvenience to passengers and senior citizens.",
        
        "Refund of Rs {amount} for cancelled ticket {pnr} has still not been received after {days} days despite multiple complaints on the railway portal.",

        "Station {station} has no drinking water facilities on platform {platform}, creating serious problems during summer travel.",

        "TC on train {train} demanded extra money of Rs {amount} for seat upgrade despite confirmed reservation status.",

        "My luggage worth Rs {amount} was stolen from coach {coach} on train {train} and no proper action has been taken by railway authorities."
    ],

    "Ministry of Finance": [
        "Income tax refund of Rs {amount} for AY {year} has not been credited even after {days} days, causing financial hardship.",

        "GST input credit of Rs {amount} has been blocked for {months} months without any clarification from the department.",

        "PAN card application status has shown processing for more than {days} days and customer support is not responding.",

        "Bank account was frozen without prior notice and an amount of Rs {amount} remains inaccessible for daily expenses.",

        "EPF withdrawal request for Rs {amount} has been pending for {months} months despite repeated reminders on the portal."
    ],

    "Ministry of Health": [
        "Ayushman Bharat card was rejected at {hospital} hospital despite valid eligibility and approved registration.",

        "Generic medicines are regularly unavailable at Jan Aushadhi store in {city}, forcing patients to buy expensive alternatives.",

        "Ambulance service did not respond in {area} for {hours} hours during a medical emergency involving an elderly patient.",

        "Doctor at {hospital} demanded Rs {amount} cash for treatment that should have been completely free under the government scheme.",

        "Vaccination certificate has not been generated even after {days} days, causing issues for travel and office verification."
    ],

    "Ministry of Education": [
        "Scholarship amount of Rs {amount} has not been disbursed for the last {months} months despite successful application approval.",

        "Correction request in school certificate has been pending for more than {days} days at the education board office.",

        "Mid-day meal has not been provided at government school in {area} for {days} days, affecting students from poor families.",

        "College admission process was blocked because of repeated technical issues on the online portal during final submission.",

        "Teacher has remained absent from school in {area} for the past {days} days and students are suffering academically."
    ],

    "Ministry of Agriculture": [
        "PM-KISAN installment of Rs {amount} has not been received for {months} months despite verification being completed successfully.",

        "Crop insurance claim worth Rs {amount} was rejected without any valid inspection or explanation from officials.",

        "Fertiliser is not available in {area} at government-approved rates and farmers are forced to buy costly private stock.",

        "Kisan credit card application has remained pending for over {days} days with no update from the agriculture office.",

        "Irrigation canal in {area} has remained blocked for {days} days, damaging nearly {acres} acres of crops."
    ],

    "Ministry of Telecommunications": [
        "Mobile network services have been unavailable in {area} for {days} days, affecting online classes and digital payments.",

        "Broadband speed remains far below promised {mbps} Mbps despite repeated complaints and full payment of monthly charges.",

        "SIM card was blocked without prior notice even though the number has active recharge and valid KYC verification.",

        "Excess bill of Rs {amount} was charged in {month} without any usage explanation from the telecom provider.",

        "Mobile number portability request has been rejected multiple times without any valid reason or customer support assistance."
    ],

    "Ministry of Power": [
        "There has been no electricity supply in {area} for {hours} hours due to an unresolved transformer fault.",

        "Electricity bill of Rs {amount} for {month} appears heavily overcharged despite normal household usage.",

        "Application for new electricity connection has been pending for more than {days} days without field inspection.",

        "Meter reading has not been conducted for {months} months and estimated bills are continuously being generated.",

        "Transformer failure in {area} has not been repaired for {days} days despite repeated complaints from residents."
    ],

    "Ministry of Housing": [
        "PM Awas Yojana house was allotted but possession has not been provided even after waiting for {months} months.",

        "Construction quality of PMAY housing project in {area} is extremely poor and walls have already started cracking.",

        "Housing loan subsidy of Rs {amount} has not been credited despite loan approval and document verification.",

        "Property registration documents have still not been issued after full payment and repeated office visits.",

        "Eviction notice was issued despite valid ownership documents being submitted to the housing department."
    ],
}



# Default templates for departments not listed above
DEFAULT_TEMPLATES = [
    "Application submitted {days} days ago with no response from {department}.",
    "Service not provided despite Rs {amount} fee paid to {department}.",
    "Document submitted to {department} not processed for {days} days.",
    "Complaint registered {months} months ago, no resolution yet from {department}.",
    "Office of {department} closed during working hours repeatedly.",
]


# Fill-in values for template placeholders
PLACEHOLDER_VALUES = {
    "train": [
        "12301",
        "22691",
        "Rajdhani",
        "Shatabdi",
        "Vande Bharat 2056"
    ],

    "city1": STATES,
    "city2": STATES,

    "station": [
        "New Delhi",
        "Mumbai CST",
        "Howrah",
        "Chennai Central",
        "Bangalore City"
    ],

    "platform": [
        "1", "2", "3", "4", "5"
    ],

    "coach": [
        "S4", "B2", "A1", "S7", "2A"
    ],

    "hospital": [
        "District Hospital",
        "AIIMS",
        "Civil Hospital",
        "PHC"
    ],

    "area": [
        "Ward 12",
        "Sector 4",
        "Block B",
        "Village Rampur",
        "Colony C"
    ],

    "city": STATES,

    "month": [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ],

    "year": [
        "2022-23",
        "2023-24",
        "2024-25"
    ],

    "mbps": [
        "50",
        "100",
        "200"
    ],
}
# ──────────────────────────────────────────────────────────────────────────────


def fill_template(template: str, department: str) -> str:
    

    dynamic_values = {

        # Dynamic values generated EVERY complaint
        "pnr": str(random.randint(2000000000, 9999999999)),
        "acres": str(random.randint(1, 50)),
        "hours": str(random.randint(2, 72)),
        "days": str(random.randint(7, 180)),
        "months": str(random.randint(2, 18)),
        "amount": str(random.randint(500, 200000)),

        # Random categorical selections
        "train": random.choice(PLACEHOLDER_VALUES["train"]),
        "city1": random.choice(PLACEHOLDER_VALUES["city1"]),
        "city2": random.choice(PLACEHOLDER_VALUES["city2"]),
        "station": random.choice(PLACEHOLDER_VALUES["station"]),
        "platform": random.choice(PLACEHOLDER_VALUES["platform"]),
        "coach": random.choice(PLACEHOLDER_VALUES["coach"]),
        "hospital": random.choice(PLACEHOLDER_VALUES["hospital"]),
        "area": random.choice(PLACEHOLDER_VALUES["area"]),
        "city": random.choice(PLACEHOLDER_VALUES["city"]),
        "month": random.choice(PLACEHOLDER_VALUES["month"]),
        "year": random.choice(PLACEHOLDER_VALUES["year"]),
        "mbps": random.choice(PLACEHOLDER_VALUES["mbps"]),
    }

    text = template.replace("{department}", department)

    for key, value in dynamic_values.items():

        placeholder = "{" + key + "}"

        if placeholder in text:
            text = text.replace(placeholder, value)

    return text



def random_date(start_days_ago: int = 365, end_days_ago: int = 0) -> datetime:
    """Random datetime within a window."""
    delta = random.randint(end_days_ago, start_days_ago)
    return datetime.now() - timedelta(days=delta)


def generate_record(i: int) -> dict:
    """Generate one synthetic grievance record."""
    department = random.choice(DEPARTMENTS)
    status = random.choices(STATUS_OPTIONS, weights=STATUS_WEIGHTS)[0]

    # Get department-specific template or default
    templates = GRIEVANCE_TEMPLATES.get(department, DEFAULT_TEMPLATES)
    template = random.choice(templates)
    text = fill_template(template, department)

    date_filed = random_date(start_days_ago=365)

    # Resolution date only if resolved/closed
    date_resolved = ""
    if status in ("resolved", "closed"):
        resolution_lag = random.randint(3, 90)
        date_resolved = (date_filed + timedelta(days=resolution_lag)).isoformat()

    feedback = ""
    if status == "resolved":
        feedback = random.choices(FEEDBACK_RATINGS, weights=FEEDBACK_WEIGHTS)[0]

    return {
        "grievance_id":    f"SYN{i:07d}",
        "text":            text,
        "department":      department,
        "state":           random.choice(STATES),
        "status":          status,
        "date_filed":      date_filed.isoformat(),
        "date_resolved":   date_resolved,
        "feedback_rating": feedback,
        "source":          "synthetic",
        "scraped_at":      datetime.now().isoformat(),
    }


def run(n_records: int = 2000, save_csv: bool = True, save_db: bool = True) -> pd.DataFrame:
    """Generate n_records synthetic grievances, save CSV and/or DB."""
    logger.info(f"Generating {n_records} synthetic grievance records...")
    records = [generate_record(i) for i in range(n_records)]
    df = pd.DataFrame(records)

    if save_csv:
        import os
        os.makedirs("data/raw", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = f"data/raw/synthetic_{timestamp}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Saved {len(df)} records to {path}")

    if save_db:
        save_to_db(df)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic grievance data")
    parser.add_argument("--records", type=int, default=2000, help="Number of records to generate")
    parser.add_argument("--no-db",   action="store_true", help="Skip saving to SQLite")
    args = parser.parse_args()

    df = run(n_records=args.records, save_db=not args.no_db)

    print(f"\nDone. Generated {len(df)} records.")
    print(f"\nDepartment distribution:")
    print(df["department"].value_counts().to_string())
    print(f"\nStatus distribution:")
    print(df["status"].value_counts().to_string())
    print(f"\nSample texts:")
    for text in df["text"].sample(3):
        print(f"  • {text}")