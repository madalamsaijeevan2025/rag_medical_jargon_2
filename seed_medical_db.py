import sqlite3

DB_FILE = "medical_terms.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            definition TEXT NOT NULL,
            clinical_significance TEXT
        )
    """)

    # Create case-insensitive index for lightning-fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_term_lower 
        ON medical_terms (LOWER(term))
    """)

    # Explanation feedback (Goal #6: user satisfaction + re-explain loop)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS explanation_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT NOT NULL,
            rating TEXT NOT NULL CHECK (rating IN ('good','fair','poor')),
            comment TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # Sample dataset of core medical terminologies & abbreviations
    terms_data = [
        # --- LAB MARKERS & ABBREVIATIONS ---
        ("eGFR", "Lab Marker", "Estimated Glomerular Filtration Rate. Measures how well the kidneys filter waste from blood.", "Values below 60 mL/min indicate impaired renal function or Chronic Kidney Disease."),
        ("BUN", "Lab Marker", "Blood Urea Nitrogen. Measures the amount of urea nitrogen in the blood.", "Elevated levels suggest kidney dysfunction, dehydration, or high protein breakdown."),
        ("HbA1c", "Lab Marker", "Glycated Hemoglobin. Measures average blood glucose levels over the preceding 2–3 months.", "Values >= 6.5% indicate Diabetes Mellitus; 5.7%–6.4% indicates Prediabetes."),
        ("ALT", "Lab Marker", "Alanine Aminotransferase. An enzyme found primarily in liver cells.", "Elevated levels signal hepatic inflammation or cell damage (e.g., fatty liver, hepatitis)."),
        ("AST", "Lab Marker", "Aspartate Aminotransferase. An enzyme present in liver, heart, and muscle tissue.", "Elevated alongside ALT points toward liver injury; isolated elevation may suggest muscle injury."),
        ("TSH", "Lab Marker", "Thyroid-Stimulating Hormone. Produced by the pituitary gland to regulate thyroid activity.", "High TSH suggests Hypothyroidism; low TSH suggests Hyperthyroidism."),
        ("CRP", "Lab Marker", "C-Reactive Protein. A protein produced by the liver in response to systemic inflammation.", "Elevated in active infections, autoimmune flares, or cardiovascular inflammation."),
        ("ESR", "Lab Marker", "Erythrocyte Sedimentation Rate. Measures how quickly red blood cells settle at the bottom of a test tube.", "Non-specific marker for systemic inflammation or chronic disease."),
        ("Troponin-I", "Lab Marker", "A cardiac structural protein released into the bloodstream during cardiac muscle damage.", "Key marker for myocardial infarction (heart attack). Any elevation warrants urgent cardiac evaluation."),
        ("WBC", "Lab Marker", "White Blood Count. Total number of leukocytes in a blood sample.", "High (Leukocytosis) suggests infection/inflammation; Low (Leukopenia) suggests bone marrow suppression."),
        ("RBC", "Lab Marker", "Red Blood Count. Total number of erythrocytes carrying oxygen throughout the body.", "Low levels indicate anemia; high levels (Polycythemia) indicate oxygen deprivation or marrow issues."),
        ("Platelets", "Lab Marker", "Cell fragments crucial for blood clotting and hemostasis.", "Low (Thrombocytopenia) increases bleeding risk; High (Thrombocytosis) increases clotting risk."),
        ("MCV", "Lab Marker", "Mean Corpuscular Volume. Measures the average physical size of red blood cells.", "Low MCV indicates Microcytic Anemia (e.g., iron deficiency); High MCV indicates Macrocytic Anemia."),
        ("Ferritin", "Lab Marker", "A blood protein that stores iron inside cells.", "Low levels are the most specific indicator of iron deficiency anemia."),
        ("Serum Sodium", "Lab Marker", "Primary extracellular electrolyte regulating fluid balance and neurological function.", "High (Hypernatremia) or Low (Hyponatremia) can cause confusion, seizures, or edema."),
        ("Serum Potassium", "Lab Marker", "Essential intracellular electrolyte regulating cardiac and neuromuscular action potentials.", "High (Hyperkalemia) or Low (Hypokalemia) can trigger dangerous cardiac arrhythmias."),
        ("Serum Calcium", "Lab Marker", "Mineral critical for bone health, nerve transmission, and muscle contraction.", "Elevated (Hypercalcemia) may indicate hyperparathyroidism or malignancy."),

        # --- RADIOLOGY & PATHOLOGY JARGON ---
        ("Anterolisthesis", "Radiology", "Forward displacement of a vertebral body relative to the vertebra below it.", "Can cause spinal stenosis or nerve root compression, resulting in lower back/leg pain."),
        ("Stenosis", "Radiology/Pathology", "Abnormal narrowing of a tubular structure, passage, or vascular lumen.", "In the spine, leads to nerve pinch; in blood vessels, restricts arterial flow."),
        ("Atelectasis", "Radiology", "Partial or complete collapse of a lung lobe or alveoli.", "Common post-surgery or in bedridden patients due to shallow breathing or mucous plugs."),
        ("Cardiomegaly", "Radiology", "Enlargement of the heart silhouette observed on chest X-ray or CT.", "Often secondary to chronic hypertension, heart failure, or cardiomyopathy."),
        ("Lymphadenopathy", "Pathology", "Abnormal enlargement or swelling of lymph nodes.", "Can indicate localized infection, systemic immune reaction, or hematologic malignancy."),
        ("Effusion", "Radiology", "An abnormal accumulation of fluid in an anatomical cavity (e.g., pleural, pericardial, or joint).", "Pleural effusion restricts lung expansion; pericardial effusion can cause cardiac tamponade."),
        ("Ischemia", "Pathology", "Inadequate blood supply to a tissue or organ causing localized oxygen deprivation.", "If uncorrected, leads to necrosis (tissue death) or infarction."),
        ("Infarction", "Pathology", "Tissue necrosis resulting from obstruction of local blood supply.", "Commonly seen in acute myocardial infarction (heart attack) or ischemic stroke."),
        ("Foraminal Stenosis", "Radiology", "Narrowing of the neural foramina where nerve roots exit the spinal column.", "Causes radiculopathy (shooting pain, numbness, or weakness in limbs)."),
        ("Steatosis", "Pathology", "Abnormal retention of lipids within parenchymal cells, typically in the liver.", "Hepatic steatosis is commonly referred to as fatty liver disease."),

        # --- CLINICAL CONDITIONS & DISEASES ---
        ("Hypertension", "Condition", "Chronically elevated systemic arterial blood pressure (>= 130/80 mmHg).", "Major risk factor for stroke, coronary artery disease, and chronic kidney disease."),
        ("Type 2 Diabetes", "Condition", "Metabolic disorder characterized by peripheral insulin resistance and hyperglycemia.", "Leads to long-term microvascular and macrovascular complications if unmanaged."),
        ("Hypernatremia", "Condition", "Elevated sodium concentration in the blood (>145 mmol/L).", "Usually caused by severe dehydration, diabetes insipidus, or excessive sodium intake."),
        ("Hyponatremia", "Condition", "Abnormally low sodium concentration in the blood (<135 mmol/L).", "Can cause cellular swelling, cerebral edema, lethargy, and neurological signs."),
        ("Chronic Kidney Disease", "Condition", "Gradual loss of kidney function over months or years (eGFR < 60 for >3 months).", "Requires blood pressure control, metabolic monitoring, and eventual renal replacement if advanced."),
        ("Hyperlipidemia", "Condition", "Elevated levels of lipids (cholesterol and/or triglycerides) in the blood.", "Accelerates atherosclerosis and increases cardiac event risk."),
        ("Anemia", "Condition", "Deficiency in red blood cells or hemoglobin leading to reduced oxygen delivery to tissues.", "Causes fatigue, pallor, and shortness of breath; underlying cause must be identified."),
        ("GERD", "Condition", "Gastroesophageal Reflux Disease. Chronic stomach acid flow back into the esophagus.", "Causes pyrosis (heartburn) and can lead to Barrett's esophagus if untreated."),
        ("COPD", "Condition", "Chronic Obstructive Pulmonary Disease. Progressive lung disease obstructing airflow.", "Characterized by persistent cough, sputum production, and progressive dyspnea."),
        ("Osteoarthritis", "Condition", "Degenerative joint disease caused by breakdown of joint cartilage and underlying bone.", "Leads to joint pain, stiffness, and reduced range of motion, typically in weight-bearing joints.")
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO medical_terms (term, category, definition, clinical_significance)
        VALUES (?, ?, ?, ?)
    """, terms_data)

    conn.commit()
    print(f"Successfully initialized '{DB_FILE}' with {len(terms_data)} seed records.")
    conn.close()

if __name__ == "__main__":
    init_db()