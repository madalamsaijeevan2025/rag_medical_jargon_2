import sqlite3
from config import get_settings

settings = get_settings()
DB_PATH = settings.dict_db_path

def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check current columns
    cursor.execute("PRAGMA table_info(medical_terms)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Current columns: {columns}")
    
    # Add new columns if they don't exist
    new_columns = [
        ("disease_category", "TEXT"),
        ("etiology", "TEXT"),
        ("symptoms", "TEXT"),
        ("related_tests", "TEXT"),  # JSON array
        ("treatment_approach", "TEXT"),
        ("icd10_code", "TEXT"),
        ("severity", "TEXT"),  # mild, moderate, severe, critical
        ("body_system", "TEXT"),  # cardiovascular, endocrine, pulmonary, etc.
    ]
    
    for col_name, col_type in new_columns:
        if col_name not in columns:
            try:
                cursor.execute(f"ALTER TABLE medical_terms ADD COLUMN {col_name} {col_type}")
                print(f"Added column: {col_name}")
            except sqlite3.OperationalError as e:
                print(f"Column {col_name} may already exist: {e}")
        else:
            print(f"Column {col_name} already exists")
    
    # Create indexes for new columns
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_disease_category ON medical_terms (disease_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_body_system ON medical_terms (body_system)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON medical_terms (severity)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_icd10 ON medical_terms (icd10_code)")
    
    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    migrate()