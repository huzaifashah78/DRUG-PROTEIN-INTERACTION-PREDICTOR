import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
MODEL_PATH = "dti_model.pkl"
DATA_FOLDER = "data"

print("🚀 STARTING AI TRAINING PIPELINE...")

# 1. READ RAW FILES
print("   Step 1: Reading File Database...")
try:
    df_drugs = pd.read_csv(os.path.join(DATA_FOLDER, "drugs.csv"))
    df_inter = pd.read_csv(os.path.join(DATA_FOLDER, "interactions.csv"))
    
    # Parse FASTA manually
    protein_map = {}
    current_id = None
    seq_lines = []
    with open(os.path.join(DATA_FOLDER, "proteins.fasta"), "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id: protein_map[current_id] = "".join(seq_lines)
                current_id = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if current_id: protein_map[current_id] = "".join(seq_lines)

except FileNotFoundError:
    print("❌ ERROR: Files missing! Run 'setup_data.py' first.")
    exit()

# 2. PREPARE DATASET
print(f"   Step 2: Linking {len(df_inter)} interactions...")
data_rows = []
drug_map = pd.Series(df_drugs.smiles.values, index=df_drugs.drug_id).to_dict()

for _, row in df_inter.iterrows():
    try:
        smiles = drug_map[row['drug_id']]
        seq = protein_map[row['protein_id']]
        label = row['label']
        
        # Weighting: Duplicate Positives to ensure model learns them well
        repeats = 5 if label == 1 else 1
        for _ in range(repeats):
            data_rows.append([smiles, seq, label])
            
    except KeyError:
        continue # Skip missing IDs

df = pd.DataFrame(data_rows, columns=['SMILES', 'Sequence', 'Label'])

# 3. FEATURIZATION
print("   Step 3: Converting Chemistry to Numbers...")
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def encode_drug(s):
    try:
        mol = Chem.MolFromSmiles(s)
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except: return np.zeros(1024)

def encode_prot(s):
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    s = s.upper()
    if not s: return np.zeros(20)
    return np.array([s.count(a)/len(s) for a in aa])

X_d = np.array([encode_drug(x) for x in df['SMILES']])
X_p = np.array([encode_prot(x) for x in df['Sequence']])
X = np.hstack([X_d, X_p])
y = df['Label'].values

# 4. TRAIN
print("   Step 4: Training Random Forest...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ SYSTEM READY. Brain saved to '{MODEL_PATH}'.")