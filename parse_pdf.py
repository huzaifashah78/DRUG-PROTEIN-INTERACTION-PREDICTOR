import pdfplumber
import pandas as pd
import re
import os

print("📄 Parsing Interactions_Report.pdf...")

data = []
# Ensure the PDF is in the same folder as this script
try:
    with pdfplumber.open("Interactions_Report.pdf") as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    # Skip empty rows or the header
                    if not row or row[0] == "Drug ID":
                        continue
                    
                    # Clean up weird PDF OCR spacing and newlines
                    drug = str(row[0]).replace('\n', '').strip().replace(' ', '_')
                    prot = str(row[1]).replace('\n', '').strip().replace(' ', '_')
                    res = str(row[2]).replace('\n', '').strip().upper()
                    
                    # Fix common PDF typographical artifacts (e.g., 'VO' instead of 'v0')
                    prot = prot.replace('_VO', '_v0').replace('_V0', '_v0')
                    prot = re.sub(r'_+', '_', prot) # Fix accidental double underscores
                    drug = re.sub(r'_+', '_', drug)
                    
                    # Convert BINDING to 1 and NO BIND to 0
                    label = 1 if "BINDING" in res and "NO" not in res else 0
                    data.append({"drug_id": drug, "protein_id": prot, "label": label})

    # Save over the old synthetic interactions file
    df = pd.DataFrame(data)
    df.to_csv("data/interactions.csv", index=False)
    print(f"✅ Extracted {len(df)} interactions from the PDF and saved to data/interactions.csv!")
    print("🚀 You are now ready to run train_pipeline.py!")

except FileNotFoundError:
    print("❌ ERROR: Could not find 'Interactions_Report.pdf'. Make sure it is in the same folder as this script.")