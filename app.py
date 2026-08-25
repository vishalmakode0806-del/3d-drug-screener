import base64
import io
import re
import urllib.parse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski
import requests
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

st.set_page_config(page_title="PubChem 3D AI Screener", layout="wide")

@st.cache_resource
def load_ai_model():
    df = pd.read_csv('https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv')
    df['mw'] = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in df['smiles']]
    df['logp'] = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in df['smiles']]
    df['rot'] = [Descriptors.NumRotatableBonds(Chem.MolFromSmiles(s)) for s in df['smiles']]
    df['aro'] = [sum([1 for a in Chem.MolFromSmiles(s).GetAtoms() if a.GetIsAromatic()])/float(Chem.MolFromSmiles(s).GetNumAtoms()) for s in df['smiles']]
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[['logp', 'mw', 'rot', 'aro']], df['measured log solubility in mols per litre'])
    return rf

rf = load_ai_model()
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def get_formula(m):
    counts = {}
    for atom in m.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
        h_count = atom.GetTotalNumHs()
        if h_count > 0: counts['H'] = counts.get('H', 0) + h_count
    if 'C' in counts:
        order = ['C', 'H'] + sorted([k for k in counts if k not in ('C', 'H')])
    elif 'H' in counts and 'S' in counts and 'O' in counts:
        order = ['H', 'S', 'O'] + sorted([k for k in counts if k not in ('H', 'S', 'O')])
    elif 'H' in counts and 'P' in counts and 'O' in counts:
        order = ['H', 'P', 'O'] + sorted([k for k in counts if k not in ('H', 'P', 'O')])
    else:
        order = sorted(counts.keys())
    res = ''
    for sym in order:
        if sym in counts and counts[sym] > 0:
            num_str = str(counts[sym]).translate(SUB) if counts[sym] > 1 else ''
            res += sym + num_str
    return res

def universal_pubchem_search(query_text):
    q = query_text.strip()
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # 1. Direct SMILES check
    try:
        if Chem.MolFromSmiles(q) and len(q) > 3 and not q.isalnum():
            return q, "Custom SMILES Structure", None
    except:
        pass

    # 2. Query PubChem by Direct Name / Synonym / Formula
    enc_q = urllib.parse.quote(q)
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc_q}/property/CanonicalSMILES,Title,IUPACName/JSON"
        r = requests.get(url, headers=headers, timeout=6).json()
        props = r['PropertyTable']['Properties'][0]
        return props['CanonicalSMILES'], props.get('Title', q.title()), props.get('IUPACName', 'N/A')
    except:
        pass

    # 3. Query PubChem by Fast Formula Match (e.g. C16H13Cl2NO4)
    if re.match(r"^([A-Z][a-z]?\d*)+$", q):
        try:
            url_f = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{enc_q}/property/CanonicalSMILES,Title,IUPACName/JSON"
            rf_res = requests.get(url_f, headers=headers, timeout=6).json()
            props_f = rf_res['PropertyTable']['Properties'][0]
            return props_f['CanonicalSMILES'], props_f.get('Title', f"Formula match ({q})"), props_f.get('IUPACName', 'N/A')
        except:
            pass

    # 4. PubChem Fuzzy Autocomplete (Handles slight spelling errors)
    try:
        auto_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{enc_q}/json?limit=5"
        r_auto = requests.get(auto_url, headers=headers, timeout=5).json()
        terms = r_auto.get('dictionary_terms', {}).get('compound', [])
        for term in terms:
            try:
                url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(term)}/property/CanonicalSMILES,Title,IUPACName/JSON"
                r2 = requests.get(url2, headers=headers, timeout=5).json()
                props2 = r2['PropertyTable']['Properties'][0]
                return props2['CanonicalSMILES'], props2.get('Title', term.title()), props2.get('IUPACName', 'N/A')
            except:
                continue
    except:
        pass

    return None, None, None

def get_3d_html(s):
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        AllChem.EmbedMolecule(m, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(m)
        b = Chem.MolToMolBlock(m)
    except:
        b = Chem.MolToMolBlock(Chem.MolFromSmiles(s))
    bc = b.replace('\n', '\\n').replace('"', '\\"')
    h = f'''<!DOCTYPE html><html><head><script src="https://3Dmol.org/build/3Dmol-min.js"></script></head><body style="margin:0;background:#0d1117;"><div id="v" style="width:100vw;height:480px;"></div><script>
let v = $3Dmol.createViewer(document.getElementById("v"), {{backgroundColor:"#0d1117"}});
v.addModel("{bc}", "sdf");
v.setStyle({{}}, {{stick:{{radius:0.14}}, sphere:{{scale:0.25}}}});
let atoms = v.selectedAtoms({{}});
let colorMap = {{ 'C': '#94a3b8', 'O': '#ef4444', 'N': '#3b82f6', 'H': '#e2e8f0', 'S': '#eab308', 'F': '#22c55e', 'Cl': '#10b981' }};
let fontColorMap = {{ 'H': '#000000', 'C': '#000000', 'S': '#000000' }};
for (let i = 0; i < atoms.length; i++) {{
    let elem = atoms[i].elem;
    let bg = colorMap[elem] || '#64748b';
    let fc = fontColorMap[elem] || '#ffffff';
    let lbl = elem === 'O' ? 'O (Acceptor)' : elem === 'N' ? 'N (Donor/Acc)' : elem;
    v.addLabel(lbl, {{fontSize: 8, fontColor: fc, backgroundColor: bg, backgroundOpacity: 0.85, position: atoms[i], inFront: true}});
}}
v.zoomTo();
v.render();
</script></body></html>'''
    return h

def get_2d_img(m, name, formula, mw, iupac):
    img = Draw.MolToImage(m, size=(450, 260))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f'''<div style="text-align:center;background:#ffffff !important;color:#0f172a !important;padding:15px;border-radius:12px;border:2px solid #1e3a8a;margin-top:10px;">
        <h3 style="color:#1e3a8a !important;margin:0 0 2px 0;text-decoration:underline;font-family:sans-serif;">{name}</h3>
        <p style="color:#475569 !important;margin:0 0 8px 0;font-size:12px;font-family:sans-serif;"><b>IUPAC:</b> {iupac}</p>
        <img src="data:image/png;base64,{img_b64}" style="max-width:100%;height:auto;margin:auto;display:block;">
        <div style="display:inline-block;border:1.5px solid #1e3a8a;border-radius:8px;padding:8px 18px;margin-top:12px;background:#f1f5f9 !important;font-size:13px;text-align:left;line-height:1.6;">
            <span style="color:#1e3a8a !important;font-weight:bold;">Molecular Formula:</span> <span style="color:#0f172a !important;font-weight:bold;">{formula}</span><br>
            <span style="color:#1e3a8a !important;font-weight:bold;">Molecular Weight:</span> <span style="color:#0f172a !important;font-weight:bold;">{mw:.2f} g/mol</span>
        </div>
    </div>'''

st.markdown("<h2 style='text-align:center;'>PubChem 3D AI Drug Screener</h2>", unsafe_allow_html=True)

query = st.text_input("🔍 Search any Drug Name / Formula / SMILES (e.g. Aceclofenac, C8H9NO2, Aspirin)", value="Aceclofenac")

if query:
    s, name, iupac = universal_pubchem_search(query)
    if not s:
        st.error(f'❌ "{query}" का डेटा PubChem पर नहीं मिला।')
    else:
        m = Chem.MolFromSmiles(s)
        formula = get_formula(m)
        mw = Descriptors.MolWt(m)
        lp = Descriptors.MolLogP(m)
        rt = Descriptors.NumRotatableBonds(m)
        ar = sum([1 for a in m.GetAtoms() if a.GetIsAromatic()])/float(m.GetNumAtoms())
        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        ps = rf.predict([[lp, mw, rt, ar]])[0]
        
        v_mw = mw > 500
        v_lp = lp > 5
        v_hbd = hbd > 5
        v_hba = hba > 10
        violations = sum([v_mw, v_lp, v_hbd, v_hba])
        
        st_mw = '⚠️' if v_mw else '✅'
        st_lp = '⚠️' if v_lp else '✅'
        st_hbd = '⚠️' if v_hbd else '✅'
        st_hba = '⚠️' if v_hba else '✅'
        st_v = '✅' if violations <= 1 else '⚠️'
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"### 💊 Compound: **{name}**")
            st.markdown(f"- **Molecular Formula:** **{formula}**")
            st.markdown(f"- **Canonical SMILES:** `{s}`")
            st.markdown(f"- **Rotatable Bonds:** {rt}")
            
            table_md = f"""| Parameter | Value | Status |
| :--- | :--- | :---: |
| **Molecular Weight** | {mw:.2f} g/mol | {st_mw} |
| **LogP** | {lp:.2f} | {st_lp} |
| **HBD** | {hbd} | {st_hbd} |
| **HBA** | {hba} | {st_hba} |
| **Lipinski Violations** | **{violations}** | **{st_v}** |"""
            st.markdown(table_md)
            st.markdown("---")
            st.markdown(f"**AI Predicted Aqueous Solubility (LogS):** **{ps:.2f} mol/L**")
            
            with st.expander("🖼️ View 2D Structure & IUPAC Details", expanded=False):
                st.components.v1.html(get_2d_img(m, name, formula, mw, iupac if iupac else 'N/A'), height=450)
                
        with col2:
            st.components.v1.html(get_3d_html(s), height=500)
            
  
