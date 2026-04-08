"""
Module 7: Suspicious Transaction Report (STR) Generator
AI-Powered Mule Account and Financial Fraud Network Detection System

Auto-generates professional PDF Suspicious Transaction Reports for
high-risk accounts flagged by the TGN + XAI pipeline.

Each STR contains:
  - Account summary and risk score
  - Key fraud indicators
  - Transaction history
  - SHAP-based explanation
  - AI-generated narrative
  - Recommended action
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def clean_text(text: str) -> str:
    """Remove all non-latin1 characters that Helvetica cannot render."""
    replacements = {
        '—': '-', '–': '-', '‒': '-',
        '’': "'", '‘': "'",
        '“': '"', '”': '"',
        '…': '...', '•': '*',
        '✓': '>', '↑': '^', '↓': 'v',
        '₹': 'Rs.', '°': ' deg',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final safety: encode to latin-1, replacing anything still unsupported
    return text.encode('latin-1', errors='replace').decode('latin-1')


try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    print("[WARN] fpdf2 not installed. Run: pip install fpdf2")


# ─────────────────────────────────────────────
#  STR PDF Generator
# ─────────────────────────────────────────────

class STRReport(FPDF):
    """Custom PDF class for Suspicious Transaction Reports."""

    def __init__(self, account_id, risk_level):
        super().__init__()
        self.account_id = account_id
        self.risk_level = risk_level
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()

    def header(self):
        # Top bar
        risk_colors = {
            'CRITICAL': (192, 0, 0),
            'HIGH':     (197, 90, 17),
            'MEDIUM':   (255, 192, 0),
            'LOW':      (55, 86, 35),
        }
        r, g, b = risk_colors.get(self.risk_level, (46, 117, 182))
        self.set_fill_color(r, g, b)
        self.rect(0, 0, 210, 18, 'F')

        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        self.cell(0, 10,
                  f'SUSPICIOUS TRANSACTION REPORT  |  {self.risk_level} RISK',
                  ln=False)
        self.set_text_color(0, 0, 0)
        self.ln(20)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5,
                  f'CONFIDENTIAL - AI-Powered Fraud Detection System  |  '
                  f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}  |  '
                  f'Page {self.page_no()}',
                  align='C')

    def section_title(self, title: str):
        self.set_font('Helvetica', 'B', 11)
        self.set_fill_color(214, 228, 240)
        self.set_text_color(31, 78, 121)
        self.cell(0, 8, f'  {title}', ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def key_value_row(self, label: str, value: str,
                      highlight: bool = False):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(
            242, 242, 242) if not highlight else self.set_fill_color(252, 228, 214)
        self.cell(65, 7, f'  {label}', border=1, fill=True)
        self.set_font('Helvetica', '', 9)
        self.cell(0, 7, f'  {value}', border=1, fill=False, ln=True)

    def narrative_box(self, text: str):
        self.set_font('Helvetica', 'I', 9)
        self.set_fill_color(240, 248, 255)
        self.set_draw_color(46, 117, 182)
        self.multi_cell(0, 6, text, border=1, fill=True)
        self.ln(3)

    def risk_gauge(self, prob: float):
        """Draw a simple horizontal risk gauge."""
        x, y = 130, self.get_y()
        w, h = 60, 8

        # Background bar
        self.set_fill_color(200, 200, 200)
        self.rect(x, y, w, h, 'F')

        # Filled portion
        filled = w * prob
        if prob >= 0.8:
            self.set_fill_color(192, 0, 0)
        elif prob >= 0.6:
            self.set_fill_color(197, 90, 17)
        elif prob >= 0.4:
            self.set_fill_color(255, 192, 0)
        else:
            self.set_fill_color(55, 86, 35)
        self.rect(x, y, filled, h, 'F')

        # Label
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(255, 255, 255)
        self.set_xy(x + 2, y + 1)
        self.cell(filled - 2, 6, f'{prob*100:.1f}%')
        self.set_text_color(0, 0, 0)


def generate_str_pdf(exp: dict,
                     transactions: pd.DataFrame,
                     save_dir: str = "output/reports") -> str:
    """Generate a single STR PDF for one flagged account."""

    acc = exp['account_id']
    prob = exp['fraud_prob']
    risk = exp['risk_level']
    feats = exp['features']
    narr = exp.get('narrative', 'No narrative available.')

    pdf = STRReport(acc, risk)

    # ── Section 1: Report Header Info ──────────────────────────
    pdf.section_title("1. REPORT INFORMATION")
    pdf.key_value_row("Report Date",     datetime.now().strftime("%d %B %Y"))
    pdf.key_value_row("Report Type",     "Suspicious Transaction Report (STR)")
    pdf.key_value_row(
        "System",          "AI-Powered Mule Account Detection System")
    pdf.key_value_row("Model",           "Temporal Graph Network (TGN) + XAI")
    pdf.key_value_row("Dataset",         "PaySim Financial Transactions")
    pdf.ln(4)

    # ── Section 2: Account Details ─────────────────────────────
    pdf.section_title("2. FLAGGED ACCOUNT DETAILS")
    pdf.key_value_row("Account ID",      acc, highlight=True)
    pdf.key_value_row("Risk Level",      risk,
                      highlight=(risk in ['CRITICAL', 'HIGH']))
    pdf.key_value_row("Fraud Probability", f"{prob*100:.2f}%", highlight=True)

    # Risk gauge
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(242, 242, 242)
    pdf.cell(65, 7, '  Risk Gauge', border=1, fill=True)
    gauge_x = pdf.get_x()
    gauge_y = pdf.get_y()
    pdf.cell(0, 7, '', border=1, ln=True)
    pdf.risk_gauge(prob)
    pdf.ln(2)

    pdf.key_value_row("Transactions Sent",
                      f"{int(feats.get('sent_count', 0)):,}")
    pdf.key_value_row("Transactions Received",
                      f"{int(feats.get('recv_count', 0)):,}")
    pdf.key_value_row("Total Amount Sent",
                      f"Rs.{feats.get('sent_total', 0):,.2f}")
    pdf.key_value_row("Total Amount Received",
                      f"Rs.{feats.get('recv_total', 0):,.2f}")
    pdf.key_value_row("Fund Dwell Time",
                      f"{feats.get('dwell_time', 0):.1f} hours")
    pdf.key_value_row("Forward Ratio",
                      f"{feats.get('forward_ratio', 0)*100:.1f}%")
    pdf.key_value_row("Balance Drains",
                      f"{int(feats.get('balance_drains', 0))}")
    pdf.key_value_row("High-Risk Tx Sent",
                      f"{int(feats.get('high_risk_sent', 0))}")
    pdf.ln(4)

    # ── Section 3: Fraud Indicators ────────────────────────────
    pdf.section_title("3. KEY FRAUD INDICATORS")

    indicators = []
    if feats.get('balance_drains', 0) >= 1:
        indicators.append(
            f"[*] BALANCE DRAIN: Account fully drained balance on "
            f"{int(feats.get('balance_drains', 0))} occasion(s)"
        )
    if feats.get('dwell_time', 999) <= 2:
        indicators.append(
            f"[*] RAPID FORWARDING: Funds forwarded within "
            f"{feats.get('dwell_time', 0):.0f} hour(s) of receipt"
        )
    if feats.get('forward_ratio', 0) > 0.8:
        indicators.append(
            f"[*] LAYERING PATTERN: {feats.get('forward_ratio', 0)*100:.0f}% "
            f"of received funds immediately forwarded"
        )
    if feats.get('high_risk_sent', 0) >= 2:
        indicators.append(
            f"[*] HIGH-RISK TYPES: {int(feats.get('high_risk_sent', 0))} "
            f"TRANSFER/CASH_OUT transactions"
        )
    if feats.get('sent_count', 0) >= 3 and feats.get('recv_count', 0) >= 3:
        indicators.append(
            f"[*] FAN-IN/FAN-OUT: Received from "
            f"{int(feats.get('recv_count', 0))} sources, sent to "
            f"{int(feats.get('sent_count', 0))} destinations"
        )
    if feats.get('send_velocity', 0) > 0.5:
        indicators.append(
            f"[*] HIGH VELOCITY: {feats.get('send_velocity', 0):.2f} "
            f"transactions/hour"
        )

    if not indicators:
        indicators.append(
            "- Elevated composite risk score from temporal pattern analysis")

    pdf.set_font('Helvetica', '', 9)
    for ind in indicators:
        pdf.set_fill_color(
            252, 228, 214) if '[*]' in ind else pdf.set_fill_color(255, 255, 255)
        pdf.cell(0, 7, f'  {clean_text(ind)}', ln=True, fill=True, border=1)
    pdf.ln(4)

    # ── Section 4: Top SHAP Drivers ────────────────────────────
    pdf.section_title("4. AI EXPLANATION - TOP RISK DRIVERS (SHAP)")
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5,
             '  SHAP values indicate the contribution of each feature to the fraud prediction.',
             ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    drivers = exp.get('top_drivers', [])
    if drivers:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(31, 78, 121)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(100, 7, '  Feature', border=1, fill=True)
        pdf.cell(0,   7, '  SHAP Contribution', border=1, fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)

        for feat_name, shap_val in drivers:
            pdf.set_font('Helvetica', '', 9)
            fill = (252, 228, 214) if shap_val > 0 else (226, 239, 218)
            pdf.set_fill_color(*fill)
            direction = "[+] Increases Risk" if shap_val > 0 else "[-] Decreases Risk"
            pdf.cell(100, 7, f'  {feat_name}', border=1, fill=True)
            pdf.cell(0,   7,
                     f'  {shap_val:+.4f}  ({direction})',
                     border=1, fill=True, ln=True)
    pdf.ln(4)

    # ── Section 5: AI Narrative ────────────────────────────────
    pdf.section_title("5. AI-GENERATED INVESTIGATION NARRATIVE")
    pdf.narrative_box(clean_text(narr))

    # ── Section 6: Transaction History ────────────────────────
    pdf.section_title("6. RECENT TRANSACTION HISTORY")

    acc_txns = transactions[
        (transactions['nameOrig'] == acc) |
        (transactions['nameDest'] == acc)
    ].sort_values('step', ascending=False).head(10)

    if len(acc_txns) > 0:
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_fill_color(31, 78, 121)
        pdf.set_text_color(255, 255, 255)
        col_widths = [18, 25, 40, 40, 25, 18, 20]
        headers = ['Step', 'Type', 'Sender',
                   'Receiver', 'Amount', 'Fraud', 'Drain']
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, f' {h}', border=1, fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        for _, row in acc_txns.iterrows():
            is_fraud = int(row.get('isFraud', 0))
            fill = (252, 228, 214) if is_fraud else (255, 255, 255)
            pdf.set_fill_color(*fill)
            pdf.set_font('Helvetica', '', 7)
            vals = [
                str(int(row['step'])),
                str(row.get('type', '')),
                str(row['nameOrig'])[:16],
                str(row['nameDest'])[:16],
                f"Rs.{row['amount']:,.0f}",
                'YES' if is_fraud else 'no',
                'YES' if row.get('balance_drained', 0) else 'no',
            ]
            for w, v in zip(col_widths, vals):
                pdf.cell(w, 6, f' {v}', border=1, fill=True)
            pdf.ln()
    else:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 7, '  No transaction history available.', ln=True)

    pdf.ln(4)

    # ── Section 7: Recommended Action ─────────────────────────
    pdf.section_title("7. RECOMMENDED ACTION")
    actions = {
        'CRITICAL': "IMMEDIATE ACCOUNT FREEZE - Escalate to senior AML officer. File STR with FIU-IND within 24 hours. Preserve all transaction records.",
        'HIGH':     "FLAG FOR ENHANCED DUE DILIGENCE - Request full transaction history. Schedule KYC re-verification. Monitor all future activity.",
        'MEDIUM':   "ENHANCED MONITORING - Set real-time transaction alerts. Review account activity weekly. Request source of funds documentation.",
        'LOW':      "STANDARD MONITORING - Log for periodic review. No immediate action required.",
    }
    action_text = actions.get(risk, "Review required.")

    risk_colors = {
        'CRITICAL': (252, 228, 214),
        'HIGH':     (255, 242, 204),
        'MEDIUM':   (255, 255, 204),
        'LOW':      (226, 239, 218),
    }
    pdf.set_fill_color(*risk_colors.get(risk, (242, 242, 242)))
    pdf.set_font('Helvetica', 'B', 10)
    pdf.multi_cell(0, 8, f'  {clean_text(action_text)}', border=1, fill=True)
    pdf.ln(4)

    # ── Section 8: Sign-off ────────────────────────────────────
    pdf.section_title("8. SIGN-OFF")
    pdf.key_value_row("Investigating Officer",
                      "_______________________________")
    pdf.key_value_row(
        "Date",                  "_______________________________")
    pdf.key_value_row("Signature",
                      "_______________________________")
    pdf.key_value_row("Case Reference No.",
                      "_______________________________")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"STR_{acc[:15]}_{risk}.pdf")
    pdf.output(filename)
    return filename


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output/reports", exist_ok=True)

    print("=" * 60)
    print("  MODULE 7: SUSPICIOUS TRANSACTION REPORT GENERATOR")
    print("=" * 60)

    if not FPDF_AVAILABLE:
        print("[ERROR] fpdf2 not installed. Run: pip install fpdf2")
        exit(1)

    # ── Load outputs ──
    required = ["output/fraud_explanations.json",
                "output/fraud_narratives.json",
                "output/transactions_processed.csv"]
    for f in required:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}")
            exit(1)

    print("[INFO] Loading explanation data...")
    with open("output/fraud_explanations.json") as f:
        explanations = json.load(f)
    with open("output/fraud_narratives.json") as f:
        narratives = json.load(f)

    df = pd.read_csv("output/transactions_processed.csv")
    print(f"[INFO] Loaded {len(explanations)} explanations")

    # Merge narratives into explanations
    narr_lookup = {n['account_id']: n['narrative'] for n in narratives}
    for exp in explanations:
        exp['narrative'] = narr_lookup.get(exp['account_id'], '')

    # ── Generate STRs for HIGH + CRITICAL accounts ──
    high_risk = [e for e in explanations
                 if e['risk_level'] in ['CRITICAL', 'HIGH', 'MEDIUM']]

    print(f"[INFO] Generating STRs for {len(high_risk)} flagged accounts...")
    print(
        f"       CRITICAL: {sum(1 for e in high_risk if e['risk_level'] == 'CRITICAL')}")
    print(
        f"       HIGH:     {sum(1 for e in high_risk if e['risk_level'] == 'HIGH')}")
    print(
        f"       MEDIUM:   {sum(1 for e in high_risk if e['risk_level'] == 'MEDIUM')}")
    print()

    generated = []
    for i, exp in enumerate(high_risk[:20]):  # Cap at 20 reports
        try:
            path = generate_str_pdf(exp, df)
            generated.append(path)
            risk = exp['risk_level']
            prob = exp['fraud_prob']
            print(
                f"  [{i+1:02d}] {risk:<8} | {prob*100:5.1f}% | {exp['account_id']} → {os.path.basename(path)}")
        except Exception as e:
            print(f"  [WARN] Failed for {exp['account_id']}: {e}")

    print()
    print("=" * 60)
    print(f"[[*]] STR generation complete.")
    print(f"    Reports generated: {len(generated)}")
    print(f"    Saved to: output/reports/")
    print(f"    Next step: Run dashboard.py")
    print("=" * 60)
