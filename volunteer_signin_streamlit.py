import streamlit as st
import pandas as pd
import gspread
import re
from oauth2client.service_account import ServiceAccountCredentials
from streamlit_autorefresh import st_autorefresh

# 🔁 Auto-refresh every 30 seconds
st_autorefresh(interval=30 * 1000, key="data_refresh")
st.title("📋 Spring Quarter Volunteer Sign-In Data (Live)")

# === Sheets API Setup ===
def get_gspread_client():
    scope = [
      "https://spreadsheets.google.com/feeds",
      "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
      "pantry-data-science-project-cafc469e7c30.json", scope
    )
    return gspread.authorize(creds)

client = get_gspread_client()
sheet  = client.open_by_url(
    "https://docs.google.com/spreadsheets/d/1D5btDIbj-15goMyBGx1NyHOZ4M9JT6TczhRp5EMF0oU"
)

# === 1) Weekly sign-up parser ===
def days(tracking):
    return [
        tracking.iloc[4:61,  1:4],
        tracking.iloc[4:61,  5:8],
        tracking.iloc[4:61,  9:12],
        tracking.iloc[4:61, 13:16],
        tracking.iloc[4:61, 17:20],
        tracking.iloc[4:62, 21:24],
        tracking.iloc[4:62, 25:28],
    ]

def weekday(day):
    return [
        day.iloc[3:11], day.iloc[11:17], day.iloc[17:23],
        day.iloc[23:29], day.iloc[29:35], day.iloc[35:41],
        day.iloc[41:47], day.iloc[47:52], day.iloc[52:57],
    ]

def weekend(day):
    return [
        day.iloc[10:14], day.iloc[16:20], day.iloc[52:57],
    ]

def entry(row):
    if pd.notna(row.iloc[1]) and str(row.iloc[1]).strip():
        return row.iloc[0], row.iloc[1], row.iloc[2]
    return None

def process_week_signup_grid(ws, week_label):
    raw     = ws.get_all_values()
    max_len = max(len(r) for r in raw)
    norm    = [r + [""]*(max_len-len(r)) for r in raw]
    df      = pd.DataFrame(norm, dtype=str)

    cols = ["Name","Week","Date","Time Block","Shift Type","Attended","Sign-Up Type"]
    out  = pd.DataFrame(columns=cols)

    blocks = days(df)
    time_labels = [
      "Stocking Shift: 9:00 AM - 10:30 AM",
      "10:00 AM - 11:00 AM","11:00 AM - 12:00 PM",
      "12:00 PM - 1:00 PM","1:00 PM - 2:00 PM",
      "2:00 PM - 3:00 PM","3:00 PM - 4:00 PM",
      "Closing Shift: 4:00 PM - 4:45 PM","Shift Covers (variable)"
    ]

    # Mon–Fri
    for di in range(5):
        block      = blocks[di]
        date_label = block.iloc[0,0].strip()
        for si, sub in enumerate(weekday(block)):
            tblock = time_labels[si]
            for _, row in sub.iterrows():
                e = entry(row)
                if not e:
                    continue
                stype, name, attend = e
                tag = stype
                if si==0: tag = f"{stype}, Stocking"
                if si==7: tag = f"{stype}, Closing"
                out.loc[len(out)] = [
                    name,
                    week_label,
                    date_label,
                    tblock,
                    tag,
                    "Yes" if str(attend).lower() in ["yes","true","✔","✓"] else "No",
                    "Weekly Sign-Up"
                ]

    # Sat–Sun
    weekend_labels = [
      "Opening Shift: 12:00 PM - 1:00 PM",
      "Closing Shift: 1:00 PM - 2:00 PM",
      "Shift Covers (variable)",
    ]
    for di in (5,6):
        block      = blocks[di]
        date_label = block.iloc[0,0].strip()
        for si, sub in enumerate(weekend(block)):
            tblock = weekend_labels[si]
            for _, row in sub.iterrows():
                e = entry(row)
                if not e:
                    continue
                stype, name, attend = e
                out.loc[len(out)] = [
                    name,
                    week_label,
                    date_label,
                    tblock,
                    stype,
                    "Yes" if str(attend).lower() in ["yes","true","✔","✓"] else "No",
                    "Weekly Sign-Up"
                ]

    return out

# === 2) Mobile Pantry parser ===
def process_mobile_pantry_grid(ws):
    raw = ws.get_all_values()
    df  = pd.DataFrame(raw, dtype=str)
    rows=[]

    for col in df.columns:
        for idx, cell in df[col].items():
            if pd.notna(cell) and "(Wk" in cell:
                header = cell.strip()
                m = re.search(r"\(Wk\s*(\d+)\s*-\s*([^)]*)\)", header)
                if not m:
                    continue
                week_label = f"Week {m.group(1)}"
                date_label = header.split(")",1)[1].strip()
                cidx = df.columns.get_loc(col)
                block = df.iloc[idx+1:, cidx:cidx+4].reset_index(drop=True)
                tblock = None

                for _, row in block.iterrows():
                    if pd.notna(row.iloc[0]) and row.iloc[0].strip():
                        tblock = row.iloc[0].strip()
                        continue
                    role = row.iloc[1].strip() if pd.notna(row.iloc[1]) else ""
                    name = row.iloc[2].strip() if pd.notna(row.iloc[2]) else ""
                    sign = str(row.iloc[3]).strip().lower()
                    if name:
                        rows.append({
                            "Name": name,
                            "Week": week_label,
                            "Date": date_label,
                            "Time Block": tblock,
                            "Shift Type": role,
                            "Attended": "Yes" if sign in ["yes","true","✔","✓"] else "No",
                            "Sign-Up Type": "Mobile Pantry"
                        })

    return pd.DataFrame(rows)

# === 3) Attendance fallback parser ===
def process_attendance_sheet(ws):
    raw = ws.get_all_records(head=22)
    df  = pd.DataFrame(raw).dropna(subset=["First + Last Name"])
    rows=[]
    for _, r in df.iterrows():
        name = r["First + Last Name"].strip()
        for col,val in r.items():
            if isinstance(val,(int,float)) and val>0:
                wk_match    = re.search(r"(Week \d+)", col)
                week_label  = wk_match.group(1) if wk_match else "All Weeks"
                shift_col   = col.replace(" Hours","").strip()
                # assign Sign-Up Type for fallback
                if "Food Recovery" in shift_col:
                    sut = "Food Recovery"
                elif "Mobile Pantry" in shift_col:
                    sut = "Mobile Pantry"
                elif "Misc" in shift_col:
                    sut = "Misc"
                else:
                    sut = "Attendance Hours"

                rows.append({
                    "Name": name,
                    "Week": week_label,
                    "Date": "Logged hours",
                    "Time Block": "",
                    "Shift Type": shift_col,
                    "Attended": "Yes",
                    "Sign-Up Type": sut
                })
    return pd.DataFrame(rows)

# === 4) Combine & Replace “Logged hours” ===
def collect_all_data():
    signup_week_dict = {}
    signup_dfs       = []
    mobile_dfs       = []

    # parse sign-up tabs
    for ws in sheet.worksheets():
        t = ws.title.lower()
        if "mobile pantry" in t:
            df_mp = process_mobile_pantry_grid(ws)
            if not df_mp.empty:
                mobile_dfs.append(df_mp)
        elif "sign-ups" in t and "week" in t:
            m = re.search(r"Week\s*(\d+)", ws.title)
            wl = f"Week {m.group(1)}" if m else ws.title
            dfw = process_week_signup_grid(ws, wl)
            if not dfw.empty:
                signup_week_dict[wl] = dfw
                signup_dfs.append(dfw)

    # fallback attendance
    attend_df = process_attendance_sheet(
        sheet.worksheet("Names and Attendance")
    )

    # replace Logged hours
    replaced = []
    for _, r in attend_df.iterrows():
        if r["Date"]=="Logged hours" and r["Week"] in signup_week_dict:
            real = signup_week_dict[r["Week"]]
            match = real[
                real["Name"].str.lower().str.strip()==r["Name"].lower().strip()
            ]
            replaced.append(match if not match.empty else pd.DataFrame([r]))
        else:
            replaced.append(pd.DataFrame([r]))
    attendance_filled = pd.concat(replaced, ignore_index=True)

    # final concat
    combined = pd.concat(
        signup_dfs + mobile_dfs + [attendance_filled],
        ignore_index=True
    )

    # drop blank names / Totals
    combined = combined[
        combined["Name"].str.strip().astype(bool)
        & (combined["Shift Type"].str.lower() != "total")
    ]

    return combined

# === 5) Display & Download ===
df_all = collect_all_data()
st.dataframe(df_all, use_container_width=True)

csv = df_all.to_csv(index=False).encode("utf-8")
st.download_button("Download as CSV", csv, "spring_volunteer_data.csv", "text/csv")
