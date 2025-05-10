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

# === 1) Weekly Sign-Ups Parser ===
def days(tracking):
    return [
        tracking.iloc[4:61, 1:4],
        tracking.iloc[4:61, 5:8],
        tracking.iloc[4:61, 9:12],
        tracking.iloc[4:61,13:16],
        tracking.iloc[4:61,17:20],
        tracking.iloc[4:62,21:24],
        tracking.iloc[4:62,25:28],
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

def process_week_signup_grid(ws, week_label):
    raw     = ws.get_all_values()
    max_len = max(len(r) for r in raw)
    norm    = [r + [""]*(max_len - len(r)) for r in raw]
    df      = pd.DataFrame(norm, dtype=str)

    cols = [
      "Name","Week","Date","Time Block",
      "Position","Shift Type","Attended","Sign-Up Type"
    ]
    out = pd.DataFrame(columns=cols)

    blocks      = days(df)
    time_labels = [
        "Stocking Shift: 9:00 AM – 10:30 AM",
        "10:00 AM – 11:00 AM","11:00 AM – 12:00 PM",
        "12:00 PM – 1:00 PM","1:00 PM – 2:00 PM",
        "2:00 PM – 3:00 PM","3:00 PM – 4:00 PM",
        "Closing Shift: 4:00 PM – 4:45 PM","Shift Covers (variable)"
    ]

    # Mon–Fri
    for di in range(5):
        blk        = blocks[di]
        date_label = blk.iloc[0,0].strip()
        for si, sub in enumerate(weekday(blk)):
            tblock = time_labels[si]
            for _, row in sub.iterrows():
                name = row.iloc[1]
                if not pd.notna(name) or not name.strip(): 
                    continue
                shift_type = row.iloc[0].strip()
                sig        = str(row.iloc[2]).strip().lower()
                attended   = "Yes" if sig in ["yes","true","✔","✓"] else "No"
                position   = "Volunteer"
                if   si == 0: shift_type += ", Stocking"
                elif si == 7: shift_type += ", Closing"
                out.loc[len(out)] = [
                    name.strip(), week_label, date_label, tblock,
                    position, shift_type, attended, "Weekly Sign-Up"
                ]

    # Sat–Sun
    weekend_labels = [
        "Opening Shift: 12:00 PM – 1:00 PM",
        "Closing Shift: 1:00 PM – 2:00 PM",
        "Shift Covers (variable)"
    ]
    for di in (5,6):
        blk        = blocks[di]
        date_label = blk.iloc[0,0].strip()
        for si, sub in enumerate(weekend(blk)):
            tblock = weekend_labels[si]
            for _, row in sub.iterrows():
                name = row.iloc[1]
                if not pd.notna(name) or not name.strip():
                    continue
                shift_type = row.iloc[0].strip()
                sig        = str(row.iloc[2]).strip().lower()
                attended   = "Yes" if sig in ["yes","true","✔","✓"] else "No"
                position   = "Volunteer"
                out.loc[len(out)] = [
                    name.strip(), week_label, date_label, tblock,
                    position, shift_type, attended, "Weekly Sign-Up"
                ]

    return out

# === 2) Mobile Pantry Parser ===
def process_mobile_pantry_grid(ws):
    import re
    raw = ws.get_all_values()
    df = pd.DataFrame(raw)
    rows = []
    max_week = 20  # Arbitrary high number to avoid skipping weeks
    for col in df.columns:
        for idx, cell in df[col].items():
            if pd.notna(cell) and "(Wk" in str(cell):
                day_label = str(cell).strip()
                # Extract the week number
                match = re.search(r"Wk\s*(\d+)", day_label)
                if match:
                    week_num = int(match.group(1))
                    if week_num > max_week:
                        continue
                else:
                    continue
                target_col = col
                start_row = idx
                # Get column index
                try:
                    target_col_idx = int(target_col)
                except ValueError:
                    target_col_idx = df.columns.get_loc(target_col)
                columns_to_extract = df.iloc[start_row + 1:, target_col_idx:target_col_idx + 4].reset_index(drop=True)
                current_time_block = None
                for _, row in columns_to_extract.iterrows():
                    # Carry forward the time block
                    if len(row) > 0 and pd.notna(row.iloc[0]) and row.iloc[0] != "":
                        current_time_block = row.iloc[0]
                    role = row.iloc[1] if len(row) > 1 else ""
                    name = row.iloc[2] if len(row) > 2 else ""
                    signup = row.iloc[3] if len(row) > 3 else ""
                    if not name or not str(name).strip():
                        continue
                    # Extract week/location/date from day_label
                    m = re.match(r"\(Wk\s*(\d+)\s*-\s*([^)]+)\)\s*(.*)", day_label)
                    if m:
                        week_lbl = f"Week {m.group(1)}"
                        location = m.group(2).strip()
                        date_lbl = m.group(3).strip()
                    else:
                        week_lbl = ""
                        location = ""
                        date_lbl = ""
                    position = "Volunteer" if "Volunteer" in str(role) else str(role)
                    attended = "Yes" if str(signup).strip().lower() in ["yes", "true", "✔", "✓"] else ("No" if str(signup).strip().lower() in ["no", "false", "✗", "✕"] else str(signup))
                    rows.append({
                        "Name": str(name).strip(),
                        "Week": week_lbl,
                        "Location": location,
                        "Date": date_lbl,
                        "Time Block": current_time_block,
                        "Position": position,
                        "Shift Type": f"Mobile Pantry ({location})",
                        "Attended": attended,
                        "Sign-Up Type": "Mobile Pantry"
                    })
    return pd.DataFrame(rows)

# === 3) Food Recovery Parser ===
def process_food_recovery_grid(ws):
    import re
    raw = ws.get_all_values()
    df = pd.DataFrame(raw)
    rows = []
    # Find the row with week headers (contains "Week" or "Finals Week")
    week_row = None
    for idx, row in df.iterrows():
        if any(re.search(r"(?:Week|Finals Week)", str(cell)) for cell in row):
            week_row = idx
            break
    if week_row is None:
        return pd.DataFrame(columns=[
            "Name", "Week", "Date", "Time Block",
            "Position", "Shift Type", "Attended", "Sign-Up Type"
        ])
    col = 4
    while col < df.shape[1]:
        week_lbl = str(df.iloc[week_row, col]).strip()
        # For each event block, scan all rows below the week header
        # Find the event name from the event header row (row 5, 9, 13, ...)
        event_row = week_row + 1
        while event_row < len(df):
            event_name = str(df.iloc[event_row, 1]).strip() if 1 < len(df.columns) and event_row < len(df) else ""
            if not event_name:
                event_row += 1
                continue
            # Find the date for this event/week from the first date found in the block
            date_lbl = ""
            for r in range(event_row, event_row + 10):
                if r >= len(df): break
                val = df.iloc[r, col] if col < len(df.columns) else ""
                if val and re.match(r"\d{2}/\d{2}/\d{4}", str(val)):
                    date_lbl = str(val).strip()
                    break
            # Now scan all rows in the block for names (including rows like 6, 10, 15, ...)
            for r in range(event_row, event_row + 20):
                if r >= len(df): break
                pos_val  = df.iloc[r, col+1] if col+1 < len(df.columns) and r < len(df) else ""
                name_val = df.iloc[r, col+2] if col+2 < len(df.columns) and r < len(df) else ""
                att_val  = df.iloc[r, col+3] if col+3 < len(df.columns) and r < len(df) else ""
                if not name_val or not str(name_val).strip():
                    continue
                # Normalize Volunteer 1/2/3/4 to 'Volunteer'
                pos_str = str(pos_val).strip()
                if re.match(r"Volunteer\s*\d+", pos_str, re.I):
                    position = "Volunteer"
                else:
                    position = pos_str if pos_str else "Volunteer"
                attended = "Yes" if str(att_val).strip().lower() in ["yes", "true", "✔", "✓"] else "No"
                rows.append({
                    "Name": str(name_val).strip(),
                    "Week": week_lbl,
                    "Date": date_lbl,
                    "Time Block": "",
                    "Position": position,
                    "Shift Type": event_name,
                    "Attended": attended,
                    "Sign-Up Type": "Food Recovery"
                })
            # Move to next event header (look for next non-empty event name in col 1)
            next_event_row = event_row + 1
            while next_event_row < len(df):
                next_event_name = str(df.iloc[next_event_row, 1]).strip() if 1 < len(df.columns) and next_event_row < len(df) else ""
                if next_event_name:
                    break
                next_event_row += 1
            event_row = next_event_row
        col += 4
    return pd.DataFrame(rows)

# === 4) Attendance fallback parser ===
def process_attendance_sheet(ws):
    raw = ws.get_all_records(head=22)
    df  = pd.DataFrame(raw).dropna(subset=["First + Last Name"])
    rows=[]
    for _, r in df.iterrows():
        nm = r["First + Last Name"].strip()
        for col,val in r.items():
            if isinstance(val,(int,float)) and val>0:
                m         = re.search(r"(Week \d+)", col)
                week_lbl  = m.group(1) if m else "All Weeks"
                shift_col = col.replace(" Hours","").strip()
                if "Food Recovery"  in shift_col: sut = "Food Recovery"
                elif "Mobile Pantry" in shift_col: sut = "Mobile Pantry"
                elif "Misc"          in shift_col: sut = "Misc"
                else:                             sut = "Attendance Hours"
                rows.append({
                    "Name":         nm,
                    "Week":         week_lbl,
                    "Date":         "Logged hours",
                    "Time Block":   "",
                    "Position":     "Volunteer",
                    "Shift Type":   shift_col,
                    "Attended":     "Yes",
                    "Sign-Up Type": sut
                })
    return pd.DataFrame(rows)

# === 5) Combine & Patch "Logged hours" ===
def collect_all_data():
    weekly_dfs   = []
    pantry_dfs   = []
    recovery_dfs = []

    for ws in sheet.worksheets():
        t = ws.title.lower()
        if 'template' in t or 'copy' in t:
            continue
        if "mobile pantry" in t:
            dfm = process_mobile_pantry_grid(ws)
            st.write(f"Mobile Pantry: {ws.title}", dfm)
            if not dfm.empty:
                pantry_dfs.append(dfm)
        elif "food recovery" in t:
            dfr = process_food_recovery_grid(ws)
            st.write(f"Food Recovery: {ws.title}", dfr)
            if not dfr.empty:
                recovery_dfs.append(dfr)
        elif "sign-ups" in t and "week" in t:
            m  = re.search(r"Week\s*(\d+)", ws.title)
            wl = f"Week {m.group(1)}" if m else ws.title
            w  = process_week_signup_grid(ws, wl)
            if not w.empty:
                weekly_dfs.append(w)

    attend = process_attendance_sheet(sheet.worksheet("Names and Attendance"))
    attend = attend[attend["Sign-Up Type"]=="Attendance Hours"]

    # patch logged-hours into weekly sign-ups
    week_map = {w["Week"].iat[0]: w for w in weekly_dfs}
    patched  = []
    for _, r in attend.iterrows():
        if r["Date"]=="Logged hours" and r["Week"] in week_map:
            real  = week_map[r["Week"]]
            match = real[
                real["Name"].str.lower().str.strip()
                == r["Name"].lower().strip()
            ]
            patched.append(match if not match.empty else pd.DataFrame([r]))
        else:
            patched.append(pd.DataFrame([r]))
    attendance_filled = pd.concat(patched, ignore_index=True)

    combined = pd.concat(
        weekly_dfs + pantry_dfs + recovery_dfs + [attendance_filled],
        ignore_index=True
    )

    # === 6) Shift Category normalization ===
    def categorize(s):
        s = s.lower()
        if any(x in s for x in ["regular pantry","attendance hours"]):
            return "Regular Pantry"
        if "normal" in s:
            return "Normal shift"
        if "stocking" in s:
            return "Stocking shift"
        if "closing" in s:
            return "Closing shift"
        if "shift lead" in s or "portioning lead" in s:
            return "Shift lead"
        if "mobile pantry" in s:
            return "Mobile Pantry"
        if "scc" in s:
            return "SCC shift"
        if "arc" in s:
            return "ARC shift"
        if "food recovery" in s:
            return "Food Recovery"
        if "dining commons recovery" in s:
            return "Dining Commons Recovery"
        if "dc food tray portioning" in s:
            return "DC food tray portioning"
        if "farmers market recovery" in s:
            return "Farmers Market Recovery + Distribution"
        if "vehicle driver" in s:
            return "Vehicle Driver"
        if "shift cover" in s:
            return "Shift cover in general"
        return "Normal shift"

    combined["Shift Category"] = combined["Shift Type"].apply(categorize)

    # drop blank names and "total"
    combined = combined[
        combined["Name"].str.strip().astype(bool)
        & (combined["Shift Type"].str.lower() != "total")
    ]

    # === 7) Sort so you see FR & MP first ===
    order = {
        "Food Recovery":    0,
        "Mobile Pantry":    1,
        "Weekly Sign-Up":   2,
        "Attendance Hours": 3
    }
    combined["__ord"] = combined["Sign-Up Type"].map(order).fillna(4).astype(int)
    combined = combined.sort_values(["__ord","Week","Date","Time Block"])
    combined = combined.drop(columns="__ord")

    return combined

# === 8) Render & Download ===
df_all = collect_all_data()

# show counts
counts = df_all["Sign-Up Type"].value_counts()
st.write("**Row counts by Sign-Up Type**")
st.write(counts.to_frame("Rows"))

st.subheader("📊 Combined Volunteer Table")
st.dataframe(df_all, use_container_width=True)

csv = df_all.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "spring_volunteer_data.csv", "text/csv")
