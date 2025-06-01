import streamlit as st
import pandas as pd
import gspread
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from oauth2client.service_account import ServiceAccountCredentials
from streamlit_autorefresh import st_autorefresh

# 🔁 Auto-refresh every day
st_autorefresh(interval=86_400_000, key="data_refresh")
st.title("📋 Spring Quarter Volunteer Engagement and Pattern Analysis")
# Create tabs for different analytics views
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Contributions",
    "Data",
    "Shifts Covered",
    "Day & Time Analysis", 
    "Cancellation Patterns", 
    "Drop-off Volunteers", 
    "Reputation"
])

with tab2:
    

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

                if not event_name or event_name in [
                    "Recoveries/Events", "Task Summary", "Meet-up Location/Time", "Date/Time Sign-Ups", "Canceled Recovery"
                ]:
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

                    if event_name == "Dining Commons Recovery":
                        time_block = "Mondays 1:15 PM"
                    elif event_name == "DC Food Tray Portioning":
                        time_block = "Thursdays 5:00 PM"
                    elif event_name == "Farmers Market Recovery + Distribution (Green or Orchard Park Apartments)":
                        time_block = "Saturday 12:45 PM"
                    else:
                        time_block = ""

                    rows.append({
                        "Name": str(name_val).strip(),
                        "Week": week_lbl,
                        "Date": date_lbl,
                        "Time Block": time_block,
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
    def collect_all_data_and_hours():
        weekly_dfs   = []
        pantry_dfs   = []
        recovery_dfs = []
        name_to_total_hours = {}

        for ws in sheet.worksheets():
            t = ws.title.lower()
            if 'template' in t or 'copy' in t:
                continue
            if "mobile pantry" in t:
                dfm = process_mobile_pantry_grid(ws)
                st.write(f"{ws.title}", dfm)
                if not dfm.empty:
                    pantry_dfs.append(dfm)
            elif "food recovery" in t:
                dfr = process_food_recovery_grid(ws)
                st.write(f"{ws.title}", dfr)
                if not dfr.empty:
                    recovery_dfs.append(dfr)
            elif "sign-ups" in t and "week" in t:
                m  = re.search(r"Week\\s*(\\d+)", ws.title)
                wl = f"Week {m.group(1)}" if m else ws.title
                w  = process_week_signup_grid(ws, wl)
                if not w.empty:
                    weekly_dfs.append(w)

        # Extract total hours from Names and Attendance worksheet (column V), dynamically finding the header row
        ws_attend = sheet.worksheet("Names and Attendance")
        raw = ws_attend.get_all_values()
        header_row_idx = None
        name_col = None
        total_col = None
        # Find the header row and column indices
        for idx, row in enumerate(raw):
            if any(cell.strip() == "First + Last Name" for cell in row) and any(cell.strip() == "Total Hours" for cell in row):
                header_row_idx = idx
                break
        if header_row_idx is not None:
            header = raw[header_row_idx]
            for i, col in enumerate(header):
                if col.strip() == "First + Last Name":
                    name_col = i
                if col.strip() == "Total Hours":
                    total_col = i
            # Only process rows below the header
            for row in raw[header_row_idx+1:]:
                if len(row) > max(name_col, total_col):
                    name = row[name_col].strip()
                    try:
                        total = float(row[total_col])
                    except (ValueError, TypeError, IndexError):
                        total = 0.0
                    if name:
                        name_to_total_hours[name] = total

        attend = process_attendance_sheet(ws_attend)
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

        return combined, name_to_total_hours

    # === 8) Render & Download ===
    df_all, name_to_total_hours = collect_all_data_and_hours()

    # show counts
    counts = df_all["Sign-Up Type"].value_counts()
    st.write("**Row counts by Sign-Up Type**")
    st.write(counts.to_frame("Rows"))

    # === 9) VOLUNTEER ANALYTICS ===

    def compute_day_time(df):
        
        # Skip rows without proper date or time block
        df = df[df['Date'].notna() & df['Time Block'].notna() & (df['Date'] != 'Logged hours')]
        
        # Group by Date and Time Block
        grouped = df.groupby(['Date', 'Time Block'])
        
        # Count total shifts and cancellations
        total_shifts = grouped.size().reset_index(name='Total Shifts')
        cancelled = df[df['Attended'] == 'No'].groupby(['Date', 'Time Block']).size().reset_index(name='Cancellations')
        
        # Merge the counts
        result = pd.merge(total_shifts, cancelled, on=['Date', 'Time Block'], how='left')
        
        # Fill NaN values in Cancellations with 0
        result['Cancellations'] = result['Cancellations'].fillna(0).astype(int)
        
        # Calculate cancellation rate
        result['Cancellation Rate'] = (result['Cancellations'] / result['Total Shifts']).round(3)
        
        # Sort by date and time block
        result = result.sort_values(['Date', 'Time Block'])
        
        return result

    def compute_cancellation_pattern(df):
        
        # Skip rows without proper date or time block
        df = df[df['Date'].notna() & df['Time Block'].notna() & (df['Date'] != 'Logged hours')]
        
        # Extract weekday from date
        def extract_weekday(date_str):
            # First try to extract from format "Monday, April 1"
            if ',' in str(date_str):
                return date_str.split(',')[0].strip()
            
            # For other formats, try to parse the date and get the weekday
            try:
                for fmt in ['%m/%d/%Y', '%Y-%m-%d']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime('%A')  # Full weekday name
                    except ValueError:
                        continue
            except:
                return 'Unknown'
        
        df['Weekday'] = df['Date'].apply(extract_weekday)
        
        # Group by Weekday and Time Block
        grouped = df.groupby(['Weekday', 'Time Block'])
        
        # Count total shifts and cancellations
        total_shifts = grouped.size()
        cancelled = df[df['Attended'] == 'No'].groupby(['Weekday', 'Time Block']).size()
        
        # Calculate cancellation rates
        cancellation_rates = (cancelled / total_shifts).fillna(0)
        
        # Reshape into a pivot table
        pivot_df = cancellation_rates.unstack(level='Time Block').fillna(0)
        
        # Order weekdays correctly
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_df = pivot_df.reindex(weekday_order, axis=0)
        
        return pivot_df

    # Original UI elements
    st.subheader("📊 Combined Volunteer Table")
    st.dataframe(df_all, use_container_width=True)

    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "spring_volunteer_data.csv", "text/csv")

    def detect_dropoffs(df, window_days=14):
        
        # Filter for rows where attendance is Yes and date is not 'Logged hours'
        attended_df = df[(df['Attended'] == 'Yes') & (df['Date'] != 'Logged hours')].copy()
        
        # Skip if no valid data
        if attended_df.empty:
            return pd.DataFrame(columns=['Name', 'Last Shift Date', 'Days Since', 'Weeks Since'])
        
        # Convert dates to datetime objects
        def parse_date(date_str):
            formats = ['%m/%d/%Y', '%Y-%m-%d', '%A, %B %d', '%A %m/%d', '%m/%d']
            for fmt in formats:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    # If parsed year is 1900 (i.e., year wasn't in the string), replace with current year
                    if parsed.year == 1900:
                        parsed = parsed.replace(year=datetime.now().year)
                    return parsed.date()
                except ValueError:
                    continue
            return None
        
        attended_df['Parsed Date'] = attended_df['Date'].apply(parse_date)
        
        # Drop rows with unparseable dates
        attended_df = attended_df.dropna(subset=['Parsed Date'])
        
        if attended_df.empty:
            return pd.DataFrame(columns=['Name', 'Last Shift Date', 'Days Since', 'Weeks Since'])
        
        # Find the latest shift date for each volunteer
        latest_shifts = attended_df.groupby('Name')['Parsed Date'].max().reset_index()
        latest_shifts.columns = ['Name', 'Last Shift Date']
        
        # Calculate days since last shift
        today = datetime.now().date()
        
        # Convert Last Shift Date column to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(latest_shifts['Last Shift Date']):
            # Calculate days since directly without using .dt accessor
            latest_shifts['Days Since'] = latest_shifts['Last Shift Date'].apply(lambda x: (today - x).days)
        else:
            latest_shifts['Days Since'] = (today - latest_shifts['Last Shift Date'].dt.date).dt.days
        
        latest_shifts['Weeks Since'] = (latest_shifts['Days Since'] / 7).round(1)
        
        # Format Last Shift Date for display
        latest_shifts['Last Shift Date'] = latest_shifts['Last Shift Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        
        # Filter for volunteers who haven't attended in the specified window
        dropoffs = latest_shifts[latest_shifts['Days Since'] > window_days]
        
        # Sort by days since last shift, descending
        dropoffs = dropoffs.sort_values('Days Since', ascending=False)
        
        return dropoffs

    def compute_reputation(df, name_to_total_hours=None):
        # Filter for attended shifts only and create a copy to avoid SettingWithCopyWarning
        attended_df = df[df['Attended'] == 'Yes'].copy()
        
        # Extract week numbers for consecutive week calculation
        def extract_week_num(week_str):
            if not week_str or not isinstance(week_str, str):
                return None
            match = re.search(r'Week\s*(\d+)', week_str)
            return int(match.group(1)) if match else None
        attended_df['Week Num'] = attended_df['Week'].apply(extract_week_num)
        
        # Calculate metrics for each volunteer
        metrics = []
        for name, group in attended_df.groupby('Name'):
            # Use total hours from mapping if available
            total_hours = 0.0
            if name_to_total_hours and name in name_to_total_hours:
                total_hours = name_to_total_hours[name]
            else:
                total_hours = 0.0
            # Count distinct shift categories
            shift_types_count = group['Shift Category'].nunique()
            # Calculate consecutive weeks
            consecutive_weeks = 0
            if not group['Week Num'].isna().all():
                weeks = sorted(group['Week Num'].dropna().unique())
                if weeks:
                    current_streak = 1
                    max_streak = 1
                    for i in range(1, len(weeks)):
                        if weeks[i] == weeks[i-1] + 1:
                            current_streak += 1
                        else:
                            current_streak = 1
                        max_streak = max(max_streak, current_streak)
                    consecutive_weeks = max_streak
            # Calculate reputation score
            reputation_score = int(total_hours // 2) + consecutive_weeks + shift_types_count
            metrics.append({
                'Name': name,
                'Total Hours': round(total_hours, 1),
                'Consecutive Weeks': consecutive_weeks,
                'Shift Types Count': shift_types_count,
                'Reputation Score': reputation_score
            })
        # Convert to DataFrame and sort by reputation score
        reputation_df = pd.DataFrame(metrics)
        if not reputation_df.empty:
            reputation_df = reputation_df.sort_values('Reputation Score', ascending=False).reset_index(drop=True)
        return reputation_df


with tab3:
    def parse_date_to_weekday(date_str, default_year=2025):
    # Handle missing or NaN values gracefully
        if pd.isna(date_str):
            return None
        # Extract month/day with regex
        match = re.search(r'(\d{1,2}/\d{1,2})', date_str)
        if match:
            md = match.group(1)  # e.g. "4/11"
            try:
                dt = datetime.strptime(f"{default_year}/{md}", "%Y/%m/%d")
                return dt.strftime('%A')  # Full weekday name
            except ValueError:
                return None  # If parsing fails
        else:
            return None  # If no date found
        
    df_all['Weekday'] = df_all['Date'].apply(parse_date_to_weekday)
                                             # Seperating the dataframes
    df_m = df_all[df_all["Sign-Up Type"] == "Mobile Pantry"]
    df_fr = df_all[df_all["Sign-Up Type"] == "Food Recovery"]
    df_p = df_all[df_all["Sign-Up Type"] == "Weekly Sign-Up"]
     # Define a helper function to plot counts per week
    def plot_weekly_counts(df, title):
        # Group by week and count
        weekly_counts = df.groupby('Week').size().reset_index(name='Count')

        # Extract numeric week number for proper sorting
        weekly_counts['WeekNum'] = weekly_counts['Week'].str.extract(r'(\d+)').astype(int)
        weekly_counts = weekly_counts.sort_values('WeekNum')

        # Plot
        fig, ax = plt.subplots(figsize=(10,6))
        sns.lineplot(data=weekly_counts, x='Week', y='Count', marker='o', ax=ax)
        ax.set_title(title)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    plot_weekly_counts(df_m, 'Volunteer Shifts Covered for Mobile Pantry per Week')
    plot_weekly_counts(df_fr, 'Volunteer Shifts Covered for Food Recovery per Week')
    plot_weekly_counts(df_p, 'Volunteer Shifts Covered for Weekly Sign-Up per Week')

    

# Apply to your dataframe column, creating a new 'Weekday' column

# Day & Time Analysis
with tab4:
    st.header("📅 Day & Time Analysis")
    st.write("Analysis of shifts by day and time block, including cancellation rates (Cancellations/Total Shifts)). Ex: A value of 0 means no one has canceled their shifts on that day.")
    
    # Compute day & time analysis
    day_time_df = compute_day_time(df_all)
    
    # Display the analysis table
    st.dataframe(day_time_df, use_container_width=True)
    
    # Bar chart of total shifts by time block with color coding by sign-up type
    st.subheader("Shifts Distribution by Time Block")
    
    # Group data by Time Block and Sign-Up Type
    signup_type_counts = df_all.groupby(['Time Block', 'Sign-Up Type']).size().reset_index(name='Count')
    
    # Get unique time blocks and sort by total count
    time_block_totals = signup_type_counts.groupby('Time Block')['Count'].sum().sort_values(ascending=False)
    sorted_time_blocks = time_block_totals.index.tolist()
    
    # Filter for non-empty time blocks
    signup_type_counts = signup_type_counts[signup_type_counts['Time Block'].notna() & 
                                           (signup_type_counts['Time Block'] != '')]
    
    # Create pivot table for stacked bar chart
    pivot_df = signup_type_counts.pivot_table(
        index='Time Block', 
        columns='Sign-Up Type', 
        values='Count',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reorder rows by total count
    pivot_df = pivot_df.reindex(sorted_time_blocks)
    
    # Define colors for different sign-up types
    colors = {
        'Weekly Sign-Up': 'skyblue',
        'Food Recovery': 'tomato',
        'Mobile Pantry': 'lightgreen',
        'Attendance Hours': 'gold',
        'Misc': 'purple'
    }
    
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each sign-up type as a separate bar segment
    bottom = np.zeros(len(pivot_df))
    for sign_up_type in pivot_df.columns:
        if sign_up_type in pivot_df.columns:
            color = colors.get(sign_up_type, 'gray')
            ax.bar(pivot_df.index, pivot_df[sign_up_type], bottom=bottom, 
                  label=sign_up_type, color=color)
            bottom += pivot_df[sign_up_type].values
    
    # Customize the chart
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Sign-Up Type')
    plt.xlabel('Time Block')
    plt.ylabel('Number of Shifts')
    
    st.pyplot(fig)
    
    

# Cancellation Patterns
with tab5:
    st.header("❌ Cancellation Patterns")
    st.write("This is a heatmap of cancellation rates by weekday and time block. Darker shades of red indicates that more people are canceling their shifts on that specific day.")
    
    # Compute cancellation patterns
    cancellation_pattern = compute_cancellation_pattern(df_all)
    
    # Display the pattern as a table
    st.dataframe(cancellation_pattern.round(2), use_container_width=True)
    
    # Create a heatmap
    st.subheader("Cancellation Rate Heatmap")
    if not cancellation_pattern.empty and not cancellation_pattern.isna().all().all():
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(cancellation_pattern, annot=True, cmap="YlOrRd", fmt=".2f", ax=ax)
        plt.title("Cancellation Rates by Weekday and Time Block")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough data to generate a meaningful heatmap.")

# Drop-off Volunteers
with tab6:
    st.header("🚶 Drop-off Volunteers")
    st.write("This enables us to identify volunteers who have not participated in any shifts for a specified number of days, allowing us to reach out and check in with them.")
    
    # Configurable window for drop-off detection
    window_days = st.slider(
        "Days without shifts to consider as drop-off:", 
        min_value=7, 
        max_value=30, 
        value=14,
        step=1
    )
    
    # Compute drop-offs
    dropoffs = detect_dropoffs(df_all, window_days=window_days)
    
    if dropoffs.empty:
        st.info(f"No volunteers found who have been inactive for more than {window_days} days.")
    else:
        st.write(f"Volunteers who haven't attended shifts in {window_days} days:")
        st.dataframe(dropoffs, use_container_width=True)

# Volunteer Reputation
with tab7:
    st.header("🌟 Volunteer Reputation")
    st.write("These reputations scores allow us to identify which members are contributing the most!")
    st.markdown("Reputation Score is Total Hours/2 + Number of Consecutive Weeks + Types of Shifts")
    # Compute reputation
    reputation_df = compute_reputation(df_all, name_to_total_hours)
    
    if reputation_df.empty:
        st.info("No sufficient data to calculate reputation scores.")
    else:
        # Display the reputation table
        st.dataframe(reputation_df, use_container_width=True)
        
        # Bar chart of top 10 volunteers by reputation
        st.subheader("Top 10 Volunteers by Reputation Score")
        top_10 = reputation_df.head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(top_10['Name'], top_10['Reputation Score'], color='green')
        ax.set_xlabel('Reputation Score')
        ax.set_ylabel('Volunteer Name')
        ax.invert_yaxis()  # To have highest score at the top
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download button for reputation data
        csv_reputation = reputation_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Reputation Data", 
            csv_reputation, 
            "volunteer_reputation.csv", 
            "text/csv"
        )
