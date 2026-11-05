"""
Consolidated Curriculum Dataset & Provisional Schedule for KTU CST414 Deep Learning
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

import datetime
from curriculum_module1 import MODULE_1_DAYS
from curriculum_module2 import MODULE_2_DAYS
from curriculum_module3 import MODULE_3_DAYS
from curriculum_module4 import MODULE_4_DAYS
from curriculum_module5 import MODULE_5_DAYS

# Merge all 36 days
ALL_36_DAYS = MODULE_1_DAYS + MODULE_2_DAYS + MODULE_3_DAYS + MODULE_4_DAYS + MODULE_5_DAYS

# Verify count
assert len(ALL_36_DAYS) == 36, f"Expected 36 days, got {len(ALL_36_DAYS)}"

# PROVISIONAL ACADEMIC CALENDAR DATES MAPPING
# Note: Academic Calendar was not found in project folder.
# This is a clearly labelled PROVISIONAL schedule mapped to 3 periods/week (Mon, Wed, Fri)
# across standard KTU S7 semester window, accounting for Kerala state holidays.

PROVISIONAL_START_DATE = datetime.date(2025, 8, 18) # Monday

# Kerala Public Holidays & Academic Breaks (Provisional 2025-26 Odd Semester)
PROVISIONAL_BLOCKED_DATES = {
    datetime.date(2025, 8, 27): "Sree Krishna Jayanti",
    datetime.date(2025, 9, 3): "First Onam",
    datetime.date(2025, 9, 4): "Thiruvonam",
    datetime.date(2025, 9, 5): "Third Onam",
    datetime.date(2025, 9, 8): "Onam Holiday Extension",
    datetime.date(2025, 9, 21): "Sree Narayana Guru Samadhi",
    datetime.date(2025, 10, 1): "Mahanavami",
    datetime.date(2025, 10, 2): "Gandhi Jayanti",
    datetime.date(2025, 10, 20): "Deepavali",
    datetime.date(2025, 10, 22): "Internal Series Test 1 (Day 1)",
    datetime.date(2025, 10, 24): "Internal Series Test 1 (Day 2)",
    datetime.date(2025, 12, 10): "Internal Series Test 2 (Day 1)",
    datetime.date(2025, 12, 12): "Internal Series Test 2 (Day 2)"
}

def generate_provisional_schedule():
    teaching_days = [0, 2, 4] # Monday (0), Wednesday (2), Friday (4)
    day_periods = {
        0: "Period 2 (10:00 AM - 11:00 AM)",
        2: "Period 4 (01:30 PM - 02:30 PM)",
        4: "Period 1 (09:00 AM - 10:00 AM)"
    }
    
    current_date = PROVISIONAL_START_DATE
    schedule = []
    day_idx = 0
    
    while day_idx < 36:
        # Check if current day of week is Mon, Wed, Fri
        if current_date.weekday() in teaching_days:
            # Check if date is blocked/holiday
            if current_date in PROVISIONAL_BLOCKED_DATES:
                reason = PROVISIONAL_BLOCKED_DATES[current_date]
                # Log holiday
            else:
                day_data = ALL_36_DAYS[day_idx]
                dow_name = current_date.strftime("%A")
                period_str = day_periods[current_date.weekday()]
                
                # Module name mapping
                mod_names = {
                    1: "Module 1 - Neural Networks",
                    2: "Module 2 - Deep Learning",
                    3: "Module 3 - Convolutional Neural Networks",
                    4: "Module 4 - Recurrent Neural Networks",
                    5: "Module 5 - Application Areas"
                }
                
                entry = {
                    "day_number": day_idx + 1,
                    "date_str": current_date.strftime("%Y-%m-%d"),
                    "date_display": current_date.strftime("%d-%b-%Y"),
                    "day_name": dow_name,
                    "period": period_str,
                    "module": day_data["module"],
                    "module_name": mod_names[day_data["module"]],
                    "topic_id": day_data["topic_id"],
                    "title": day_data["title"],
                    "planned_hours": 1,
                    "status": "Completed" if day_idx < 12 else "Scheduled",
                    "co": day_data["co"],
                    "blooms": day_data["blooms"],
                    "data": day_data
                }
                schedule.append(entry)
                day_idx += 1
        
        current_date += datetime.timedelta(days=1)
        
    return schedule

PROVISIONAL_SCHEDULE = generate_provisional_schedule()

print(f"Loaded {len(ALL_36_DAYS)} curriculum days.")
print(f"Generated provisional schedule from {PROVISIONAL_SCHEDULE[0]['date_display']} to {PROVISIONAL_SCHEDULE[-1]['date_display']}.")

