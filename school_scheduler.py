from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value, LpStatus
import pandas as pd
import json
from ortools.sat.python import cp_model
import plotly.graph_objects as go


def hour_to_timerange_label(hour):
    start = 8 + (hour - 1)
    end = start + 1
    return f"{start:02d}:00-{end:02d}:00"

def create_timetable_for_teacher(teacher_schedule, all_schedule, day_names, time_labels, days):
    """Create a timetable for a specific teacher, handling multiple teachers in homeroom periods"""
    timetable = pd.DataFrame("", index=[day_names[d] for d in days], columns=time_labels)
    
    # First, fill in regular teaching assignments
    for _, row in teacher_schedule.iterrows():
        day = day_names[row["Day"]]
        period = hour_to_timerange_label(row["Hour"])
        grade = row["Grade"]
        
        # For homeroom periods (1 and 8), check if there are other teachers
        if row["Hour"] in [1, 8]:
            # Find all teachers teaching this grade at this time
            other_teachers = all_schedule[
                (all_schedule["Grade"] == grade) &
                (all_schedule["Day"] == row["Day"]) &
                (all_schedule["Hour"] == row["Hour"]) &
                (all_schedule["Teacher"] != row["Teacher"])
            ]["Teacher"].tolist()
            
            if other_teachers:
                # Combine current teacher with others
                all_teachers = [row["Teacher"]] + other_teachers
                timetable.loc[day, period] = f"{grade}\n({', '.join(sorted(all_teachers))})"
            else:
                timetable.loc[day, period] = f"{grade}\n({row['Teacher']})"
        else:
            # Regular period - just show grade
            timetable.loc[day, period] = grade
            
    return timetable

def create_timetable_for_grade(grade_schedule, day_names, time_labels, days):
    """Create a timetable for a specific grade"""
    timetable = pd.DataFrame("", index=[day_names[d] for d in days], columns=time_labels)
    
    # Group by Day and Hour to get all teachers for each time slot
    grouped = grade_schedule.groupby(['Day', 'Hour'])['Teacher'].agg(lambda x: ', '.join(sorted(x))).reset_index()
    
    for _, row in grouped.iterrows():
        day = day_names[row["Day"]]
        period = hour_to_timerange_label(row["Hour"])
        teachers = row["Teacher"]
        timetable.loc[day, period] = teachers
            
    return timetable


class SchoolScheduler:
    def __init__(self):
        self.model = None
        self.solver = None
        self.schedule_df = None
        self.homeroom_df = None
        self.params = None
        self.enable_pe_constraints = True
        self.homeroom_mode = 2

    def get_inputs(self,
        n_teachers=13,
        grades=["P1", "P2", "P3", "P4", "P5", "P6", "M1", "M2", "M3"],
        pe_teacher="T13",
        pe_grades=["P4", "P5", "P6", "M1", "M2", "M3"],
        pe_day=3,
        n_pe_periods=6,
        start_hour=8,
        n_hours=8,
        lunch_hour=5,
        days_per_week=5,
        enable_pe_constraints=True,
        homeroom_mode=2
    ):
        self.enable_pe_constraints = enable_pe_constraints
        self.homeroom_mode = homeroom_mode
        try:
            # self._validate_inputs(n_teachers, grades, pe_teacher, pe_grades, pe_day,
            #                       n_pe_periods, start_hour, n_hours, lunch_hour, days_per_week)

            self.params = {
                'n_teachers': n_teachers,
                'grades': grades,
                'pe_teacher': pe_teacher,
                'pe_grades': pe_grades,
                'pe_day': pe_day,
                'n_pe_periods': n_pe_periods,
                'start_hour': start_hour,
                'n_hours': n_hours,
                'lunch_hour': lunch_hour,
                'days_per_week': days_per_week,
                'teachers': [f"T{i+1}" for i in range(n_teachers)],
                'days': list(range(1, days_per_week + 1)),
                'day_names': {i: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i-1] for i in range(1, days_per_week + 1)},
                'hours': list(range(1, n_hours + 1)),
                'non_pe_grades': [g for g in grades if g not in pe_grades],
                'time_labels': [hour_to_timerange_label(h) for h in range(1, n_hours + 1)],
                'grade_to_number': {
                    "": 0,    # Empty (no class)
                    "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5,
                    "P6": 6, "M1": 7, "M2": 8, "M3": 9
                },
                'colors': [
                    [0.0, "white"],         # 0 (empty) → white
                    [0.1, "lightblue"],     # 1 → P1
                    [0.2, "skyblue"],       # 2 → P2
                    [0.3, "dodgerblue"],    # 3 → P3
                    [0.4, "steelblue"],     # 4 → P4
                    [0.5, "deepskyblue"],   # 5 → P5
                    [0.6, "cornflowerblue"],# 6 → P6
                    [0.7, "lightpink"],     # 7 → M1
                    [0.8, "salmon"],        # 8 → M2
                    [0.9, "tomato"],        # 9 → M3
                    [1.0, "orangered"]      # fill to top
                ]
            }
            return True

        except Exception as e:
            print(f"Error in input validation: {str(e)}")
            return False

    def set_pe_constraints_enabled(self, enabled: bool):
        self.enable_pe_constraints = enabled

    def set_homeroom_mode(self, mode: int):
        assert mode in [0, 1, 2], "homeroom_mode must be 0, 1, or 2"
        self.homeroom_mode = mode

    def get_model(self):
        p = self.params
        model = cp_model.CpModel()
        x = {}
        for t in p['teachers']:
            for g in p['grades']:
                for d in p['days']:
                    for h in p['hours']:
                        if h != p['lunch_hour']:
                            x[t, g, d, h] = model.NewBoolVar(f"x_{t}_{g}_{d}_{h}")

        homeroom_grade = {}
        if self.homeroom_mode > 0:
            for t in p['teachers']:
                for g in p['grades']:
                    homeroom_grade[t, g] = model.NewBoolVar(f"hr_{t}_{g}")

        self.vars = {'x': x, 'homeroom_grade': homeroom_grade}
        self._add_constraints(model)
        self.model = model
        return True

    def _add_constraints(self, model):
        p = self.params
        x = self.vars['x']
        homeroom_grade = self.vars['homeroom_grade']
        teachers = p['teachers']
        grades = p['grades']
        days = p['days']
        hours = p['hours']
        lunch_hour = p['lunch_hour']
        pe_teacher = p['pe_teacher']
        pe_day = p['pe_day']
        n_hours = p['n_hours']
        pe_grades = p['pe_grades']
        non_pe_grades = p['non_pe_grades']
        n_pe_periods = p['n_pe_periods']

        for g in grades:
            for d in days:
                for h in hours:
                    if h != lunch_hour:
                        model.Add(sum(x[t, g, d, h] for t in teachers) == 1)

        for t in teachers:
            for d in days:
                for h in hours:
                    if h != lunch_hour:
                        model.Add(sum(x[t, g, d, h] for g in grades) <= 1)

        if self.homeroom_mode > 0:
            for t in teachers:
                model.Add(sum(homeroom_grade[t, g] for g in grades) <= 1)

            for g in grades:
                model.Add(sum(homeroom_grade[t, g] for t in teachers) >= 1)

            for t in teachers:
                for g in grades:
                    for d in days:
                        if self.homeroom_mode == 2:
                            model.Add(x[t, g, d, 1] <= homeroom_grade[t, g])
                            model.Add(x[t, g, d, n_hours] <= homeroom_grade[t, g])
                        elif self.homeroom_mode == 1:
                            model.Add(x[t, g, d, n_hours] <= homeroom_grade[t, g])

            for g in grades:
                for d in days:
                    if self.homeroom_mode == 2:
                        model.Add(sum(x[t, g, d, 1] for t in teachers) >= 1)
                        model.Add(sum(x[t, g, d, n_hours] for t in teachers) >= 1)
                    elif self.homeroom_mode == 1:
                        model.Add(sum(x[t, g, d, n_hours] for t in teachers) >= 1)

        if self.enable_pe_constraints and n_pe_periods > 0:
            # PE class can be adaptive across days
            pe_day_set = [pe_day]  # Make it a list to allow flexibility

            for g in pe_grades:
                model.Add(sum(x[pe_teacher, g, d, h] for d in pe_day_set for h in hours if h != lunch_hour) == 1)

            for g in non_pe_grades:
                model.Add(sum(x[pe_teacher, g, d, h] for d in days for h in hours if h != lunch_hour) == 0)

            model.Add(sum(x[pe_teacher, g, d, h] for g in grades for d in days for h in hours if h != lunch_hour) == n_pe_periods)

        for t in teachers:
            for g in grades:
                for d in days:
                    for h in range(1, n_hours):
                        if h != lunch_hour and h + 1 != lunch_hour:
                            model.Add(x[t, g, d, h] + x[t, g, d, h + 1] <= 1)

    def set_pe_constraints_enabled(self, enabled: bool):
        self.enable_pe_constraints = enabled
        
    def get_solution(self):
        """Solve the model and process results"""
        try:
            if not self.model:
                raise ValueError("Model not created. Call get_model first.")
                
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 300
            status = solver.Solve(self.model)
            
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                print(f"Solver status: {solver.StatusName(status)}")
                
                # Process schedule
                data = []
                p = self.params
                x = self.vars['x']
                
                for t in p['teachers']:
                    for g in p['grades']:
                        for d in p['days']:
                            for h in p['hours']:
                                if h != p['lunch_hour'] and solver.Value(x[t, g, d, h]) == 1:
                                    data.append({
                                        "Teacher": t,
                                        "Grade": g,
                                        "Day": d,
                                        "Hour": h,
                                        "DayName": p['day_names'][d],
                                        "TimeSlot": f"{p['start_hour'] + h - 1:02d}:00-{p['start_hour'] + h:02d}:00"
                                    })

                self.schedule_df = pd.DataFrame(data)
                
                # Process homeroom assignments
                homeroom_data = []
                homeroom_grade = self.vars['homeroom_grade']
                
                for t in p['teachers']:
                    if t != p['pe_teacher']:
                        for g in p['grades']:
                            if solver.Value(homeroom_grade[t, g]) == 1:
                                homeroom_data.append({"Teacher": t, "Grade": g})
                
                self.homeroom_df = pd.DataFrame(homeroom_data)


                # Get all homeroom assignments
                homeroom_assignments = []
                for t in p['teachers']:
                    if t != "T13":  # Skip PE teacher
                        for g in p['grades']:
                            if solver.Value(homeroom_grade[t, g]) == 1:
                                homeroom_assignments.append({"Teacher": t, "Grade": g})

                # Create a copy of the existing schedule
                self.extended_schedule = self.schedule_df.copy()


                # Add missing homeroom duties based on mode
                new_rows = []
                for assignment in homeroom_assignments:
                    teacher = assignment["Teacher"]
                    grade = assignment["Grade"]
                    
                    for d in p['days']:
                        if self.homeroom_mode == 2:
                            check_hours = [1, 8]
                        elif self.homeroom_mode == 1:
                            check_hours = [8]
                        else:
                            check_hours = []

                        for h in check_hours:
                            if not any((self.extended_schedule["Teacher"] == teacher) & 
                                    (self.extended_schedule["Grade"] == grade) & 
                                    (self.extended_schedule["Day"] == d) & 
                                    (self.extended_schedule["Hour"] == h)):
                                new_rows.append({
                                    "Teacher": teacher,
                                    "Grade": grade,
                                    "Day": d,
                                    "Hour": h,
                                    "DayName": p['day_names'][d],
                                    "TimeSlot": f"{self.params['start_hour'] + h - 1:02d}:00-{self.params['start_hour'] + h:02d}:00",
                                    "IsHomeroom": "Yes"
                                })


                # Add IsHomeroom column to original rows
                self.extended_schedule["IsHomeroom"] = self.extended_schedule.apply(
                    lambda row: "Yes" if row["Hour"] in [1, 8] and 
                    any(ha["Teacher"] == row["Teacher"] and ha["Grade"] == row["Grade"] 
                        for ha in homeroom_assignments) 
                    else "No", axis=1
                )

                # Add new rows for non-teaching homeroom duties
                if new_rows:
                    self.extended_schedule = pd.concat([
                        self.extended_schedule, 
                        pd.DataFrame(new_rows)
                    ], ignore_index=True)

                # Sort and display the complete schedule
                print("\nComplete Schedule (including non-teaching homeroom duties):")
                print(self.extended_schedule.sort_values(["Grade", "Day", "Hour"]))
                
                            
                return True
                
            else:
                print(f"No feasible solution. Solver status: {solver.StatusName(status)}")
                return False
                
        except Exception as e:
            print(f"Error solving model: {str(e)}")
            return False
    
    def get_plotting(self):
        """Create visualizations for the schedule"""
        try:
            if self.extended_schedule is None:
                raise ValueError("No solution available. Call get_solution first.")
            
            # Create teacher-based visualization
            fig_teacher = self._plot_teacher_schedule()
            
            # Create grade-based visualization
            fig_grade = self._plot_grade_schedule()
            
            return fig_teacher, fig_grade
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            return None, None

    def _plot_teacher_schedule(self):
        teacher_options = sorted(self.extended_schedule["Teacher"].unique())
        p = self.params
        fig = go.Figure()

        for teacher in teacher_options:
            df_teacher = self.extended_schedule[self.extended_schedule["Teacher"] == teacher]
            timetable = create_timetable_for_teacher(df_teacher, self.schedule_df, p['day_names'], p['time_labels'], p['days'])

            # Create numeric values for heatmap
            z = []
            text = []
            for day in timetable.index:
                z_row = []
                text_row = []
                for period in timetable.columns:
                    cell_value = timetable.loc[day, period]
                    if cell_value == "":
                        z_row.append(0)
                        text_row.append("")
                    else:
                        # Extract grade from the cell (it's either just the grade or grade with teachers)
                        grade = cell_value.split('\n')[0] if '\n' in cell_value else cell_value
                        z_row.append(p['grade_to_number'][grade])
                        text_row.append(cell_value)
                z.append(z_row)
                text.append(text_row)

            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=timetable.columns,
                    y=timetable.index,
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorscale=p['colors'],
                    showscale=False,
                    hoverinfo="text",
                    zmin=0,
                    zmax=9
                )
            )

        fig.data[0].visible = True
        for i in range(1, len(teacher_options)):
            fig.data[i].visible = False

        buttons = []
        for i, teacher in enumerate(teacher_options):
            buttons.append(
                dict(
                    method="update",
                    label=teacher,
                    args=[
                        {"visible": [j == i for j in range(len(teacher_options))]},
                        {"title": f"Schedule for {teacher}"}
                    ]
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    x=1.1,
                    y=1,
                    xanchor="left",
                    yanchor="top"
                )
            ],
            title=dict(
                text="Schedule for Teacher",
                x=0.5,
                xanchor="center"
            ),
            yaxis_title="Day",
            xaxis=dict(
                tickmode="array",
                tickvals=p['time_labels'],
                ticktext=p['time_labels'],
                side="top",
                showgrid=True,
                gridcolor='black'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='black',
                autorange="reversed"
            ),
            width=1200,
            height=600,
            margin=dict(l=50, r=150, t=100, b=50)
        )

        return fig

    def _plot_grade_schedule(self):
        p = self.params
        grade_options = p['grades']  # Use the existing grades list
        fig = go.Figure()

        for grade in grade_options:
            df_grade = self.extended_schedule[self.extended_schedule["Grade"] == grade]
            timetable = create_timetable_for_grade(df_grade, p['day_names'], p['time_labels'], p['days'])

            # Create numeric values for heatmap
            z = []
            text = []
            for day in timetable.index:
                z_row = []
                text_row = []
                for period in timetable.columns:
                    cell_value = timetable.loc[day, period]
                    if cell_value == "":
                        z_row.append(0)
                        text_row.append("")
                    else:
                        z_row.append(1)  # Use 1 for filled cells
                        text_row.append(cell_value)
                z.append(z_row)
                text.append(text_row)

            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=timetable.columns,
                    y=timetable.index,
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorscale=[[0, "white"], [1, "lightgreen"]],
                    showscale=False,
                    hoverinfo="text",
                    zmin=0,
                    zmax=1
                )
            )

        fig.data[0].visible = True
        for i in range(1, len(grade_options)):
            fig.data[i].visible = False

        buttons = []
        for i, grade in enumerate(grade_options):
            buttons.append(
                dict(
                    method="update",
                    label=grade,
                    args=[
                        {"visible": [j == i for j in range(len(grade_options))]},
                        {"title": f"Schedule for Grade {grade}"}
                    ]
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    x=1.1,
                    y=1,
                    xanchor="left",
                    yanchor="top"
                )
            ],
            title=dict(
                text="Schedule by Grade",
                x=0.5,
                xanchor="center"
            ),
            yaxis_title="Day",
            xaxis=dict(
                tickmode="array",
                tickvals=p['time_labels'],
                ticktext=p['time_labels'],
                side="top",
                showgrid=True,
                gridcolor='black'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='black',
                autorange="reversed"
            ),
            width=1200,
            height=600,
            margin=dict(l=50, r=150, t=100, b=50)
        )

        return fig