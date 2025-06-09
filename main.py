from school_scheduler import *

def create_school_schedule(
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
    enable_pe_constraints=False,
    homeroom_mode=1
):
    scheduler = SchoolScheduler()
    scheduler.set_pe_constraints_enabled(enable_pe_constraints)
    scheduler.set_homeroom_mode(homeroom_mode)

    if not scheduler.get_inputs(
        n_teachers, grades, pe_teacher, pe_grades,
        pe_day, n_pe_periods, start_hour, n_hours,
        lunch_hour, days_per_week,
        enable_pe_constraints=enable_pe_constraints,
        homeroom_mode=homeroom_mode
    ):
        return None

    scheduler.get_model()
    if not scheduler.get_solution():
        return None, None, None, None, None
    
    fig_teacher, fig_grade = scheduler.get_plotting()
    print(fig_teacher)
    
    return scheduler.schedule_df, scheduler.homeroom_df, scheduler.extended_schedule, fig_teacher, fig_grade