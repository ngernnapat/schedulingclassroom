from generate_planner_content import (
    ChatWrapper,
    ChatWrapperConfig,
    DayPlan,
    PlannerContent,
    Task,
    TimeStamp,
    infer_plan_intent,
)
from planner_enrichment import EnrichmentConfig, _directive_system_prompt


def test_travel_defaults_to_real_itinerary():
    assert infer_plan_intent(
        "travel",
        "Kyoto family trip",
        "Four days visiting temples, markets, and restaurants",
    ) == "itinerary"


def test_explicit_pre_trip_work_stays_preparation():
    assert infer_plan_intent(
        "travel",
        "Prepare for Japan",
        "Apply for a visa, book flights, and create a packing checklist",
    ) == "preparation"


def test_free_form_categories_resolve_execution_intent():
    assert infer_plan_intent("event", "Plan our wedding", "") == "event"
    assert infer_plan_intent("other", "Quarterly leadership meeting", "Agenda, venue, and catering") == "event"
    assert infer_plan_intent("coding", "Build and launch a portfolio app", "") == "project"
    assert infer_plan_intent("other", "เรียนภาษาจีน", "") == "learning"


def test_travel_prompt_prohibits_quiz_and_requires_map_ready_places():
    prompt = ChatWrapper(ChatWrapperConfig())._build_system_prompt(
        "travel",
        intent_type="itinerary",
    )

    assert "Never add quizzes" in prompt
    assert "Google Places enrichment" in prompt
    assert "chronological HH:MM" in prompt


def test_place_enrichment_never_substitutes_an_unnamed_venue():
    prompt = _directive_system_prompt(EnrichmentConfig())

    assert "ONLY when the task text already names a specific" in prompt
    assert "must not silently change the itinerary" in prompt
    assert EnrichmentConfig().max_place_queries == 24


def test_every_plan_type_gets_a_recalculated_budget():
    plan = PlannerContent(
        planName="Team meeting",
        category="other",
        totalDays=1,
        currency="usd",
        totalBudget=999,
        createdAt=TimeStamp(seconds=1, nanoseconds=0),
        days=[
            DayPlan(
                dayNumber=1,
                title="Meeting day",
                summary="Run the quarterly meeting",
                tasks=[
                    Task(text="Book the meeting room and AV", estimatedCost=200),
                    Task(text="Facilitate the team retrospective", estimatedCost=0),
                    Task(text="Order lunch for the attendees", estimatedCost=75.5),
                ],
            )
        ],
    )

    assert plan.currency == "USD"
    assert plan.totalBudget == 275.5


if __name__ == "__main__":
    test_travel_defaults_to_real_itinerary()
    test_explicit_pre_trip_work_stays_preparation()
    test_free_form_categories_resolve_execution_intent()
    test_travel_prompt_prohibits_quiz_and_requires_map_ready_places()
    test_place_enrichment_never_substitutes_an_unnamed_venue()
    test_every_plan_type_gets_a_recalculated_budget()
    print("plan intent tests passed")
