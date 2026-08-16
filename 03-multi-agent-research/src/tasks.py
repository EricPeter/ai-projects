from crewai import Task
from crewai import Agent


def research_task(topic: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Research the following topic thoroughly: '{topic}'. "
            "Gather key facts, statistics, recent developments, and multiple perspectives. "
            "Produce structured research notes with source references where possible."
        ),
        expected_output="Structured research notes covering background, key facts, recent trends, and perspectives.",
        agent=agent,
    )


def write_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Using the research notes provided, write a clear 3-5 section report. "
            "Include: Executive Summary, Background, Key Findings, Analysis, and Conclusion."
        ),
        expected_output="A well-structured markdown report with all five sections.",
        agent=agent,
    )


def fact_check_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Review the draft report. For every factual claim, assess its verifiability. "
            "Mark uncertain or unverified claims with [UNCERTAIN: <reason>]. "
            "Provide a summary of what was verified and what needs confirmation."
        ),
        expected_output=(
            "The report with [UNCERTAIN] markers added where appropriate, "
            "plus a verification summary at the end."
        ),
        agent=agent,
        human_input=True,  # Pause for human approval before finalising
    )
