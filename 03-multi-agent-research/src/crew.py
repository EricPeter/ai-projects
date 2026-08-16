from crewai import Crew, Process

from .agents import researcher, writer, fact_checker
from .tasks import research_task, write_task, fact_check_task
from .audit import log


def run_research(topic: str) -> str:
    log("system", "start", topic, "pipeline initiated")

    r_agent = researcher()
    w_agent = writer()
    fc_agent = fact_checker()

    tasks = [
        research_task(topic, r_agent),
        write_task(w_agent),
        fact_check_task(fc_agent),
    ]

    crew = Crew(
        agents=[r_agent, w_agent, fc_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    output = str(result)

    log("system", "complete", topic, output[:500])
    return output
