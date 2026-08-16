from crewai import Agent
from crewai_tools import SerperDevTool

# Use SerperDevTool if SERPER_API_KEY is set, else skip web search
try:
    search_tool = SerperDevTool()
    tools = [search_tool]
except Exception:
    tools = []


def researcher() -> Agent:
    return Agent(
        role="Research Analyst",
        goal="Gather comprehensive, accurate information on the given topic from multiple angles",
        backstory=(
            "You are a meticulous researcher who finds authoritative sources, "
            "synthesises diverse viewpoints, and surfaces key facts and statistics."
        ),
        tools=tools,
        verbose=True,
    )


def writer() -> Agent:
    return Agent(
        role="Technical Writer",
        goal="Transform research notes into a clear, well-structured report",
        backstory=(
            "You write concise, engaging reports that present complex information "
            "accessibly, with a logical structure and clear executive summary."
        ),
        verbose=True,
    )


def fact_checker() -> Agent:
    return Agent(
        role="Fact Checker",
        goal="Verify claims in the draft report and flag uncertain or unsupported statements",
        backstory=(
            "You are a skeptical editor who reads carefully, checks each factual claim, "
            "and marks anything uncertain or unverified with [UNCERTAIN] and a brief explanation."
        ),
        verbose=True,
    )
