import anthropic


def compress_turns(turns: list[dict]) -> str:
    if not turns:
        return ""
    client = anthropic.Anthropic()
    transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Summarise this conversation in 3-5 sentences, preserving key facts and decisions:\n\n{transcript}"
        }],
    )
    return msg.content[0].text


def extract_facts(turns: list[dict]) -> list[str]:
    if not turns:
        return []
    client = anthropic.Anthropic()
    transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns[-4:])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"Extract 0-3 specific facts about the user from this conversation. "
                f"Output one fact per line, or 'NONE' if there are no facts to extract.\n\n{transcript}"
            )
        }],
    )
    lines = [l.strip() for l in msg.content[0].text.splitlines() if l.strip() and l.strip() != "NONE"]
    return lines[:3]
