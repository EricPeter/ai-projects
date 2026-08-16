import anthropic
from .memory import short_term, long_term
from .memory.compressor import compress_turns, extract_facts

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


def chat(session_id: str, user_id: str, message: str) -> str:
    long_term_memories = long_term.recall(user_id, message, k=5)
    recent_turns = short_term.get_turns(session_id)

    if len(recent_turns) >= 8:
        summary = compress_turns(recent_turns[:-4])
        compressed_turns = [{"role": "user", "content": f"[Earlier summary: {summary}]"}, *recent_turns[-4:]]
    else:
        compressed_turns = recent_turns

    system_parts = ["You are a helpful assistant with persistent memory."]
    if long_term_memories:
        facts = "\n".join(f"- {f}" for f in long_term_memories)
        system_parts.append(f"\nWhat you know about this user:\n{facts}")

    messages = [*compressed_turns, {"role": "user", "content": message}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="\n".join(system_parts),
        messages=messages,
    )
    answer = response.content[0].text

    short_term.add_turn(session_id, "user", message)
    short_term.add_turn(session_id, "assistant", answer)

    new_facts = extract_facts([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
    for fact in new_facts:
        long_term.store(user_id, fact)

    return answer
