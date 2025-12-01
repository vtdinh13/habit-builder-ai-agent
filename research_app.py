import asyncio
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from pydantic_ai.messages import FunctionToolCallEvent, ModelRequest, ModelResponse, TextPart

from research_agent import SearchResultResponse, create_search_agent


@st.cache_resource(show_spinner=False)
def load_agent():
    """Instantiate and cache the multipurpose agent across reruns."""
    return create_search_agent()


def run_agent(
    agent,
    question: str,
    message_history: List[Any],
    event_handler,
):
    async def _run():
        return await agent.run(
            user_prompt=question,
            message_history=message_history,
            event_stream_handler=event_handler,
        )

    return asyncio.run(_run())


def parse_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def make_tool_logger(
    lines: List[str], placeholder: st.delta_generator.DeltaGenerator
):
    tool_labels = {
        "search_embeddings": "Searching Elasticsearch..",
        "search_web": "Searching the web...",
        "get_page_content": "Fetching content...",
    }

    async def _logger(ctx, event):
        async def _walk(ev):
            if hasattr(ev, "__aiter__"):
                async for nested in ev:
                    await _walk(nested)
                return

            if isinstance(ev, FunctionToolCallEvent):
                tool_name = ev.part.tool_name
                if tool_name == "final_result":
                    return
                args = parse_args(getattr(ev.part, "args", None))

                description = f"- `{tool_name}`"
                query = args.get("query")
                if query:
                    description += f" · **query:** {query}"
                url = args.get("url")
                if url:
                    description += f" · **url:** {url}"

                lines.append(description)
                placeholder.markdown("\n".join(lines))

        await _walk(event)

    return _logger


def build_summary_messages(question: str, summary: str) -> List[Any]:
    """Create compact user/assistant messages from the latest exchange."""
    request = ModelRequest.user_text_prompt(question.strip())
    response = ModelResponse(parts=[TextPart(summary.strip())])
    return [request, response]


SUMMARY_SECTION_LIMIT = 3
SUMMARY_CHARS_PER_SECTION = 240
SUMMARY_FALLBACK_CHARS = 600


def build_abbreviated_summary(
    output: Optional[SearchResultResponse], fallback_text: str
) -> str:
    """Create a compact summary for message history to reduce token usage."""
    if output:
        lines: List[str] = []
        description = (output.description or "").strip()
        if description:
            lines.append(description)
        for section in output.sections[:SUMMARY_SECTION_LIMIT]:
            snippet = (section.content or "").strip()
            if not snippet:
                continue
            snippet = " ".join(snippet.split())
            if len(snippet) > SUMMARY_CHARS_PER_SECTION:
                cutoff = snippet[:SUMMARY_CHARS_PER_SECTION].rsplit(" ", 1)[0]
                snippet = cutoff + "..."
            heading = (section.heading or "Section").strip()
            lines.append(f"{heading}: {snippet}")
        if lines:
            return "\n".join(lines)

    fallback = " ".join(fallback_text.split())
    if len(fallback) > SUMMARY_FALLBACK_CHARS:
        fallback = fallback[:SUMMARY_FALLBACK_CHARS].rsplit(" ", 1)[0] + "..."
    return fallback


HISTORY_WINDOW = 3


st.set_page_config(page_title="Multipurpose Research Agent", page_icon="🧠", layout="wide")
st.title("🧠 Multipurpose Research Agent")
st.markdown(
    "Ask about sleep, fitness, neuroscience, motivation, or general health. "
    "The agent rewrites your question, runs a single vector search across the Huberman Lab archive, "
    "and consults curated web sources when you request fresh information."
)

with st.sidebar:
    st.header("Session Controls")
    st.caption("Reinitialize the agent if you update environment variables or tool credentials.")
    if st.button("Reload Agent"):
        load_agent.clear()
        st.success("Agent cache cleared.")

    clear_chat = st.button("Clear Conversation")
    if st.button("Stop loading", type="primary"):
        st.warning("Stopping response...")
        st.stop()
    st.divider()
    st.markdown("**Notes**")
    st.markdown(
        "- Responses cite podcast segments or web sources.\n"
        "- Web access is only used after an initial vector search.\n"
        "- Educational use; not affiliated with the Huberman Lab."
    )

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if clear_chat:
    st.session_state["chat_history"] = []
    st.session_state["message_history"] = []
    st.rerun()

for exchange in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.markdown(exchange["question"])
    with st.chat_message("assistant"):
        tool_activity = exchange.get("tool_activity") or []
        if tool_activity:
            st.markdown("**Tool activity**")
            st.markdown("\n".join(tool_activity))
        st.markdown(exchange["answer"])


user_question = st.chat_input(
    "Ask a question about sleep, motivation, neuroscience, fitness, or performance."
)

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    agent = load_agent()
    with st.chat_message("assistant"):
        tool_container = st.container()
        tool_container.markdown("**Tool activity**")
        tool_placeholder = tool_container.empty()
        tool_lines: List[str] = []

        tool_logger = make_tool_logger(tool_lines, tool_placeholder)
        answer_placeholder = st.empty()

        status_placeholder = st.empty()
        status_placeholder.info("Generating response...")
        try:
            history = st.session_state["message_history"]
            recent_history = history[-HISTORY_WINDOW:] if HISTORY_WINDOW else history
            result = run_agent(
                agent,
                user_question.strip(),
                recent_history,
                tool_logger,
            )
        except Exception as exc:
            status_placeholder.empty()
            tool_placeholder.empty()
            st.error(f"Something went wrong: {exc}")
        else:
            status_placeholder.empty()
            output = getattr(result, "output", None)
            if hasattr(output, "format_response"):
                answer_md = output.format_response()
            elif hasattr(output, "model_dump"):
                answer_md = json.dumps(output.model_dump(), indent=2)
            else:
                answer_md = str(output)

            answer_placeholder.markdown(answer_md)
            st.session_state["chat_history"].append(
                {
                    "question": user_question,
                    "answer": answer_md,
                    "tool_activity": list(tool_lines),
                }
            )
            try:
                structured_output = (
                    output if isinstance(output, SearchResultResponse) else None
                )
                summary_text = build_abbreviated_summary(structured_output, answer_md)
                compact_messages = build_summary_messages(user_question, summary_text)
                st.session_state["message_history"].extend(compact_messages)
            except Exception:
                st.warning(
                    "Unable to store the latest conversation context for follow-up questions."
                )

if not st.session_state["chat_history"]:
    st.info("Start by asking about sleep, performance, or neuroscience.")
