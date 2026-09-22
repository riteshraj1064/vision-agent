import os
import asyncio
import json
from typing import Optional

from dotenv import load_dotenv

# Load Stream keys from the parent repo .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
# Local .env can override values
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from getstream.models import MemberRequest
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.agents.events import (
    AgentTurnEndedEvent,
    UserTranscriptEvent,
)
from vision_agents.core.instructions import Instructions
from vision_agents.core.llm.events import LLMResponseFinalEvent
from vision_agents.plugins import getstream, gemini


# ============================================================
# CONSTANTS
# ============================================================

AGENT_USER_ID = "ai-teacher"

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ko": "Korean",
}


# ============================================================
# EDUCATIONAL SAFETY
# ============================================================

EDUCATIONAL_BOUNDARIES = """
EDUCATIONAL PURPOSE ONLY
=========================
You are an AI language teacher. Your purpose is language learning,
conversation practice, vocabulary, pronunciation, grammar, reading,
listening, role-play, and other educational activities.

You MUST keep the conversation educational.

Allowed:
- Language learning and translation
- Grammar and vocabulary
- Pronunciation and speaking practice
- Everyday conversation practice
- Travel language
- School/college language
- Workplace and interview language practice
- Role-play for educational practice
- General educational explanations
- Cultural context when relevant to language learning
- User-selected learning topics

Do NOT:
- Provide sexual or erotic content
- Engage in sexual role-play
- Generate romantic/sexual content intended for arousal
- Provide instructions for illegal activity
- Help with weapons, violence, self-harm, or dangerous activities
- Provide instructions for drug use or other dangerous behavior
- Act as a medical, legal, or financial professional
- Encourage harmful, abusive, hateful, or discriminatory behavior
- Ask for passwords, authentication codes, private keys, financial credentials,
  or other highly sensitive personal information
- Continue a conversation that has moved away from educational language learning

If a user asks for something outside educational language learning:
1. Briefly refuse the off-topic request.
2. Redirect to a safe language-learning alternative.
3. Continue teaching.

Example:
"I can help with language learning, but I can't help with that topic.
We can practice English for a job interview, travel, or daily conversation."
"""


# ============================================================
# GENERAL AI TEACHER PROMPT
# ============================================================

DEFAULT_SYSTEM_PROMPT = f"""
You are a warm, patient, energetic AI language teacher having a
real-time voice conversation with a student.

{EDUCATIONAL_BOUNDARIES}

CORE TEACHING RULES
===================
1. The student chooses what they want to learn or talk about.
2. You do NOT need lesson JSON in free-teacher mode.
3. You may create new educational questions, examples, vocabulary,
   sentences, and exercises based on the student's chosen goal/topic.
4. Keep everything relevant to language learning.
5. Ask ONE main question at a time.
6. Wait for the student's actual spoken answer before evaluating it.
7. Never pretend the student answered.
8. Do not praise an answer before the student answers.
9. Adapt difficulty to the student's demonstrated level.
10. Prefer short, natural spoken responses.
11. Keep the conversation interactive instead of giving long lectures.
12. After teaching something, give the student a chance to practice it.
13. If the student asks a question, answer it first, then continue teaching.
14. Remember the current topic and use it for follow-up questions.
15. If the student says they want a different topic, switch topics naturally.
16. If the student asks to practice instead of explaining, immediately practice.
17. If the student makes a mistake, correct it gently and explain briefly.
18. Do not correct every tiny mistake if doing so would interrupt fluency.
19. For beginners, use simple vocabulary and short sentences.
20. For advanced students, use natural and more challenging language.

LANGUAGE DIRECTION
==================
The application will provide:
- source_language = the language the learner already understands
- target_language = the language the learner is learning

Use the target language for practice.
Use the source language for short explanations, translations,
clarifications, or when the learner clearly needs help.

For Hindi -> English:
- Practice primarily in English.
- You may use Hindi for short explanations.
- Encourage the student to answer in English.

CONVERSATION LOOP
=================
A good teaching loop is:

1. Ask what the student wants to learn, OR use the supplied topic/goal.
2. Ask one manageable question.
3. Listen.
4. Evaluate meaning + grammar + vocabulary.
5. Give a short correction if needed.
6. Give a better/natural version when useful.
7. Ask a related follow-up question.
8. Gradually increase difficulty.
9. Occasionally review words or grammar from the conversation.

Do not turn every response into a test.
Mix:
- conversation
- teaching
- questions
- examples
- corrections
- mini exercises
- role-play
- vocabulary practice

USER CHOICE
===========
The user is allowed to choose:
- topic
- learning goal
- conversation style
- difficulty
- role-play scenario

Examples:
- "I want to practice English for travel."
- "Let's talk about cricket."
- "Teach me restaurant English."
- "I have a job interview."
- "I want beginner English conversation."
- "Correct my grammar while we talk."

If no topic is supplied, start by asking:
"What would you like to practice today?"

Do NOT ask multiple setup questions at once.
Ask one simple question and let the student choose.

CORRECTION STYLE
================
When the student makes a mistake:
- First understand what they meant.
- Do not shame them.
- Give a concise correction.
- Explain the important part.
- Ask them to try again when useful.

Example:
Student: "Yesterday I go market."
Teacher: "Good try. Say: 'Yesterday I went to the market.' Can you say it?"

Do not over-explain unless the student asks for more detail.

VOICE STYLE
===========
- Natural
- Friendly
- Encouraging
- Patient
- Concise
- Human-like
- Educational

Ask one question and then stop speaking.
Do not keep talking after asking a question.
"""


# ============================================================
# HELPERS
# ============================================================

def normalize_language(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback

    value = str(value).strip().lower()

    if not value:
        return fallback

    if "-" in value:
        value = value.split("-")[0]

    if "_" in value:
        value = value.split("_")[0]

    aliases = {
        "english": "en",
        "hindi": "hi",
        "spanish": "es",
        "french": "fr",
        "japanese": "ja",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "korean": "ko",
    }

    return aliases.get(value, value)


def language_name(code: str, fallback: str = "the target language") -> str:
    code = normalize_language(code)

    return LANGUAGE_NAMES.get(code, fallback)


# ============================================================
# STREAM CALL CUSTOM DATA
# ============================================================

async def get_call_custom_data(call) -> dict:
    """
    Correct Vision Agents SDK path:

        response.data.call.custom
    """

    try:
        response = await call.get()

        data = getattr(response, "data", None)

        if data is None:
            print("[agent] response.data is None")
            return {}

        stream_call = getattr(data, "call", None)

        if stream_call is None:
            print("[agent] response.data.call is None")
            return {}

        custom = getattr(stream_call, "custom", None)

        if custom is None:
            print("[agent] response.data.call.custom is None")
            return {}

        if not isinstance(custom, dict):
            try:
                custom = dict(custom)
            except Exception as error:
                print("[agent] Could not convert custom data:", error)
                return {}

        print("[agent] CALL CUSTOM:")
        print(json.dumps(custom, ensure_ascii=False, indent=2, default=str))

        return custom

    except Exception as error:
        print("[agent] Failed to read Stream call custom:", error)
        return {}


async def wait_for_call_custom_data(
    call,
    timeout: float = 20.0,
) -> dict:
    start = asyncio.get_event_loop().time()

    while True:
        custom = await get_call_custom_data(call)

        # We need at least mode/language information.
        if custom.get("mode") or custom.get("source_language") or custom.get("target_language"):
            return custom

        elapsed = asyncio.get_event_loop().time() - start

        if elapsed >= timeout:
            print("[agent] Timed out waiting for call custom data.")
            return {}

        await asyncio.sleep(0.5)


# ============================================================
# FREE TEACHER PROMPT
# ============================================================

def build_free_teacher_prompt(custom: dict) -> str:
    source_code = normalize_language(
        custom.get("source_language"),
        "hi",
    )

    target_code = normalize_language(
        custom.get("target_language"),
        "en",
    )

    source_name = language_name(
        source_code,
        "the learner's known language",
    )

    target_name = language_name(
        target_code,
        "the language being learned",
    )

    topic = str(custom.get("topic") or "").strip()
    goal = str(custom.get("goal") or "").strip()
    difficulty = str(custom.get("difficulty") or "beginner").strip()
    conversation_style = str(
        custom.get("conversation_style") or "guided_conversation"
    ).strip()

    topic_instruction = (
        f'The learner selected topic: "{topic}".'
        if topic
        else "No topic was selected yet. Ask the learner what they want to practice."
    )

    goal_instruction = (
        f'The learner selected goal: "{goal}".'
        if goal
        else "No specific learning goal was selected yet."
    )

    return f"""
FREE AI TEACHER MODE
====================

This is NOT a database lesson.

The learner can choose the topic and direction of the conversation.

LANGUAGE
--------
Source language: {source_name} ({source_code})
Target language: {target_name} ({target_code})

The learner is learning {target_name}.

{topic_instruction}
{goal_instruction}

Difficulty: {difficulty}
Conversation style: {conversation_style}

IMPORTANT
---------
You are allowed to create new educational examples, questions,
sentences, vocabulary, and exercises.

However, everything MUST remain within educational language learning.

Do not behave like a general unrestricted chatbot.

FIRST TURN
----------
If a topic/goal is already supplied:
- briefly welcome the learner
- mention the selected topic/goal naturally
- immediately begin a simple exercise or question

If no topic is supplied:
- ask ONE simple question:
  "What would you like to practice today?"
- then wait.

TOPIC CONTROL
-------------
The learner can change the topic at any time.

If the learner says:
"Let's talk about travel."

Switch to travel language practice.

If the learner says:
"I want grammar."

Switch to grammar teaching.

If the learner says:
"Interview."

Switch to job-interview language practice.

Do not force the learner to continue the previous topic.

TEACHING LOOP
-------------
For each learner response:
1. Understand what they intended to say.
2. Decide whether correction is useful.
3. Give a short correction if needed.
4. Explain briefly when helpful.
5. Continue with ONE related question or exercise.

Keep the learner speaking as much as possible.

DIFFICULTY ADAPTATION
---------------------
Start at the supplied difficulty.

If the learner performs well:
- gradually increase sentence complexity
- introduce natural expressions
- ask more open-ended questions

If the learner struggles:
- simplify vocabulary
- shorten sentences
- provide a small hint
- optionally give a model answer
- ask them to try again

DO NOT overwhelm beginners.

ROLE-PLAY
---------
Role-play is allowed when it is educational.

Examples:
- restaurant
- hotel
- airport
- shopping
- classroom
- doctor vocabulary practice
- workplace
- job interview
- travel
- customer service

Keep role-play clearly educational and language-focused.

SAFETY
------
{EDUCATIONAL_BOUNDARIES}

VOICE
-----
Speak naturally.
Keep responses short.
Ask one question at a time.
After asking a question, STOP and wait for the student.
"""


# ============================================================
# CREATE AGENT
# ============================================================

async def create_agent(**kwargs) -> Agent:
    # IMPORTANT:
    # User MUST come from vision_agents.core.
    # Do NOT import User from getstream.models.

    return Agent(
        edge=getstream.Edge(),
        llm=gemini.Realtime(),
        agent_user=User(
            name="AI Teacher",
            id=AGENT_USER_ID,
        ),
        instructions=DEFAULT_SYSTEM_PROMPT,
    )


# ============================================================
# END STREAM CALL
# ============================================================

async def end_stream_call(call) -> None:
    try:
        result = call.end()

        if asyncio.iscoroutine(result):
            await result

        print("[agent] Stream call ended.")

    except Exception as error:
        print("[agent] Failed to end Stream call:", error)


# ============================================================
# JOIN CALL
# ============================================================

async def join_call(
    agent: Agent,
    call_type: str,
    call_id: str,
    **kwargs,
) -> None:

    print("==================================================")
    print("[agent] FREE/LESSON AI TEACHER SESSION")
    print("[agent] call_type:", call_type)
    print("[agent] call_id:", call_id)
    print("==================================================")

    # --------------------------------------------------------
    # Create Vision Agents call
    # --------------------------------------------------------

    call = await agent.create_call(
        call_type,
        call_id,
    )

    # --------------------------------------------------------
    # Read Stream custom data
    # --------------------------------------------------------

    custom = await wait_for_call_custom_data(call)

    mode = str(custom.get("mode") or "lesson").strip().lower()

    print("[agent] mode:", mode)

    # --------------------------------------------------------
    # Build the correct mode prompt
    # --------------------------------------------------------

    if mode == "free_teacher":
        source_language = normalize_language(
            custom.get("source_language"),
            "hi",
        )

        target_language = normalize_language(
            custom.get("target_language"),
            "en",
        )

        prompt = build_free_teacher_prompt(custom)

        print("[agent] Starting FREE TEACHER mode.")
        print("[agent] source:", source_language)
        print("[agent] target:", target_language)
        print("[agent] topic:", custom.get("topic"))
        print("[agent] goal:", custom.get("goal"))
        print("[agent] difficulty:", custom.get("difficulty"))

    else:
        # Backward compatibility:
        # Your current lesson flow still works.
        source_language = normalize_language(
            custom.get("source_language"),
            "hi",
        )

        target_language = normalize_language(
            custom.get("target_language"),
            "en",
        )

        print("[agent] Starting LESSON mode.")

        # Import the lesson prompt behavior inline so the existing
        # lesson payload continues to work.
        questions = custom.get("questions") or []
        vocabulary = custom.get("vocabulary") or []
        phrases = custom.get("phrases") or []
        topics = custom.get("topics") or []

        lesson_title = custom.get("lesson_title") or "this lesson"

        source_name = language_name(source_language)
        target_name = language_name(target_language)

        lesson_json = json.dumps(
            {
                "lesson_id": custom.get("lesson_id"),
                "lesson_title": lesson_title,
                "lesson_description": custom.get("lesson_description"),
                "unit_title": custom.get("unit_title"),
                "course_title": custom.get("course_title"),
                "source_language": source_language,
                "target_language": target_language,
                "questions": questions,
                "vocabulary": vocabulary,
                "phrases": phrases,
                "topics": topics,
            },
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""
CURRENT LESSON
==============

Use ONLY the following lesson JSON for structured lesson practice.

{lesson_json}

LANGUAGE DIRECTION
==================

Source language: {source_name} ({source_language})
Target language: {target_name} ({target_language})

The learner is learning {target_name}.

Use {source_name} only for short explanations or hints.
The learner should normally answer in {target_name}.

QUESTION RULES
==============

There are {len(questions)} supplied questions.

For every supplied question:
1. Ask one clear question.
2. Wait for the student's actual answer.
3. Evaluate it.
4. If correct, briefly praise and move forward.
5. If wrong, keep the SAME question and give a short hint.
6. Never advance after a wrong answer.

WORD PRACTICE
=============

Practice supplied word/vocabulary items.

SENTENCE PRACTICE
=================

Practice COMPLETE sentences.

If target_text is "I go to school"
and target_translation is "मैं स्कूल जाता हूँ",
the learner should produce:
"I go to school"

Do not reduce a sentence to only one word.

FIELD MEANING
=============

target_text = target-language answer.

target_translation = source-language meaning/help.

NEVER swap these fields.

STRICT SCOPE
============

Do not invent material outside the supplied lesson.

COMPLETION
==========

When all supplied practice items are completed correctly:
- congratulate the learner
- say the lesson is complete
- say goodbye
- do not ask another question
- end the call

{EDUCATIONAL_BOUNDARIES}
"""

    # --------------------------------------------------------
    # Apply mode-specific instructions
    # --------------------------------------------------------

    agent.instructions = Instructions(
        input_text=(
            DEFAULT_SYSTEM_PROMPT
            + "\n\n"
            + prompt
        )
    )

    # --------------------------------------------------------
    # Give agent admin role
    # --------------------------------------------------------

    try:
        await call.update_call_members(
            update_members=[
                MemberRequest(
                    user_id=AGENT_USER_ID,
                    role="admin",
                )
            ]
        )

        print("[agent] Agent admin role granted.")

    except Exception as error:
        print("[agent] Could not set admin role:", error)

    # --------------------------------------------------------
    # Go live
    # --------------------------------------------------------

    try:
        await call.go_live()
        print("[agent] Call is live.")

    except Exception as error:
        print(
            "[agent] go_live failed "
            f"(may be expected for default call type): {error}"
        )

    # --------------------------------------------------------
    # Event state
    # --------------------------------------------------------

    ending_call = False
    stop_event = asyncio.Event()

    # --------------------------------------------------------
    # HARD 2-MINUTE SERVER-SIDE SESSION LIMIT
    # --------------------------------------------------------
    # The frontend also has a 2-minute timer for UX, but this
    # server-side timer is authoritative and protects the limit
    # if the app is backgrounded, frozen, or modified.
    # --------------------------------------------------------

    max_duration = 120

    try:
        max_duration = int(
            custom.get(
                "max_duration_seconds",
                120,
            )
        )
    except Exception:
        max_duration = 120

    max_duration = max(
        1,
        min(max_duration, 120),
    )

    async def enforce_session_timeout():
        nonlocal ending_call

        try:
            await asyncio.sleep(max_duration)

            if ending_call:
                return

            ending_call = True

            print(
                f"[agent] {max_duration}-second session limit reached."
            )

            try:
                await agent.send_custom_event(
                    {
                        "type": "session_timeout",
                        "reason": "two_minute_limit",
                        "duration_seconds": max_duration,
                    }
                )
            except Exception:
                pass

            try:
                await agent.simple_response(
                    text=(
                        "Our two-minute practice session is complete. "
                        "Great work! See you next time."
                    )
                )
            except Exception:
                pass

            # Allow the final audio to play briefly.
            await asyncio.sleep(1.0)

            await end_stream_call(call)

            try:
                await agent.finish()
            except Exception:
                pass

            stop_event.set()

        except asyncio.CancelledError:
            pass
        except Exception as error:
            print(
                "[agent] Session timeout task failed:",
                error,
            )

    timeout_task = asyncio.create_task(
        enforce_session_timeout()
    )

    # --------------------------------------------------------
    # Agent turn ended
    # --------------------------------------------------------

    @agent.events.subscribe
    async def on_agent_turn_ended(
        event: AgentTurnEndedEvent,
    ) -> None:

        try:
            await agent.send_custom_event(
                {
                    "type": "transcript_agent_done",
                    "interrupted": event.interrupted,
                }
            )

        except Exception as error:
            print(
                "[agent] transcript_agent_done error:",
                error,
            )

    # --------------------------------------------------------
    # Final AI transcript
    # --------------------------------------------------------

    @agent.events.subscribe
    async def on_llm_response_final(
        event: LLMResponseFinalEvent,
    ) -> None:

        nonlocal ending_call

        if not event.text:
            return

        try:
            await agent.send_custom_event(
                {
                    "type": "transcript_partial",
                    "speaker": "agent",
                    "text": event.text,
                }
            )

        except Exception as error:
            print(
                "[agent] transcript_partial error:",
                error,
            )

        # ----------------------------------------------------
        # Only automatically end in lesson mode.
        # Free teacher stays alive until the user leaves,
        # or the agent/session is explicitly stopped.
        # ----------------------------------------------------

        if mode != "lesson":
            return

        lower = event.text.lower()

        completion_phrases = [
            "lesson is complete",
            "lesson complete",
            "lesson completed",
            "practice is complete",
            "practice complete",
            "practice completed",
            "all questions completed",
            "all questions are completed",
            "goodbye",
            "good bye",
            "bye bye",
            "see you next time",
        ]

        if (
            not ending_call
            and any(
                phrase in lower
                for phrase in completion_phrases
            )
        ):
            ending_call = True

            print("[agent] Lesson completion detected.")

            try:
                await agent.send_custom_event(
                    {
                        "type": "end_call",
                        "reason": "lesson_completed",
                    }
                )
            except Exception:
                pass

            # Allow goodbye audio to finish.
            await asyncio.sleep(1.0)

            await end_stream_call(call)

            try:
                await agent.finish()
            except Exception:
                pass

            stop_event.set()

    # --------------------------------------------------------
    # User transcript
    # --------------------------------------------------------

    @agent.events.subscribe
    async def on_user_transcript(
        event: UserTranscriptEvent,
    ) -> None:

        if not event.text:
            return

        try:
            await agent.send_custom_event(
                {
                    "type": "transcript_partial",
                    "speaker": "user",
                    "text": event.text,
                }
            )

        except Exception as error:
            print(
                "[agent] user transcript error:",
                error,
            )

    # --------------------------------------------------------
    # Join call
    # --------------------------------------------------------

    async with agent.join(call):

        print("[agent] AI Teacher joined call.")

        # ----------------------------------------------------
        # Wait for student
        # ----------------------------------------------------

        try:
            await agent.wait_for_participant(
                timeout=60.0
            )

        except Exception as error:
            print(
                "[agent] Participant wait failed:",
                error,
            )

        # ----------------------------------------------------
        # Start conversation
        # ----------------------------------------------------

        if mode == "free_teacher":

            topic = str(
                custom.get("topic") or ""
            ).strip()

            goal = str(
                custom.get("goal") or ""
            ).strip()

            target_name = language_name(
                target_language
            )

            if topic:
                opening = (
                    f"Hi! I'm your AI English teacher. "
                    f"Today we'll practice {topic}. "
                    f"Let's start with an easy question."
                )

            elif goal:
                opening = (
                    f"Hi! I'm your AI language teacher. "
                    f"Let's work on {goal}. "
                    f"Let's start with a simple question."
                )

            else:
                opening = (
                    f"Hi! I'm your AI language teacher. "
                    f"What would you like to practice in {target_name} today?"
                )

            try:
                await agent.simple_response(
                    text=opening
                )

                print("[agent] Free teacher conversation started.")

            except Exception as error:
                print(
                    "[agent] Failed to start free teacher conversation:",
                    error,
                )

        else:

            lesson_title = (
                custom.get("lesson_title")
                or "this lesson"
            )

            language = language_name(
                target_language
            )

            try:
                await agent.simple_response(
                    text=(
                        "The student has joined. "
                        f"This is the '{lesson_title}' lesson. "
                        f"The target language is {language}. "
                        "Greet the student briefly, then ask the FIRST "
                        "supplied practice question. "
                        "Ask only one question and stop speaking."
                    )
                )

                print("[agent] First lesson question requested.")

            except Exception as error:
                print(
                    "[agent] Failed to start lesson conversation:",
                    error,
                )

        # ----------------------------------------------------
        # Keep session alive.
        #
        # IMPORTANT:
        # Do NOT call agent.finish() here.
        # ----------------------------------------------------

        try:
            await stop_event.wait()

        except asyncio.CancelledError:
            print("[agent] Call task cancelled.")

        finally:
            if not timeout_task.done():
                timeout_task.cancel()

            try:
                await timeout_task
            except asyncio.CancelledError:
                pass


# ============================================================
# RUNNER
# ============================================================

runner = Runner(
    AgentLauncher(
        create_agent=create_agent,
        join_call=join_call,
    )
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if not os.getenv("STREAM_API_KEY"):
        raise RuntimeError(
            "Missing required environment variable: STREAM_API_KEY"
        )

    if not os.getenv("STREAM_API_SECRET"):
        raise RuntimeError(
            "Missing required environment variable: STREAM_API_SECRET"
        )

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing required environment variable: GOOGLE_API_KEY"
        )

    runner.cli()
