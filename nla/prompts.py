"""Prompt templates for AV (chat-formatted) and AR (raw text, suffix-anchored).

The AV's marker token is the position whose input embedding gets replaced by
the scaled activation vector. The AR's last-prompt-token position is where the
value head reads the predicted activation.

Following the paper appendix, the AV is chat-formatted and the AR is not.
"""

AV_USER_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to "
    "describe the semantic content of that activation vector.\n"
    "We will pass the vector enclosed in <concept> tags into your context. "
    "You must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets "
    "describing that vector.\n"
    "Here is the vector:\n"
    "<concept>{marker}</concept>\n"
    "Please provide an explanation."
)

# AR prompt is suffix-anchored: the final token is where we read the prediction.
# `... <summary>` is the paper's convention.
AR_TEMPLATE = "<text>{explanation}</text> <summary>"


def build_av_messages(marker_token: str) -> list[dict]:
    """Return chat messages for the AV. Apply tokenizer.apply_chat_template
    on the result to get token IDs. The marker token will appear once and is
    where the activation vector gets injected (replacing its embedding)."""
    return [{"role": "user", "content": AV_USER_TEMPLATE.format(marker=marker_token)}]


def build_ar_prompt(explanation: str) -> str:
    return AR_TEMPLATE.format(explanation=explanation)
