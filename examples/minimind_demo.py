"""Live demo: qvac-sdk's OpenAI-compatible client against a self-trained local model.

QVAC itself isn't required here — any OpenAI-compatible local server works,
which is what this proves. The model is MiniMind (github.com/jingyaogong/minimind),
a ~64M-param model trained from scratch on a Kaggle T4 (see eva vault:
vault/kaggle/minimind-learn/).

Prerequisites:
    1. Clone and train/serve MiniMind separately (not a qvac-sdk dependency):
       git clone https://github.com/jingyaogong/minimind
       cd minimind/scripts && python3 serve_openai_api.py --weight full_sft --device cpu
    2. This starts an OpenAI-compatible server on http://localhost:8998.

Run:
    python3 examples/minimind_demo.py
"""

from qvac_sdk.openai_compat import QVACOpenAI


def main() -> None:
    client = QVACOpenAI(base_url="http://localhost:8998")

    # MiniMind's server defaults to streaming responses; request non-streaming
    # explicitly so the client returns a parsed ChatCompletion object.
    response = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "你有什么特长？"}],
        stream=False,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
