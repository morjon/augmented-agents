import openai
import re
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate(prompt, use_openai=True):
    """Generates a text completion for a given prompt using OpenAI's API."""
    if use_openai:
        model_engine = "text-davinci-002"
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )
        message = response.choices[0].text

        return message.strip()


def get_rating(x):
    """Extracts a rating from a string."""
    nums = [int(i) for i in re.findall(r"\d+", x)]
    if len(nums) > 0:
        return min(nums)
    else:
        return None


def summarize_simulation(log_output):
    """Summarize a single loop of the simulation using the model."""
    prompt = f"Summarize the simulation loop:\n\n{log_output}"
    response = generate(prompt)

    return response
