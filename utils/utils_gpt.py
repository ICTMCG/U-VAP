import re
from openai import OpenAI

# Attribute prior setting. You can manually change as you want.
template = {"color": "\"red\", \"green\"", "object": "\"dog\"\, \"plane\""}

client = OpenAI(
    api_key="sk-...",  # your openai API
)

def chat_with_gpt(init_prompt, attri, n_p):
    pattern = r'\b(\w+)\b\s+(?=\b' + re.escape(attri) + r'\b)'
    pseudo = re.findall(pattern, init_prompt)

    sys_text = "Hi chatgpt, you are a imaginative sentence rewriter. Given an sentence and a special attribute word in it, you will change the word by another new counterfactual attribute word. This new word should be different form special attribute word apparently. "
    if attri not in template:
        text = f"There is a sentence: \"{init_prompt}\". Please change the \"{pseudo}\" before the \"{attri}\" with {n_p} different words to describe different \"{attri}\", and output {n_p} new sentences. Each new word must be different!"
    else:
        text = f"There is a sentence: \"{init_prompt}\". Please change the \"{pseudo}\" before the \"{attri}\" with {n_p} different words to describe different \"{attri}\" (such as {template[attri]}.), and output {n_p} new sentences. Each new word must be different!"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sys_text},
            {"role": "user", "content": text},
        ]
    )
    generate_text = response.choices[0].message.content

    pattern = r'\ba\b.*?\bcolor\b'
    fragments = re.findall(pattern, generate_text, re.DOTALL)
    
    return fragments
