from typing import Any, cast

from transformers import AutoModelForCausalLM, AutoTokenizer

# cspell:ignore Qwen
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = cast(Any, AutoModelForCausalLM.from_pretrained(model_name))


def get_fallback_answer(question: str) -> str:
    q = question.lower()

    if "aws" in q and ("ban" in q or "banned" in q or "around" in q or "avoid" in q):
        return (
            "If you are asking about AWS, the safest approach is to use AWS services legally and within the platform's terms. "
            "Do not try to bypass limits or abuse accounts. AWS offers services like EC2, S3, Lambda, and IAM for legitimate cloud work."
        )

    if "aws service" in q or "what are aws" in q or "aws services" in q:
        return (
            "AWS services include EC2 for computing, S3 for storage, Lambda for serverless functions, IAM for access control, "
            "and RDS for managed databases."
        )

    if "data" in q and ("what is" in q or "define" in q):
        return (
            "Data is information that is collected, stored, and processed to support decisions, applications, and business operations. "
            "In modern systems, data is often managed in databases or data lakes instead of older manual systems."
        )

    if "replace" in q and "data management" in q:
        return (
            "A modern data platform can replace older data management systems by centralizing storage, improving access, and enabling analytics."
        )

    if "python" in q:
        return (
            "Python is a popular programming language used for automation, web development, data analysis, machine learning, and cloud automation. "
            "It is known for its simple syntax and large ecosystem of libraries."
        )

    return (
        "I can help with AWS, cloud services, data concepts, and Python. "
        "Try asking about AWS services, data definition, Python, or how cloud systems work."
    )


def ask_llm(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that gives accurate and detailed answers."},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        answer = text.split("assistant")[-1].strip() if "assistant" in text else text

        if answer and len(answer) > 3:
            return answer
    except Exception:
        pass

    return get_fallback_answer(question)