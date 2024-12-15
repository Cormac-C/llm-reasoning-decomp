# Returns a list of dictionaries with the question and answer for instruction-tuned training
def chat_format_qa_instance(qa_instance, assistant_role="assistant", user_role="user"):
    messages = []
    if qa_instance["question"]:
        messages.append({"role": user_role, "content": qa_instance["question"]})
    if qa_instance["answer"]:
        messages.append({"role": assistant_role, "content": qa_instance["answer"]})
    return messages


# Returns a big string with the question and answer for LM training
def lm_format_qa_instance(
    qa_instance, question_marker="### Question", answer_marker="### Answer:"
):
    formatted = f"{question_marker} {qa_instance['question']}\n{answer_marker} {qa_instance['answer']}"
    return formatted


# Few-shot prompt constructor in text format
def lm_create_fewshot_prompt(
    query,
    examples,
    question_marker="### Question",
    answer_marker="### Answer:",
):
    prompt = ""
    for example in examples:
        prompt += (
            lm_format_qa_instance(example, question_marker, answer_marker) + "\n\n"
        )

    prompt += lm_format_qa_instance(
        {"question": query, "answer": ""}, question_marker, answer_marker
    )
    return prompt


# Few-shot prompt constructor in chat format
def chat_create_fewshot_prompt(
    qa_instance, examples, assistant_role="assistant", user_role="user"
):
    prompt = []
    for example in examples:
        prompt += chat_format_qa_instance(
            example, assistant_role=assistant_role, user_role=user_role
        )

    prompt += chat_format_qa_instance(
        qa_instance,
        assistant_role=assistant_role,
        user_role=user_role,
    )
    return prompt
