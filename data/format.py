# Returns a list of dictionaries with the question and answer for instruction-tuned training
def chat_format_qa_instance(qa_instance, assistant_role="assistant", user_role="user"):
    # Expects qa dictionary with "question" and "answer"
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
    # Expects qa dictionary with "question" and "answer"
    formatted = f"{question_marker} {qa_instance['question']}\n{answer_marker} {qa_instance['answer']}"
    return formatted
