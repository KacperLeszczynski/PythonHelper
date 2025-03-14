import tiktoken

from services.openai_service import OpenAIService, get_openai_service


class SummaryBufferMemory:
    def __init__(self, openai_service: OpenAIService, tokenizer, max_tokens=3500, window_size=5, model="gpt4o-mini"):
        self.history = []
        self.openai_service = openai_service
        self.tokenizer = tokenizer
        self.summary = ""
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.model = model

    def add_interaction(self, query, assistant_output):
        self.history.append({"user": query, "assistant": assistant_output})
        if self.get_token_count() > self.max_tokens:
            self.summarize_history()

    def get_token_count(self):
        text = " ".join([f"{h['user']} {h['assistant']}" for h in self.history])
        return len(self.tokenizer.encode(text))

    def summarize_history(self):
        conversation_text = "\n".join(
            f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history[:-self.window_size]
        )

        summarization_prompt = f"""
        Your task is to summarize the following conversation between a user and an assistant.
        - Focus ONLY on key technical details (technologies, libraries, coding languages, user's project specifics).
        - Omit greetings, general questions, or small talk.
        - Limit your summary to 7-10 concise sentences.

        Conversation:
        {conversation_text}

        Summary:
        """

        messages = [{"role": "user", "content": summarization_prompt}]
        response = self.openai_service.generate(messages=messages, model=self.model, temperature=0.3)

        self.summary = response.choices[0].message.content

        self.history = self.history[-self.window_size:]

    def get_prompt(self, query, context, python_version="3.10"):
        prompt_model = """
        ### Context:
        {context}

        ### Python version: 
        {python_version}

        ** Instructions **
        - If user asks you to generate code and by using context you cannot do it, then generate it on your own
        - If user doesn't ask to generate code and the context does not contain answer for query answer based on your knowledge.
        - If the user's question does not specify Python, rephrase it internally as a Python-related question before answering.
        - If there is a code in your output explain this code to the user step by step
        - Do not answer any other question than about python programming language
        - If topic is complex provide summary at the end of your answer
        - Do not make up any information
        - Provide consise and structured answer

        ### Summary of previous conversation:
        {summary}

        ### Recent conversation history:

        """

        prompt = prompt_model.format(context=context, query=query, python_version=python_version,
                                     summary=self.summary)

        for msg in self.history:
            prompt += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"

        prompt += f"User: {query}\nAssistant:"

        return prompt


tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

summary_buffer_memory = SummaryBufferMemory(
    openai_service=get_openai_service(),
    tokenizer=tokenizer
)

def get_summary_buffer_memory() -> SummaryBufferMemory:
    return summary_buffer_memory