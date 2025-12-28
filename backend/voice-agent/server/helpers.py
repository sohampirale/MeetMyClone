
class StatementJudgeContextFilter(FrameProcessor):
    """Extracts recent user messages and constructs an LLMMessagesFrame for the classifier LLM.

    This processor takes the OpenAILLMContextFrame from the main conversation context,
    extracts the most recent user messages, and creates a simplified LLMMessagesFrame
    for the statement classifier LLM to determine if the user has finished speaking.
    """

    def __init__(self, notifier: BaseNotifier, **kwargs):
        super().__init__(**kwargs)
        self._notifier = notifier

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        # Just treat an LLMMessagesFrame as complete, no matter what.
        if isinstance(frame, LLMMessagesFrame):
            await self._notifier.notify()
            return

        # Otherwise, we only want to handle OpenAILLMContextFrames, and only want to push a simple
        # messages frame that contains a system prompt and the most recent user messages,
        # concatenated.
        if isinstance(frame, OpenAILLMContextFrame):
            # Take text content from the most recent user messages.
            messages = frame.context.messages
            # logger.debug(f"Processing context messages: {messages}")

            user_text_messages = []
            last_assistant_message = None
            for message in reversed(messages):
                role = get_message_field(message, "role")
                # logger.debug(f"Processing message with role: {role}")

                if role != "user":
                    if role == "assistant" or role == "model":
                        last_assistant_message = message
                        # logger.debug(f"Found assistant/model message: {message}")
                    break

                text = get_message_text(message)
                # logger.debug(f"Extracted user message text: {text}")
                if text:
                    user_text_messages.append(text)

            # If we have any user text content, push an LLMMessagesFrame
            if user_text_messages:
                user_message = " ".join(reversed(user_text_messages))
                # logger.debug(f"Final user message: {user_message}")
                messages = [
                    glm.Content(role="user", parts=[glm.Part(text=CLASSIFIER_SYSTEM_INSTRUCTION)])
                ]
                if last_assistant_message:
                    assistant_text = get_message_text(last_assistant_message)
                    # logger.debug(f"Assistant message text: {assistant_text}")
                    if assistant_text:
                        messages.append(
                            glm.Content(role="assistant", parts=[glm.Part(text=assistant_text)])
                        )
                messages.append(glm.Content(role="user", parts=[glm.Part(text=user_message)]))
                # logger.debug(f"Pushing classifier messages: {messages}")
                await self.push_frame(LLMMessagesFrame(messages))
            # else:
            # logger.debug("No user text messages found to process")
            return

        # Fallback: for any frames not otherwise handled, forward them.
        await self.push_frame(frame, direction)



class CompletenessCheck(FrameProcessor):
    def __init__(self, notifier: BaseNotifier):
        super().__init__()
        self._notifier = notifier

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and frame.text == "YES":
            logger.debug("!!! Completeness check YES")
            await self.push_frame(UserStoppedSpeakingFrame())
            await self._notifier.notify()
        elif isinstance(frame, TextFrame) and frame.text == "NO":
            logger.debug("!!! Completeness check NO")
        else:
            await self.push_frame(frame, direction)



class OutputGate(FrameProcessor):
    def __init__(self, *, notifier: BaseNotifier, start_open: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._gate_open = start_open
        self._frames_buffer = []
        self._notifier = notifier

    def close_gate(self):
        self._gate_open = False

    def open_gate(self):
        self._gate_open = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            if isinstance(frame, StartFrame):
                await self._start()
            if isinstance(frame, (EndFrame, CancelFrame)):
                await self._stop()
            if isinstance(frame, StartInterruptionFrame):
                self._frames_buffer = []
                self.close_gate()
            await self.push_frame(frame, direction)
            return

        # Don't block function call frames
        if isinstance(frame, (FunctionCallInProgressFrame, FunctionCallResultFrame)):
            await self.push_frame(frame, direction)
            return

        # Ignore frames that are not following the direction of this gate.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if self._gate_open:
            await self.push_frame(frame, direction)
            return

        self._frames_buffer.append((frame, direction))

    async def _start(self):
        self._frames_buffer = []
        self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        await self.cancel_task(self._gate_task)

    async def _gate_task_handler(self):
        while True:
            try:
                await self._notifier.wait()
                self.open_gate()
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
            except asyncio.CancelledError:
                break

