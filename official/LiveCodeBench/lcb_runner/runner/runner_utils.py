from LiveCodeBench.lcb_runner.lm_styles import LMStyle, LanguageModel


def build_runner(args, model: LanguageModel):
    if model.model_style == LMStyle.OpenAIChat:
        from LiveCodeBench.lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style in [LMStyle.OpenAIReason, LMStyle.OpenAIReasonPreview]:
        from LiveCodeBench.lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style in [LMStyle.Gemini, LMStyle.GeminiThinking]:
        from LiveCodeBench.lcb_runner.runner.gemini_runner import GeminiRunner

        return GeminiRunner(args, model)
    if model.model_style in [LMStyle.Claude3, LMStyle.Claude3Thinking]:
        from LiveCodeBench.lcb_runner.runner.claude3_runner import Claude3Runner

        return Claude3Runner(args, model)
    if model.model_style == LMStyle.Claude:
        from LiveCodeBench.lcb_runner.runner.claude_runner import ClaudeRunner

        return ClaudeRunner(args, model)
    if model.model_style == LMStyle.MistralWeb:
        from LiveCodeBench.lcb_runner.runner.mistral_runner import MistralRunner

        return MistralRunner(args, model)
    if model.model_style == LMStyle.CohereCommand:
        from LiveCodeBench.lcb_runner.runner.cohere_runner import CohereRunner

        return CohereRunner(args, model)
    if model.model_style == LMStyle.DeepSeekAPI:
        from LiveCodeBench.lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model)
    if model.model_style == LMStyle.DeepSeekAPI:
        from LiveCodeBench.lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model)
    if model.model_style == LMStyle.Grok:
        from LiveCodeBench.lcb_runner.runner.grok_runner import GrokRunner

        return GrokRunner(args, model)
    if model.model_style == LMStyle.TogetherAI:
        from LiveCodeBench.lcb_runner.runner.together_runner import TogetherAIRunner

        return TogetherAIRunner(args, model)
    if "/fireworks/" in model.model_name:
        from LiveCodeBench.lcb_runner.runner.fireworks_runner import FireWorksRunner

        return FireWorksRunner(args, model)
    if model.model_style == LMStyle.GptOssVLLM:
        from LiveCodeBench.lcb_runner.runner.vllm_kflow_runner import VLLMKflowRunner

        return VLLMKflowRunner(args, model)
    if model.model_style == LMStyle.OpenAIReasonAgent:
        from LiveCodeBench.lcb_runner.runner.oai_agent_runner import OAIAgentRunner

        return OAIAgentRunner(args, model)
    elif model.model_style in []:
        raise NotImplementedError(
            f"Runner for language model style {model.model_style} not implemented yet"
        )
    else:
        from LiveCodeBench.lcb_runner.runner.vllm_runner import VLLMRunner

        return VLLMRunner(args, model)
