// Fixed-set evaluation config (dev / test_normal) for online-GRPO checkpoints and base.
// Same simplified ReAct agent + prompt as the rollout config, but greedy (temperature 0)
// and a fixed seed. Parametrised by env (appworld evaluates jsonnet with ext_vars=os.environ):
//   EVAL_MODEL_NAME  model name to request: the LoRA name (checkpoint) or the base's served name
//   EVAL_DATASET     dev | test_normal | test_challenge
//   VLLM_PORT        port of the vLLM server (MODEL_SERVER_URL must also be set; see eval_ckpts.py)
local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_react_code_agent",
            "model_config": {
                "client_name": "openai",
                "api_type": "chat_completions",
                "base_url": "http://localhost:" + std.extVar("VLLM_PORT") + "/v1",
                "api_key_env_name": "NO_API_KEY",
                "name": std.extVar("EVAL_MODEL_NAME"),
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 100,
                "max_completion_tokens": 2048,
                "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "input_cache_write": 0.0, "output": 0.0},
                "retry_after_n_seconds": 5,
                "use_cache": false,
                "max_retries": 20,
            },
            "appworld_config": {"random_seed": 100, "raise_on_extra_parameters": true},
            "logger_config": {"color": false, "verbose": false},
            "usage_tracker_config": {"max_cost_overall": 1000, "max_cost_per_task": 10, "max_output_tokens_per_task": 200000},
            "prompt_file_path": experiment_prompts_path + "/react_code_agent/instructions.txt",
            "ignore_multiple_calls": true,
            "max_prompt_length": null,
            "max_output_length": null,
            "max_steps": 50,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": std.extVar("EVAL_DATASET"),
    },
}
