// AppWorld experiment config for online-GRPO rollouts (grpo_online/rollout.py).
// Tracked copy: grpo_online/appworld_configs/ ; installed copy (gitignored):
// appworld/experiments/configs/rollout/online/qwen35_9b.jsonnet (launch_vllm.sh syncs it).
//
// Reconstructed from the upstream `simplified_react_code_agent` config: the
// lm_calls.jsonl fixture in grpo/tests/fixtures (schema + first prompt) matches this
// agent with prompts/react_code_agent/instructions.txt, and trajectories cap at 50 steps.
// Per-seed sampling params come from the environment set by rollout.run_rollout
// (appworld evaluates jsonnet with ext_vars = os.environ).
local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local seed = std.parseInt(std.extVar("ROLLOUT_SEED"));
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_react_code_agent",
            "model_config": {
                "client_name": "openai",
                "api_type": "chat_completions",
                // Concrete URL from VLLM_PORT (set by rollout.py). The agent still calls
                // fill_model_server_url on it, which requires MODEL_SERVER_URL in the env
                // (rollout.py sets that too) but leaves a placeholder-free URL unchanged.
                "base_url": "http://localhost:" + std.extVar("VLLM_PORT") + "/v1",
                "api_key_env_name": "NO_API_KEY",
                // Must equal OnlineConfig.gen_model_name (= the LoRA name under vLLM).
                "name": "Qwen/Qwen3.5-9B",
                "temperature": std.parseJson(std.extVar("ROLLOUT_TEMPERATURE")),
                "top_p": std.parseJson(std.extVar("ROLLOUT_TOP_P")),
                "seed": seed,
                "max_completion_tokens": 2048,
                "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "input_cache_write": 0.0, "output": 0.0},
                "retry_after_n_seconds": 5,
                "use_cache": false,
                "max_retries": 20,
            },
            "appworld_config": {
                "random_seed": seed,
                "raise_on_extra_parameters": true,
            },
            "logger_config": {"color": false, "verbose": false},
            "usage_tracker_config": {
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "max_output_tokens_per_task": 200000,
            },
            "prompt_file_path": experiment_prompts_path + "/react_code_agent/instructions.txt",
            "ignore_multiple_calls": true,
            "max_prompt_length": null,
            "max_output_length": null,
            "max_steps": 50,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": std.extVar("ROLLOUT_DATASET"),
    },
}
