from jsonargparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        default_config_files=[
            "./config/rag.yaml",            # base config — always loaded first
            "./config/llm/llama32_3b.yaml", # LLM settings
            "./config/data/mmlu.yaml",      # default dataset (overridden by --config)
        ]
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    # Controls how much output you see in the terminal.
    # Set to DEBUG for full detail, INFO for normal, WARNING for minimal.
    parser.add_argument("--log_level", type=str, help="DEBUG, INFO, WARNING, ERROR, or CRITICAL")

    # ── LLM settings ──────────────────────────────────────────────────────────
    # These come from config/llm/llama32_3b.yaml automatically.
    # You only need to change them if you switch to a different model.
    parser.add_argument("--llm.base_url", type=str)   # e.g. http://localhost:11434
    parser.add_argument("--llm.name", type=str)        # e.g. llama3
    parser.add_argument("--llm.num_ctx", type=int)     # context window size

    # ── Dataset settings ──────────────────────────────────────────────────────
    # These are overridden when you run:
    #   python simulator.py --config config/data/news.yaml
    # That news.yaml replaces the default mmlu.yaml values below.
    parser.add_argument("--data.load.path", type=str)      # HuggingFace dataset path
    parser.add_argument("--data.load.name", type=str)      # dataset subset name
    parser.add_argument("--data.load.split", type=str)     # train / test / validation
    parser.add_argument("--data.task_type", type=str)      # ogqa or mcqa
    parser.add_argument("--data.topic_path", type=str)     # which field is the topic
    parser.add_argument("--data.question_path", type=str)  # which field is the question
    parser.add_argument("--data.choices_path", type=str)   # which field has choices (mcqa only)
    parser.add_argument("--data.answer_path", type=str)    # which field is the answer
    parser.add_argument("--data.num_samples", type=int)    # how many samples to use

    # ── RAG network core settings ─────────────────────────────────────────────
    # These control how the DRAG network is built and how queries are routed.
    parser.add_argument("--rag.random_seed", type=int)                    # reproducibility seed
    parser.add_argument("--rag.log_every_n_steps", type=int)              # how often to save logs
    parser.add_argument("--rag.test_mode", type=bool)                     # True = only 20 samples
    parser.add_argument("--rag.network_type", type=str)                   # DRAG / CRAG / NoRAG
    parser.add_argument("--rag.num_peers", type=int)                      # number of peers in network
    parser.add_argument("--rag.num_peer_attachments", type=int)           # edges per new node (Barabasi-Albert)
    parser.add_argument("--rag.search_algorithm", type=str)               # TARW / RW / FL
    parser.add_argument("--rag.query_confidence_threshold", type=float)   # minimum confidence to accept answer
    parser.add_argument("--rag.num_query_neighbor", type=int)             # neighbours queried per hop
    parser.add_argument("--rag.query_ttl", type=int)                      # max hops before query dies
    parser.add_argument("--rag.filter_out_topic_ratio", type=float)       # fraction of topics to hide
    parser.add_argument("--rag.filter_out_qa_ratio", type=float)          # fraction of QA pairs to hide
    parser.add_argument("--rag.replication_factor", type=int)             # how many peers store each topic

    # ── DDoS attack settings ──────────────────────────────────────────────────
    # These control the congestion-based DDoS attack simulation.
    # All values are read from rag.yaml — change them there for each experiment.
    #
    # enable_node_attack    : true = run attack, false = baseline only
    # node_attack_type      : always "ddos" for your thesis
    # node_attack_ratio     : fraction of peers targeted per wave (0.3 = 30%)
    # node_attack_iterations: how many attack waves to run
    # attack_strategy       : random / targeted / sequential
    # confidence_threshold  : availability warning threshold
    # target_peers          : [] lets strategy decide, or list specific peer indices
    # ddos_duration         : seconds until a peer recovers from congestion
    # ddos_intensity_min    : lower bound of attack intensity per peer (0.0 to 1.0)
    # ddos_intensity_max    : upper bound of attack intensity per peer (0.0 to 1.0)
    parser.add_argument("--rag.enable_node_attack", type=bool, default=False)
    parser.add_argument("--rag.node_attack_type", type=str, default="none")
    parser.add_argument("--rag.node_attack_ratio", type=float, default=0.3)
    parser.add_argument("--rag.node_attack_iterations", type=int, default=3)
    parser.add_argument("--rag.attack_strategy", type=str, default="random")
    parser.add_argument("--rag.confidence_threshold", type=float, default=0.5)
    parser.add_argument("--rag.target_peers", type=list, default=[])
    parser.add_argument("--rag.ddos_duration", type=float, default=60.0)
    parser.add_argument("--rag.ddos_intensity_min", type=float, default=0.5)
    parser.add_argument("--rag.ddos_intensity_max", type=float, default=1.0)

    # ── Config file flag ───────────────────────────────────────────────────────
    # This is what makes --config work on the command line.
    # When you run: python simulator.py --config config/data/news.yaml
    # jsonargparse reads news.yaml and merges it on top of the defaults above.
    parser.add_argument("--config", action="config")

    cfg = parser.parse_args()
    return cfg