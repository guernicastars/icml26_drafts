Experiment ID	Experiment Name	Paper & Track	Priority	Compute (GPU-Hrs)	Lead Owner	Status	Outcome / Notes
P1	Information cascade in multi-agent LLM debate	Paper 2, Track P	Must-have	~80	Eugene (Lead)	Not Started	
P3	Homeostatic deception via honest aggregation	Paper 2, Track P	Must-have	~50	Eugene (Lead)	Not Started	
P5	Emergent collusion under shared objective	Paper 2, Track P	Must-have	~100	Eugene (Lead)	Not Started	
E1	Evidential-weight decay under adversarial pressure	Paper 3, Track E	Must-have	~180	Eugene / Vasiliy (Lead)	Not Started	
E4	Elicitation of evidential weight from the model itself	Paper 3, Track E	Must-have	~60	Eugene / Vasiliy (Lead)	Not Started	
E5	Second-order calibration of self-reported honesty	Paper 3, Track E	Must-have	~40	Eugene / Vasiliy (Lead)	Not Started	
P2	Monoculture failure under distribution shift	Paper 2, Track P	Standard	~60	Eugene (Lead)	Not Started	
P4	Chain-of-LLMs faithfulness erosion	Paper 2, Track P	Standard	~40	Eugene (Lead)	Not Started	
P6	Aggregation-induced Goodhart	Paper 2, Track P	Standard	~80	Eugene (Lead)	Not Started	
E2	Probe reliability vs OOD distance	Paper 3, Track E	Standard	~30	Eugene / Vasiliy (Lead)	Not Started	
E3	Meta-probe for probe reliability	Paper 3, Track E	Standard	~40	Eugene / Vasiliy (Lead)	Not Started	
A2	Interventional introspection - the gold-standard test	Paper 1, Track A	Tier 1	~50	Vasiliy	Not Started	
B1	Discriminant-validity stress test - the six-way split	Paper 1, Track B	Tier 1	~150	Eugene	Not Started	
C1	Fine-tune against the probe - Goodhart at mech-interp layer	Paper 1, Track C	Tier 1	~60	Tamara	Not Started	
C2	Fine-tune for CoT faithfulness against auditor	Paper 1, Track C	Tier 1	~200	Tamara	Not Started	
D1	Triangle-of-disagreement across models - headline figure	Paper 1, Track D	Tier 1	~100	Joint	Not Started	
A1	Nisbett-Wilson replication for LLMs	Paper 1, Track A	Tier 2	~200	Vasiliy	Not Started	
A4	Faithfulness-under-paraphrase	Paper 1, Track A	Tier 2	~80	Vasiliy	Not Started	
B2	Probe-probe agreement on OOD deception	Paper 1, Track B	Tier 2	~30	Eugene	Not Started	
B3	SAE decomposition of the honesty direction	Paper 1, Track B	Tier 2	~80	Eugene	Not Started	
C3	Adversarial probe-evasion via prefix tuning	Paper 1, Track C	Tier 2	~40	Tamara	Not Started	
A3	Introspection-behaviour dissociation under pressure	Paper 1, Track A	Tier 3	~100	Vasiliy	Not Started	
A5	Multi-lingual introspection consistency	Paper 1, Track A	Tier 3	~60	Vasiliy	Not Started	
B4	Cross-model probe transfer	Paper 1, Track B	Tier 3	~40	Eugene	Not Started	
B5	Fine-tuning-stage stability	Paper 1, Track B	Tier 3	~40	Eugene	Not Started	
C4	Measurement-induced drift over iterative deployment	Paper 1, Track C	Tier 3	~150	Tamara	Not Started	
D2	Scaling of the paradox within a family	Paper 1, Track D	Tier 3	~120	Joint	Not Started	
							
							
							
Module	Description	Owner	Timeline	Status	Notes		
Module 1	Model harness (Transformers, vLLM, TransformerLens)	Maxim	Week 1-2	Not Started	Highest priority.		
Module 2	Probe library (RepE, Azaria-Mitchell, Burns CCS, Mass-mean)	Maxim	Week 1-2	Not Started	Highest priority alongside Mod 1.		
Module 3	Evaluation datasets (MASK, Berger-style, Six-way, etc.)	Arina	Week 1-2	Not Started	Berger-style set and six-way split are critical.		
Module 4	Interventional toolkit (Patching, prefix tuning, LoRA, PPO/DPO)	Maxim	Week 1-2	Not Started	Requires TRL / OpenRLHF for C2.		
Module 5	Multi-agent harness (Paper 2 specific; sequential, parallel, multi-party)	Eugene	Before Wk 8	Not Started	Reuse NExT-Game infrastructure where possible.		
Module 6	Judge & scoring infrastructure (LLM-judge prompts)	Eugene / Joint	Week 1-2	Not Started	Pre-register prompts; budget ~$1000 API.		
							