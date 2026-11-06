# Acknowledgments and component provenance

We provide the NeuroMem memory framework, including its consolidation controller, support-path retrieval, and delayed credit assignment.

- Our EHRSQL task-agent interface and tabular tools are adapted from [EHRAgent](https://github.com/wshi83/EhrAgent/tree/7dfcaf36c230d0b987a16aabe64b8d3024ccd8a6), commit `7dfcaf36c230d0b987a16aabe64b8d3024ccd8a6`. Please also cite Shi et al., [EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records](https://aclanthology.org/2024.emnlp-main.1245/). The benchmark prompt resource is obtained separately from this commit.
- Obtain [EHRSQL](https://github.com/glee4810/EHRSQL), the matching EHRAgent benchmark inputs, and [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) from their respective distributors and follow their terms. We do not distribute clinical data, model weights, or generated experiment outputs.

## Licensing

This notice records attribution; it does not grant a software license. The EHRAgent revision identified above does not contain an explicit code license. Permission to redistribute the adapted code and the license for this repository remain to be confirmed before public redistribution.
