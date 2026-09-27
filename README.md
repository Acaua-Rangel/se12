# lws — Lossless Weight Streaming

Experimento: acelerar a inferência de LLMs (decode, batch pequeno) lendo pesos
BF16 comprimidos **sem perdas** num formato de taxa fixa (SE12, ~12 bits/peso)
e reconstruindo os valores exatos dentro do kernel, logo antes da multiplicação.

## Setup

```bash
# 1. extraia este pacote na raiz do repositório, depois:
npx https://github.com/Acaua-Rangel/onp-spec-driven.git init --agents claude
#    (o init mantém a constitution.md e o onpspec.config.json deste pacote)

# 2. ambiente Python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. modelo original (nunca é modificado) — Gemma 2 2B em BF16
export LWS_MODEL_DIR=/caminho/para/gemma-2-2b-it
```

## Antes de começar (pré-requisitos)

- [x] Conta do Kaggle com telefone verificado (GPU e internet nos notebooks)
- [ ] Licença do Gemma 2 aceita no Hugging Face (ou no Kaggle Models) e `HF_TOKEN` guardado como secret do Kaggle — nunca no código (P-002)
- [ ] Licença do Llama 3.1 aceita no Hugging Face (só para o survey de entropia, AC-034)
- [ ] Repositório git criado e publicado no GitHub como público (o `verify.sh` clona o repo no Kaggle e o artefato do artigo precisa ser público)

## Onde cada coisa roda

- **Local, sem GPU:** o Claude Code implementa as tarefas; os testes de CPU
  (codec de referência, packer, lógica do `lws.device` com GPUs simuladas,
  dispatch do modo `auto`) passam, os de GPU viram *skip*.
- **Qualquer GPU NVIDIA com compute capability ≥ 6.0:** a prova. Vale de uma
  placa doméstica (GTX 10xx, RTX 20/30/40/50) até datacenter (P100, T4, A100,
  H100, B200), em Linux ou WSL2 — seu PC, Kaggle, Colab ou uma máquina alugada.
  Faça commit também de `.claude/` (a skill com o motor), clone o repo na
  máquina e rode:

  ```bash
  export LWS_MODEL_DIR=/caminho/para/gemma-2-2b-it
  export LWS_DEVICE=0            # opcional: qual GPU (índice do nvidia-smi)
  bash scripts/verify.sh <feature>
  ```

  O script escolhe um PyTorch que roda na arquitetura detectada (sm_60/70 →
  build cu126; Blackwell → cu128+), instala Node e CuPy, roda o *doctor*
  (`python -m lws.device`) e depois `verify` + `audit` na GPU. No Kaggle,
  `scripts/kaggle_verify.sh` chama o mesmo script. Depois faça commit de
  `.spec/verification/<feature>.json` e `reports/` (inclui a cópia por GPU em
  `reports/verification/<feature>-<gpu-slug>.json`).

Requisitos mínimos de VRAM (Gemma 2 2B, perfil de referência):

| O que roda | VRAM livre |
|---|---|
| codec + kernels + microbenchmark | ~4 GB |
| modelo empacotado (SE12) | ≥ 6 GB |
| modelo original BF16 (baseline e2e, AC-016/AC-017) | ≥ 8 GB |

Perfis maiores (9B, 27B) são opcionais em GPUs com mais VRAM; o `lws.device`
diz o que cabe. Numa GPU em que o original não cabe, o e2e registra
"does not fit" e mede só o empacotado — é o ganho de capacidade.

O kernel é um único fonte CUDA C compilado em tempo de execução (NVRTC) para a
arquitetura detectada, com fallback para PTX. Nada no código depende do nome da
GPU: lançamento e tuning vêm das propriedades consultadas (SMs, L2, VRAM).

## Roadmap (ordem obrigatória)

| # | Feature | Tarefas | Portão para seguir |
|---|---|---|---|
| 1 | `weight-codec` | T-001, T-002, T-003, T-017 | audit exit 0 + relatório de entropia confirma ASM-001 |
| 2 | `fused-decode-gemv` | T-011, T-004, T-005, T-012, T-006 | audit exit 0 com o veredito **medido** (GO ou NO-GO) na P100 e na T4 |
| 3 | `performance-model` | T-013, T-014 | audit exit 0 (erro de previsão ≤ 10% no experimento controlado) |
| 4 | `model-integration` | T-007..T-009 | audit exit 0 (inclui exatidão ponta a ponta, AC-033) |
| 5 | `weight-offload` | T-015, T-016 | audit exit 0 |
| 6 | `local-api` | T-018, T-010 | opcional — fora do artigo (vLLM onde suportado, FastAPI nas demais GPUs) |

O NO-GO deixou de bloquear a integração: exatidão ponta a ponta e offload têm
valor mesmo que o kernel fundido não ganhe do BF16 lendo da VRAM. O veredito
continua **por GPU** (`reports/verdict-<gpu-slug>.json`) e o runtime em
`mode="auto"` só usa o kernel fundido onde houve GO medido. O plano do artigo
(contribuições, experimentos, riscos) está em [docs/paper-plan.md](docs/paper-plan.md).

## Arquitetura e estilo de código

Todo o código Python segue **arquitetura hexagonal** (constituição P-013) e
**Object Calisthenics** (P-014), onde se aplicam:

```
src/lws/
  domain/        regras puras (codec SE12, entropia, decisões por GPU, veredito) — só stdlib + numpy
  application/   casos de uso + ports (typing.Protocol) — não conhece torch/cupy/safetensors
  adapters/      safetensors, cuda (+ kernels .cu), transformers, filesystem, http — única camada com libs externas
  analyze.py, pack.py, device.py, tune.py, bench/, runtime/, server/   entrypoints (composition roots)
```

As 9 regras de Object Calisthenics valem integralmente em `domain` e
`application`. Nos adapters, as regras 3, 4, 5, 8 e 9 cedem onde a biblioteca
impõe a forma (`nn.Module`, schemas FastAPI, caminhos quentes por token). Os
kernels CUDA C, os testes e os scripts estão fora do escopo. O
`tests/test_principle_architecture.py` (T-001) verifica tudo isso via AST em
cada `verify`.

## Testes → prova

`conftest.py` converte o resultado do pytest em TAP com as tags
`@spec:AC-xxx` das docstrings, que é o que o `onp-spec verify` lê. Testes de
GPU/modelo são *skip* sem GPU ou sem `LWS_MODEL_DIR` — e skip **não conta como
prova**: rode o verify na máquina com a GPU.
