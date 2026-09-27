# Plano do artigo

> Pesquisador independente · orçamento zero · GPUs: Kaggle P100 e T4 · alvo: periódico (estrato A)

## Tese

A compressão **sem perdas** de pesos BF16 num formato de **taxa fixa** acelera
a inferência exatamente onde o link de memória é estreito em relação ao
cálculo disponível, e um modelo analítico com poucas grandezas medidas prevê
esse ponto em qualquer GPU. Onde acelera, o resultado é **bit-idêntico** ao do
modelo original — não só os pesos, as saídas.

Título provisório: *When Does Lossless Weight Compression Pay Off? A
Predictive Model, Exact Kernels and Compressed Offload for LLM Decoding.*

## Contribuições → onde estão no plano

| # | Contribuição | Feature / critérios | Por que é nova |
|---|---|---|---|
| C1 | Modelo preditivo da razão fused ÷ baseline e da fronteira GO/NO-GO, validado por experimento controlado e por previsão cruzada P100 ↔ T4 | `performance-model` AC-026..AC-028; base em `fused-decode-gemv` AC-012..AC-014 | Os trabalhos anteriores relatam ganhos e perdas isolados sem explicar por que uma GPU ganha e outra perde |
| C2 | Offload de pesos **comprimidos** por PCIe, com o efeito composto de capacidade + transferência | `weight-offload` AC-029..AC-031 | O ganho de capacidade (modelo passa a caber) já foi relatado; o regime em que nem o comprimido cabe e o ganho composto abaixo de 0,75 não |
| C3 | Exatidão ponta a ponta: kernel fundido com a mesma ordem de redução do kernel BF16 → logits bit-idênticos | `fused-decode-gemv` AC-032, `model-integration` AC-033 | O Split12 divergiu após 26/32 tokens pela ordem de redução; aqui a divergência é zero por construção, com o custo medido |
| M | Motivação: entropia do expoente em 5 famílias de modelos | `weight-codec` AC-001, AC-034 | Mostra que o formato generaliza além do Gemma (custo zero de GPU) |

## Perguntas de pesquisa

- **RQ1** — Quanto dos 25% de bytes economizados vira tempo, em cada GPU e batch, lendo da VRAM? (AC-012..AC-014)
- **RQ2** — Um modelo com banda, custo de cálculo por peso e um único parâmetro de sobreposição prevê essa razão com erro ≤ 10%? (AC-026..AC-028)
- **RQ3** — Com o modelo maior que a VRAM, quanto o SE12 reduz o tempo por token ao transmitir pesos por PCIe, e como isso varia com o orçamento de VRAM? (AC-029..AC-031)
- **RQ4** — Qual o custo de exigir saídas bit-idênticas (ordem de redução fixa) em relação ao melhor kernel ajustado livremente? (AC-032, AC-033)

## Experimentos e custo em GPU (Kaggle)

A cota do Kaggle é semanal (da ordem de 30 h de GPU; confira o valor atual) e
as sessões têm limite de horas. As estimativas abaixo são ordens de grandeza,
já com uma repetição.

| Experimento | Onde | GPU-h (P100 + T4) |
|---|---|---|
| Codec, pack, survey de entropia (AC-001..AC-008, AC-034) | máquina local, CPU | 0 |
| Decode e kernels, exatidão (AC-009..AC-011, AC-032) | Kaggle | ~4 |
| Microbenchmark + veredito (AC-012..AC-014) | Kaggle | ~6 |
| Calibração, varredura de k, previsão cruzada (AC-026..AC-028) | Kaggle | ~8 |
| Integração, equivalência, e2e 2B (AC-015..AC-018, AC-033) | Kaggle | ~8 |
| Offload 9B, 4 orçamentos × 2 formatos + llama.cpp (AC-029..AC-031) | Kaggle | ~12 |
| **Total** | | **~40 h ≈ 2 semanas de cota** |

## Baselines

| Onde | Baseline | Motivo |
|---|---|---|
| Kernel (VRAM) | kernel BF16 cru de mesma estrutura (P-009) | isola bytes lidos + decode; P100/T4 não têm cuBLAS BF16 |
| Kernel (referência) | torch FP32 `F.linear` com pesos convertidos | só referência; lê 4 bytes/peso |
| Offload | nosso pipeline de offload BF16 (mesmo código, só muda o formato) | isola o efeito do formato |
| Offload (externo) | llama.cpp BF16 com o mesmo `-ngl` | sistema conhecido; mostra que nosso baseline não é fraco |
| Exatidão | caminho padrão do transformers (AC-017) vs runtime de referência (AC-033) | separa divergência por ordem de redução de divergência por pesos (nenhuma) |

## Trabalhos relacionados a ler e comparar (confirmar números nas fontes)

- **Compressão sem perdas de pesos:** DFloat11 (arXiv 2504.11651 — inclui comparação com offload para CPU), ZipNN, SplitZip (arXiv 2605.01708), rANS em tiles fundido ao GEMM (arXiv 2606.15789), Split12 (`brianbell-x/weight-compression`, ver weight-codec design.md).
- **Offload:** FlexGen (ICML 2023), DeepSpeed ZeRO-Inference, llama.cpp (offload parcial), PowerInfer (usa esparsidade — contraste: não é exato).
- **Modelagem de desempenho:** modelo roofline (Williams, Waterman e Patterson) e extensões para GPU.
- **Determinismo na inferência:** trabalhos recentes sobre kernels *batch-invariant* e inferência determinística — ponte direta com C3.

## Ameaças à validade (escrever no artigo, não esconder)

- **Duas GPUs só, e nenhuma com BF16 nativo.** C1 é validado por experimento controlado e previsão cruzada, mas o artigo não afirma nada medido sobre sm_80+ (A100/H100/RTX 30+); só previsões, marcadas como tal.
- **VRAM emulada** para 6–12 GB (ASM-020): mesmos SMs e banda de uma GPU de 16 GB.
- **Ruído do Kaggle:** host compartilhado, sem travar clocks (sem root). Mitigação: P-006 registra clocks e temperatura; medianas com p10/p90; repetição em outra sessão.
- **Modelos:** speedups só com Gemma 2 (2B e 9B); a generalização vem só da entropia (AC-034).
- **Batch ≤ 16 e CUDA cores apenas:** serving em batch alto e tensor cores ficam como trabalho futuro.

## Publicação e artefato

- **Periódicos candidatos:** IEEE TPDS, ACM TACO, JPDC, FGCS. Confira o estrato de cada um na lista Qualis vigente da CAPES antes de escolher, e prefira a opção sem taxa de acesso aberto (APC) — o orçamento é zero.
- **Afiliação:** "Independent Researcher" é aceita nesses periódicos. Ter um coautor com vínculo acadêmico ajuda (revisão interna, acesso a SDumont/LNCC e programas como o NVIDIA Academic Grant para um segundo artigo), mas não é requisito.
- **Preprint:** arXiv (cs.DC ou cs.LG) antes ou junto da submissão; primeiro envio numa categoria pode exigir *endorsement* de um autor já ativo.
- **Artefato:** repositório público + notebooks públicos do Kaggle (reprodutível a custo zero — ponto forte a destacar) + DOI no Zenodo (P-015).

## Trabalho futuro (fora do primeiro artigo)

- **F5 — tensor cores:** decode direto em fragmentos de `mma` para batch > 1 e MoE em sm_80+ (exige A100/H100: pedir alocação ou grant).
- **Formato K15** (~11,2 bits/peso estimado pelo Split12) como `format_version` 2.
- **Transferência por rede** (sincronização de pesos em RL, cold start).
- **Outras GPUs:** relatórios da comunidade via `scripts/verify.sh` (Q-008), AMD/Intel (Q-007).

## Fases (sem prazo fixo)

| Fase | Sai quando | Produz para o artigo |
|---|---|---|
| 0. Pré-requisitos | licenças aceitas, repo público | — |
| 1. Codec + survey | `weight-codec` auditada | Seção de motivação (entropia, 5 famílias) |
| 2. Kernels | `fused-decode-gemv` auditada na P100 e na T4 | RQ1, RQ4 (nível kernel) |
| 3. Modelo preditivo | `performance-model` auditada | RQ2 — figura central (curva prevista × medida) |
| 4. Integração | `model-integration` auditada | RQ4 ponta a ponta, e2e 2B |
| 5. Offload | `weight-offload` auditada | RQ3 — curva razão × orçamento de VRAM |
| 6. Escrita | todos os números regenerados de `reports/` (P-015) | manuscrito + artefato com DOI |

## Decisões registradas

| Pergunta | Decisão |
|---|---|
| Q-002 — comprimir embedding / LM head | Sim |
| Q-004 — limiar de GO | 5% |
| Q-005 — prefill vs decode | Prefill sempre exato; decode segue o modo `auto` |
| Q-006 — API local | vLLM onde suporta a GPU + FastAPI nas demais (fora do artigo) |
| Q-007 — GPUs não NVIDIA | Só trabalho futuro |
| Q-008 — relatórios da comunidade | Não no primeiro artigo |
| Q-009 — FlexGen | Só trabalho relacionado; baseline externo = llama.cpp |
| Survey de entropia (AC-034) | Na máquina local, um shard por vez direto do Hugging Face |

## Pendências práticas (antes da fase 1)

- Aceitar as licenças do Gemma 2 e do Llama 3.1 no Hugging Face; guardar `HF_TOKEN` como secret do Kaggle.
- Criar o repositório git e publicá-lo no GitHub (público).
