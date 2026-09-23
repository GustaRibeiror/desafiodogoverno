# WORCAP — previsão climática de precipitação

Pipeline reproduzível para a competição privada do INPE/WORCAP no Kaggle. O
objetivo é prever a precipitação média mensal de janeiro/2023 a dezembro/2024
em uma grade de 0,25° sobre a América do Sul.

O critério de seleção é causal: o leaderboard público de 2023 é apenas uma
verificação de sanidade; modelos são promovidos pelos backtests históricos de
24 meses, com prioridade para o segundo ano, que imita o privado de 2024.

## Estrutura principal

**Melhor resultado público informado: 1,66882.** O script responsável pelo
ajuste final é [scripts/prepare_final_scale_candidate.py](scripts/prepare_final_scale_candidate.py).
Ele aplica uma única vez o fator 0,97 ao ensemble do projeto do Lucas
(original: 1,69147). Não é o treino A+B do `worcap_pipeline.py`.
Leia [o handoff da submissão 1,66882](docs/SUBMISSAO_166882.md) para reproduzir,
localizar a fonte do ensemble e continuar os experimentos. CSVs não são versionados.

- `scripts/worcap_pipeline.py`: LightGBM, folds causais, calibração, inferência,
  blend e auditoria da submissão.
- `scripts/worcap_unet.py`: U-Net espacial experimental com early stopping.
- `scripts/package_winner.py`: pacote final com manifesto e hashes verificáveis.
- `notebooks/worcap_final.ipynb`: orquestração fina para Kaggle Notebook.
- `baseline_169874/`: pacote preservado da submissão pública 1,69874.
- `experiments/registry.csv`: registro auditável de cada execução.
- `docs/ENTREGA_MODELO.md`: ambiente, comandos e protocolo para entrega do
  código-fonte vencedor.

## Início rápido no Kaggle

Anexe os dados da competição, habilite GPU e abra
`notebooks/worcap_final.ipynb`. O notebook executa verificações antes de cada
etapa e nunca submete automaticamente um CSV.

Na linha de comando:

```bash
python -m pip install -r requirements-worcap.txt
python scripts/worcap_pipeline.py --data-dir /kaggle/input/previsao-climatica-de-precipitacao-sobre-a-america-do-sul validate --config regularized --folds 1997 2015 2021 --spatial-sample a
```

Consulte [docs/ENTREGA_MODELO.md](docs/ENTREGA_MODELO.md) antes de treinar o
artefato final.
