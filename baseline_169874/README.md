# Entrega reproduzível — submissão pública 1,69874

## Conteúdo
- `submission_ocean_2023_2024.csv`: arquivo submetido, pontuação pública 1,69874.
- `scripts/generate_ocean_submission.py`: treino final e inferência da submissão.
- `scripts/train_residual_lightgbm.py`: pipeline de treino residual original.
- `scripts/backtest_residual_lgbm.py` e `scripts/run_temporal_experiments.py`: validação temporal.
- `data/indices`: SOI e Niño 1+2, 3 e 4 usados como covariáveis.

## Ambiente
Python 3.11+; `numpy`, `pandas`, `xarray`, `lightgbm`, `scikit-learn`, `joblib`, `threadpoolctl`, `netCDF4`.

## Dados da competição
Baixe os arquivos fornecidos pela competição e coloque-os em `data/`: `treino_tp.nc`, `treino_tp_alvo.nc`, os arquivos `treino_*.nc`, `teste_features.nc` e `sample_submission.csv`.

## Reprodução
No diretório `scripts`, execute:

```powershell
python generate_ocean_submission.py
```

O comando treina LightGBM com 30 anos (1993–2022), 224 folhas, taxa 0,025 e 700 iterações; usa três meses de campos atmosféricos e índices SOI/Niño com defasagens 1–3. O resultado é escrito em `output/submission_ocean_2023_2024.csv`.

## Identidade da entrega

O CSV histórico que obteve **1,69874** tem SHA-256:

`2b5084b0bb00131948e1cba3720b7e96774b0f5b10d96119646c0508fdd8782b`

Depois de uma execução limpa, confirme o arquivo gerado com `sha256sum` (Linux)
ou `shasum -a 256` (macOS). O código foi preservado sem alterar a lógica do
modelo; foram corrigidos somente dois defeitos de empacotamento do ZIP original:
o diretório de cache passa a ser criado na primeira execução, e o metadata agora
registra as 700 iterações realmente utilizadas (antes dizia 850).

## Controle temporal
Para prever o mês T, os campos atmosféricos vêm de T−1 e meses anteriores. Os índices climáticos entram somente defasados. Não há precipitação observada de 2023–2024 usada como recurso.
