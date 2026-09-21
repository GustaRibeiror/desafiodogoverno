# LucasTeste 1 — experimento residual LightGBM

Este diretório contém o experimento que prevê a anomalia de precipitação em
relação à climatologia mensal por pixel.

## Arquivos

- `train_residual_lightgbm.py`: treina o modelo e gera a submissão.
- `backtest_residual_lgbm.py`: avalia o método no bloco 2015--2016, com a
  precipitação congelada para reproduzir a defasagem do conjunto de teste.
- `common.py`: caminho dos dados e RMSE usados apenas pelo experimento.
- `requirements.txt`: dependências do experimento.

## Execução

Na raiz do repositório:

```powershell
python -m pip install -r "lucasteste 1/requirements.txt"
python "lucasteste 1/backtest_residual_lgbm.py"
python "lucasteste 1/train_residual_lightgbm.py"
```

Os dados continuam em `data/` e os artefatos grandes são escritos em
`outputs/`; ambos ficam fora do Git para manter o repositório leve.
