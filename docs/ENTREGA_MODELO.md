# Entrega reproduzível do modelo WORCAP

Este documento é parte obrigatória do artefato final. Ele descreve o código de
treinamento, o código de inferência e o ambiente necessário para regenerar a
submissão. O repositório não contém automação de envio ao Kaggle: todo envio
exige aprovação humana depois das auditorias.

## 1. Dados e tarefa

Entrada oficial: os 13 arquivos da competição, sem alteração de conteúdo. A
grade tem 301 latitudes × 261 longitudes. O alvo de um mês `T` é a precipitação
média em mm/dia; seus campos atmosféricos correspondem a `T-1`.

Arquivos esperados:

- `treino_tp.nc` e `treino_tp_alvo.nc`;
- os nove arquivos `treino_*.nc` atmosféricos;
- `teste_features.nc`;
- `sample_submission.csv`.

Defina `WORCAP_DATA_DIR` ou passe `--data-dir`. No Kaggle, o caminho é detectado
automaticamente em:

`/kaggle/input/previsao-climatica-de-precipitacao-sobre-a-america-do-sul`

Os únicos dados auxiliares são SOI e Niño 1+2, Niño 3 e Niño 4, preservados em
`baseline_169874/scripts/data/indices/`. Seus hashes entram no metadata do
modelo final. São séries mensais retrospectivas; a versão de divulgação não foi
reconstruída, portanto elas entram sempre defasadas e essa ressalva acompanha o
artefato.

## 2. Contrato temporal e proteção contra overfit

Para cada alvo `T`:

- os campos atmosféricos terminam em `T-1`;
- índices oceânicos usam defasagens 1, 2 e 3;
- durante um bloco de 24 meses, a precipitação é congelada na última observação
  anterior ao bloco;
- climatologias, médias, desvios e anomalias são ajustados somente no treino;
- cada fold usa no máximo os 30 anos anteriores.

Folds oficiais de desenvolvimento:

1. janeiro/1997–dezembro/1998;
2. janeiro/2015–dezembro/2016;
3. janeiro/2021–dezembro/2022.

A métrica de decisão é o RMSE agregado dos meses 13–24. O RMSE dos 24 meses é o
guardrail. Um candidato só é promovido se melhorar o segundo ano agregado,
mantiver ou melhorar os 24 meses, não piorar nenhum segundo ano em mais de 0,02
e melhorar mais de um fold. O placar público de 2023 não escolhe modelos.

## 3. Código entregue

### LightGBM

`scripts/worcap_pipeline.py` contém:

- `validate_model(config, folds)`: validações causais e OOF;
- `fit_final(config)`: treino final somente em janeiro/1993–dezembro/2022;
- `predict_submission(model)`: inferência e CSV na ordem do sample;
- `blend_submissions(inputs, weights)`: blend de candidatos aprovados;
- calibração por horizontes 1–6, 7–12, 13–18 e 19–24;
- registro em `experiments/registry.csv` e auditoria de submissão.

Três configurações são fixas (`wide`, `regularized`, `shallow`). Não há busca
ampla de hiperparâmetros. As amostras espaciais `a` e `b` são disjuntas,
estratificadas por latitude, longitude e regime de chuva.

### U-Net experimental

`scripts/worcap_unet.py` implementa uma U-Net base 16, três níveis, `AdamW`,
MSE, mixed precision, batch 2, no máximo 30 épocas e early stopping de cinco
épocas. A grade é preenchida para 304×264 e recortada para 301×261. Não há
flips, rotações ou outra transformação geograficamente inválida.

A U-Net começa em 2015–2016. Só continua para 2021–2022 se melhorar sozinha ou
se o blend com LightGBM ganhar pelo menos 0,01. O fold 1997–1998 só é executado
depois dos dois primeiros passarem. O treino final usa a mediana das épocas
selecionadas.

### Baseline preservado

`baseline_169874/` contém o pacote integral da submissão pública 1,69874. O CSV
histórico correto possui SHA-256:

`2b5084b0bb00131948e1cba3720b7e96774b0f5b10d96119646c0508fdd8782b`

Foram corrigidos apenas o diretório de cache ausente e o metadata incorreto de
850 árvores (o treino real usa 700). A lógica do baseline foi preservada.

## 4. Ambiente e recursos

Referência:

- Python 3.11;
- dependências em `requirements-worcap.txt`;
- CPU de 8 threads para LightGBM;
- aproximadamente 29 GB de RAM;
- GPU NVIDIA P100 ou T4 para a U-Net;
- armazenamento temporário recomendado: 10 GB;
- tempo máximo planejado: 12 horas.

Instalação limpa:

```bash
python -m pip install -r requirements-worcap.txt
python -m unittest discover -s tests -v
```

No ambiente final, preserve também:

```bash
python --version > artifacts/python-version.txt
python -m pip freeze > artifacts/pip-freeze.txt
nvidia-smi > artifacts/nvidia-smi.txt
```

Esses três comandos são executados pelo notebook antes do empacotamento final.

## 5. Backtests LightGBM

Execute os três modelos em `a` primeiro. A suíte reutiliza a preparação de cada
fold para respeitar a janela de 12 horas:

```bash
python scripts/worcap_pipeline.py \
  --data-dir "$WORCAP_DATA_DIR" \
  validate-suite --configs wide regularized shallow --folds 1997 2015 2021 \
  --spatial-sample a --n-points 8000 --validation-stride 4
```

Somente a configuração vencedora é repetida em `--spatial-sample b`, usando o
comando `validate --config NOME`.

O ensemble espacial é decidido mecanicamente:

```bash
python scripts/worcap_pipeline.py evaluate-blend \
  --component-a experiments/CONFIG_a_fold1997_oof.npz experiments/CONFIG_a_fold2015_oof.npz experiments/CONFIG_a_fold2021_oof.npz \
  --component-b experiments/CONFIG_b_fold1997_oof.npz experiments/CONFIG_b_fold2015_oof.npz experiments/CONFIG_b_fold2021_oof.npz \
  --spatial-average --output experiments/CONFIG_spatial_average.json
```

`accept_average` só será verdadeiro se a média 50/50 superar o melhor dos dois
componentes no segundo ano de cada fold.

Calibração OOF do candidato aprovado:

```bash
python scripts/worcap_pipeline.py calibrate \
  experiments/regularized_a_fold1997_oof.npz \
  experiments/regularized_a_fold2015_oof.npz \
  experiments/regularized_a_fold2021_oof.npz \
  --output experiments/regularized_a_calibration.json
```

O relatório usa leave-one-fold-out para medir a calibração sem reaproveitar o
alvo avaliado. Os quatro pesos finais são ajustados em todo o OOF apenas depois
da configuração ser escolhida.

## 6. Treino e inferência finais LightGBM

Exemplo (substituir pelo candidato realmente aprovado):

```bash
python scripts/worcap_pipeline.py \
  --data-dir "$WORCAP_DATA_DIR" fit-final \
  --config regularized --spatial-sample a --n-points 8000 \
  --calibration-json experiments/regularized_a_calibration.json

python scripts/worcap_pipeline.py \
  --data-dir "$WORCAP_DATA_DIR" predict \
  --model artifacts/lgb_regularized_a_final.json \
  --output artifacts/submission_lgb_regularized_a.csv
```

Se `a` e `b` melhorarem os três folds em média simples, treine os dois e passe
dois `--model` com `--weights 0.5 0.5`.

## 7. U-Net e blend

Primeiro gate:

```bash
python scripts/worcap_unet.py --data-dir "$WORCAP_DATA_DIR" \
  validate --fold 2015 --max-epochs 30 --patience 5
```

Se passar, execute 2021 e só então 1997. Após escolher a mediana de épocas:

```bash
python scripts/worcap_unet.py --data-dir "$WORCAP_DATA_DIR" \
  fit-final --epochs NUMERO_MEDIANO --output artifacts/unet_final.pt

python scripts/worcap_unet.py --data-dir "$WORCAP_DATA_DIR" \
  predict --checkpoint artifacts/unet_final.pt \
  --output artifacts/submission_unet.csv
```

Pesos de blend são não negativos, em passos de 0,05, e escolhidos somente com
OOF. O blend só passa se ganhar pelo menos 0,01 no segundo ano agregado sem
regressão relevante em nenhum fold.

## 8. Auditoria obrigatória antes de enviar

```bash
python scripts/worcap_pipeline.py --data-dir "$WORCAP_DATA_DIR" \
  audit-submission artifacts/submission_candidate.csv
```

A auditoria exige:

- 1.885.464 linhas;
- colunas exatamente `id,tp_mm_day`;
- IDs únicos e na ordem do `sample_submission.csv`;
- nenhum `NaN`, infinito ou valor negativo;
- metadata com hash do CSV, pesos e modelos.

Além disso, o notebook executa testes sintéticos de invariância temporal. Uma
piora pública superior a 0,03 é tratada como suspeita de pipeline, não como
evidência para retunar no ano público.

## 9. Manifesto final do vencedor

Antes de entregar ao patrocinador, inclua no ZIP:

- todo o conteúdo versionado deste repositório;
- modelo(s) final(is) (`.txt` e/ou `.pt`);
- metadata JSON de cada modelo;
- CSV vencedor e seu metadata;
- `experiments/registry.csv` e OOFs que justificam a seleção;
- `python-version.txt`, `pip-freeze.txt` e `nvidia-smi.txt`;
- hashes SHA-256 de todos os itens acima;
- uma cópia das regras/licenças aplicáveis aos dados.

O comando de empacotamento deve ser executado somente após o hash do CSV local
coincidir com o arquivo efetivamente enviado ao Kaggle.

Exemplo, repetindo `--artifact` para todos os modelos, metadados, OOFs e arquivos
de ambiente usados:

```bash
python scripts/package_winner.py \
  --submission artifacts/submission_candidate.csv \
  --sample "$WORCAP_DATA_DIR/sample_submission.csv" \
  --artifact artifacts/lgb_CONFIG_a_final.txt \
  --artifact artifacts/lgb_CONFIG_a_final.json \
  --artifact artifacts/pip-freeze.txt \
  --artifact artifacts/python-version.txt \
  --artifact artifacts/nvidia-smi.txt \
  --output artifacts/entrega_vencedora.zip
```

O ZIP inclui `MANIFEST.json` com hashes, commit Git, estado do diretório, pontos
de entrada de treino/inferência e a auditoria do CSV. Os dados oficiais não são
redistribuídos.
