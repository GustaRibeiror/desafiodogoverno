# Handoff — submissão pública 1,66882

Resultado informado pelo usuário no Kaggle: **1,66882**, contra **1,69147**
do ensemble original. O privado de 2024 permanece desconhecido.

## Algoritmo que gerou o resultado

`scripts/prepare_final_scale_candidate.py` lê o CSV original do ensemble,
preserva IDs e aplica `tp_mm_day_final = 0.97 * tp_mm_day_original` em todos
os meses de 2023 e 2024. Não houve retreino nesta etapa.
O arquivo é salvo com dez casas decimais e relido para auditoria.

O original é um ensemble 50/50 de LightGBM residual com climatologia,
atmosfera e índices NOAA, com e sem previsões GEOSS2S, treinado com alvos
até dezembro de 2022. A origem foi confirmada pelo usuário e pelo hash.
Essa candidata NÃO é o nosso `submission_regularized_ab_8000.csv` (~1,74).

Fonte do ensemble: https://github.com/Drslukas/TESTEdesafiodogoverno

Snapshot recebido: `TESTEdesafiodogoverno-main.zip`, commit registrado no ZIP:
`8dbe7e82049dcd62440e13baceaec542371670bf`.
Nesse projeto, os pontos de entrada são:

- `lucasteste1/generate_ocean_submission.py`: treino e inferência;
- `lucasteste1/nmme_features.py`: features GEOSS2S;
- `lucasteste1/evaluate_ensemble.py`: avaliação dos pesos históricos;
- `blend_submissions.py`: combinação dos CSVs;
- `run_ocean_repro.ps1`: orquestração Windows;
- `STATUS.md`: métricas e decisões documentadas.

O snapshot de código não inclui os dados, caches externos, modelos nem
previsões históricas. O colega que executou o treino deve preservar esses
arquivos. Refazer downloads externos pode mudar o resultado; o hash abaixo
identifica exatamente o original usado nesta entrega.

## Reproduzir o pós-processamento

Na raiz deste repositório, com Python, numpy e pandas instalados:

```bash
python scripts/prepare_final_scale_candidate.py --input /caminho/submission_candidate_2023_2024.csv --output-dir artifacts/final_candidate
```

No Kaggle, execute o arquivo como script, sem colar seu conteúdo na célula:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "/kaggle/working/desafiodogoverno/scripts/prepare_final_scale_candidate.py",
    "--input", "/kaggle/working/submission_candidate_2023_2024.csv",
    "--output-dir", "/kaggle/working/final_candidate",
], check=True)
```

Ajuste os caminhos à localização real dos arquivos. O script exige o hash
original e recusa sobrescrever a saída existente. Nunca usar como entrada o
CSV já multiplicado por 0,97. Não existe envio automático ao Kaggle.

Saídas: `submission_candidate_scale097.csv` e `scale097_audit.json`.

| Arquivo | SHA-256 |
|---|---|
| Original 1,69147 | `777633a4dee553192ba54c982d8691ba89d70ecbd105bb50561c2dff22c4ef7a` |
| Ajustado 1,66882 | `23adaa90263ec5dc6b1c16ddc05eec87bb424c215f4853fda4ae465dc3f913e5` |

Ambos têm 1.885.464 linhas, colunas `id,tp_mm_day`. A auditoria confirmou
IDs únicos e mesma ordem do original, valores finitos e não negativos.

## Evidência e limites

O `STATUS.md` recebido relata que escala 0,97 melhorou o RMSE de
2015–2016 de 1,78041 para 1,77962 e de 2021–2022 de 1,78996 para 1,78735.
Esses números foram lidos no documento, não recalculados aqui. Não há
métrica separada do segundo ano para esse ajuste no documento recebido.
A escolha usou validações de desenvolvimento, não um teste intocado.

Não usar precipitação observada de 2023/2024 ou atmosfera do mês-alvo.
Treino com alvos até 2022 não significa que os preditores da inferência
precisem terminar em 2022: atmosfera T−1 e previsões emitidas antes do alvo
seguem o contrato temporal adotado. Disponibilidade e regras das fontes
externas precisam continuar sendo respeitadas.

## Próximo experimento preparado, ainda NÃO executado nos dados reais

`scripts/check_final_scale_transfer.py` ajusta um único fator em 2015–2016
e o avalia congelado em 2021–2022 contra o atual 0,97. Exige melhora de
pelo menos 0,001 no bloco completo e no segundo ano, sem piorar o primeiro.
Não inclui o fold isolado 2022 na agregação, pois sobrepõe 2021–2022.

Execute no projeto do colega com os NPZ históricos:

```bash
python /caminho/check_final_scale_transfer.py --results lucasteste1/results
```

O script só imprime diagnóstico. Seus testes sintéticos passaram, mas
nenhum novo fator foi aprovado em dados reais nesta tarefa. Não escolher
outro fator apenas a partir das notas públicas de 2023. Preservar a entrega
1,66882 como referência e pedir confirmação antes de qualquer submissão.
