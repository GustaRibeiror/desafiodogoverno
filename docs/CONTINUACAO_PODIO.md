# Estado e decisões — 22/09/2026

## Objetivo e limites

Melhorar a chance no privado de 2024, meta pública aspiracional 1,48895.
Nenhuma submissão sem aprovação explícita. Não apagar submissões.
Não usar precipitação observada de 2023/2024 nem atmosfera do mês-alvo.
Atmosfera fornecida para T-1 é entrada permitida para prever T. O CSV contém
2023 e 2024; só 2023 é pontuado publicamente. Nenhum RMSE histórico equivale
diretamente ao score público.

Este documento registra fatos observados e hipóteses; não promete pódio.

## Ambiente e artefatos

- Repositório local: `/Users/desenvolvimento/study/desafiodogoverno`.
- Notebook: https://www.kaggle.com/code/gustavofloresribeiro/worcap-valida-o-causal-final/edit
- Fonte privada anexada: Dataset `Notebook-Gustavo`. Versão enviada não contém
  necessariamente arquivos novos locais. Não supor sincronização automática.
- Dados no Kaggle: `/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul`.
- Código em execução: `/kaggle/working/desafiodogoverno`.
- Hash do pipeline de A e B: `db3a679f8b2a764a0049821b00f13951931024b511dae17bdea55b043f9d584c`.
- Melhor CSV antigo: `/Users/desenvolvimento/Downloads/submission_ocean_2023_2024.csv`.
- SHA-256 antigo: `2b5084b0bb00131948e1cba3720b7e96774b0f5b10d96119646c0508fdd8782b`.
- Score antigo: 1,69874. Fonte em `baseline_169874/`.
- GPU liberada pelo usuário; sessão inspecionada ainda CPU. Não reiniciar enquanto B roda.

## Evidência observada

Regularized, seed 2026, 600 árvores, 77 features, 8.000 pontos de treino por mês,
grade de avaliação stride 4 (5.016 pontos), janelas de treino de 30 anos.

| Fold | Modelo 24m | Climatologia 24m | Modelo ano 2 | Climatologia ano 2 |
|---|---:|---:|---:|---:|
| 1997–1998 | 1,947765 | 2,143565 | 1,935616 | 2,192612 |
| 2015–2016 | 1,794775 | 1,850706 | 1,845618 | 1,849041 |
| 2021–2022 | 1,812241 | 1,857034 | 1,815064 | 1,863505 |

Conclusão: ganho nos três segundos anos, porém quase nulo em 2016 (0,003423).
Ainda NÃO demonstramos superioridade ao modelo antigo de score 1,69874.
A escolha de regularized veio da triagem em 2015; esse fold não é teste intocado.
Não comparar diretamente o smoke stride 8 com stride 4: as grades mudaram.

## Calibração rejeitada (executada de verdade no Kaggle)

Calibrar em dois folds e avaliar no terceiro produziu ano 2:

- 1998: 1,978458 (piora 0,042842).
- 2016: 1,846135 (piora 0,000517).
- 2022: 1,815070 (piora 0,000006).

Rejeitar: manter pesos [1,1,1,1]. Os pesos ajustados em todos os folds,
[1,061103, 1,026987, 0,876105, 0,849486], NÃO foram aprovados para inferência.
LOFO testa transferência entre regimes, mas usa folds posteriores para calibrar
folds anteriores; não apresentar isso como avaliação cronológica do calibrador.
Uma checagem adicional útil: ajustar em 1997 e 2015 e testar só 2021 (já presente
no resultado LOFO de 2021). Os folds continuam sujeitos à seleção de modelos.

## Execução em andamento e próximo passo exato

Os artefatos A foram copiados e verificados por hash no Kaggle em:
`experiments/a_8000_stride4_preserved/regularized_a_fold{1997,2015,2021}{.json,.txt,_oof.npz}`.
Calibração exploratória salva em `experiments/calibration_a_diagnostic.json`.

Atualizacao 17:40 BRT: a execucao B concluiu em
`/kaggle/working/desafiodogoverno/experiments/b_8000_stride4_20260922T202930Z`.
O `SPATIAL_AVERAGE_REPORT` foi lido. A media A+B 50/50 ficou:

| Fold | A ano 2 | B ano 2 | A+B ano 2 | A 24m | B 24m | A+B 24m |
|---|---:|---:|---:|---:|---:|---:|
| 1997 | 1,935616 | 1,932051 | 1,928439 | 1,947765 | 1,936522 | 1,937588 |
| 2015 | 1,845618 | 1,852546 | 1,843553 | 1,794775 | 1,806056 | 1,795214 |
| 2021 | 1,815064 | 1,825021 | 1,817318 | 1,812241 | 1,812568 | 1,809909 |

O helper marcou `accept_average: false` porque exigia melhora do ano 2 em todos
os folds. Pelo protocolo original, A+B passa: melhora o ano 2 agregado, melhora
ou mantem 24m agregado, melhora 2 de 3 folds e a unica regressao no ano 2
(2021) e de apenas +0,002253, bem abaixo do limite de +0,02. Nao ajustar pesos.
O candidato operacional agora e `regularized A+B 8000`, pesos 0.5/0.5.

Proximo passo pratico: treinar final A e B em 1993-2022 e gerar CSV A+B.
Isso ainda NAO envia ao Kaggle.

```python
import subprocess, sys
from pathlib import Path

DATA = Path("/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul")
REPO = Path("/kaggle/working/desafiodogoverno")

subprocess.run([
    sys.executable, "scripts/worcap_pipeline.py",
    "--data-dir", str(DATA),
    "fit-final",
    "--config", "regularized",
    "--spatial-sample", "a",
    "--n-points", "8000",
], cwd=REPO, check=True)

subprocess.run([
    sys.executable, "scripts/worcap_pipeline.py",
    "--data-dir", str(DATA),
    "fit-final",
    "--config", "regularized",
    "--spatial-sample", "b",
    "--n-points", "8000",
], cwd=REPO, check=True)

subprocess.run([
    sys.executable, "scripts/worcap_pipeline.py",
    "--data-dir", str(DATA),
    "predict",
    "--model", "artifacts/lgb_regularized_a_final.json",
    "--model", "artifacts/lgb_regularized_b_final.json",
    "--weights", "0.5", "0.5",
    "--output", "artifacts/submission_regularized_ab_8000.csv",
], cwd=REPO, check=True)
```

Depois da geracao, conferir o JSON impresso pela auditoria:
`rows` deve ser 1.885.464, `unique_ids` deve ser 1.885.464, sem NaN/infinito,
minimo >= 0. O arquivo esperado sera:
`/kaggle/working/desafiodogoverno/artifacts/submission_regularized_ab_8000.csv`.

Nao submeter automaticamente. Baixar/abrir o CSV, conferir metadata ao lado e
pedir confirmacao explicita do usuario antes de gastar tentativa no Kaggle.

Itens que continuam pendentes depois desse candidato:

1. Comparar ao algoritmo antigo nos mesmos folds, com o mesmo corte temporal
   e a mesma grade de avaliação. `baseline_169874/scripts/backtest_residual_lgbm.py`
   NÃO reproduz sozinho o algoritmo ocean final: ele usa janela/rounds diferentes.
   Inspecionar `generate_ocean_submission.py` e suas dependências antes de portar.
2. Reproduzir a submissao antiga 1,69874 por hash quando houver tempo.
3. Testar contexto climatologico coarse/fino somente depois de ter este CSV novo.

## Ferramenta nova local

`scripts/diagnose_oof.py` lê três OOFs sem treinar. Produz métricas por mês,
ano, climatologia, calibração em folds separados e RMSE conjunto correto
(raiz da soma dos erros quadráticos dividida pelo número de observações).
Rejeita grades diferentes e caminhos duplicados; passar UM OOF por fold da
MESMA configuração. `tests/test_diagnose_oof.py` verifica transferência dos
pesos entre folds, agregação e rejeição de calibração que piora.

Exemplo após transferir o script para o Kaggle:

```python
subprocess.run([sys.executable, 'scripts/diagnose_oof.py', '--oof',
    *[str(p) for p in paths], '--output', 'experiments/diagnostic_a.json'], check=True)
```

## Insights de revisão para trabalho posterior

1. **Referência antiga é indispensável.** Bater climatologia é necessário,
   mas não prova que melhoramos a melhor submissão. Reproduzir o hash do CSV
   original também continua pendente; ter a fonte não significa reproduzi-lo.
2. **Âncora de chuva tem diferença treino/teste.** `build_table` sorteia lead
   por mês, logo a âncora de treino pode ser qualquer mês. Em todos os blocos
   de teste a âncora é dezembro. Hipótese: testar UMA ablação sem `tp_anchor`
   e `lag_meses`, ou simulação de blocos iniciados em janeiro. Fazer isso só
   depois da referência e com configuração congelada, sem busca ampla.
3. **Calibração por horizonte não transferiu.** Não ajustar dezenas de pesos
   regionais para tentar resgatar o resultado: só temos 72 meses de avaliação.
4. **U-Net tem seleção no próprio fold externo.** `train_unet` escolhe época
   pelo menor erro no mesmo bloco usado para reportar OOF. Corrigir para
   seleção em bloco interno anterior, depois retreinar até o corte externo
   por número fixo de épocas e avaliar bloco externo uma única vez.
5. **Grades da U-Net e LightGBM diferem.** U-Net salva 78.561 pontos por mês;
   tabular salva 5.016. Extrair exatamente `point_flat` do tabular nas predições
   da U-Net e verificar truth/ordem antes de qualquer blend.
6. **Bug de agregação no blend genérico.** `optimize_two_model_blend` escolhe
   peso com erros agrupados, mas calcula gate usando média aritmética de RMSEs.
   Corrigir antes de usar esse caminho. Ele também seleciona e reporta peso
   no mesmo conjunto OOF; usar pesos fixos ou avaliação em folds separados.
7. **Nomes de experimentos colidem.** O pipeline antigo não codifica n_points,
   stride ou hash no nome. Não mudar o código de A/B a meio da comparação:
   usar output_dir único, como a célula B já faz. Incorporar identidade completa
   em uma próxima versão e registrar versões distintas.
8. **Invariância atual é parcial.** Teste existente altera features já montadas;
   ampliar depois para preparação e inferência completas, alterando linhas
   futuras do input. Testar também alteração do alvo oculto e âncora congelada.
9. **Alvo e latência real.** Índices NOAA revisados e ERA5 têm atraso de publicação;
   o contrato operacional deste desafio é mês de origem. Documentar isso sem
   afirmar que o pipeline reproduz disponibilidade em tempo real rigorosa.
10. **Dados externos novos** (NMME/SEAS5) estão fora da janela autorizada do
    plano atual. Não baixar precipitação ERA5 de 2023/2024 em hipótese alguma.

## Entrega e retomada

Manter notebook/dataset privados. Não afirmar conclusão de execução apenas por
ver logs de carregamento. Checar retorno e artefatos. Não modificar notebook
de forma que um Run All sobrescreva os OOFs preservados.
O treinamento B pode levar aproximadamente 20–25 minutos pelo tempo de A;
isso é estimativa, não prazo garantido. O reset de tokens não altera a sessão Kaggle.
Para a continuação: ler primeiro este documento e o relatório B; não refazer A.

## Revisão adicional — últimos minutos de orçamento

### Achado de alta prioridade: mudamos a escala espacial da referência

Verificado no código, não uma suposição:

- O antigo `train_residual_lightgbm._clim_context` opera em grade stride 4,
  portanto centros de células separados por 1°. Usa OITO vizinhos, exclui o
  centro, e calcula gradientes por índice nessa grade. Na inferência, o
  `generate_ocean_submission.py` interpola essas features para a grade cheia.
- O novo `_clim_context_full` opera em grade cheia de 0,25°, usa NOVE células
  incluindo centro e gradientes por índice nessa grade.
- Logo vizinhança antiga tem extremos separados por 2° em cada eixo;
  a nova por 0,5°. Gradientes também têm escala distinta (aproximadamente 4x
  para campo suave). Não é erro interno treino/inferência do modelo novo,
  mas é mudança importante ao comparar algoritmos.

Experimento prioritário depois de A/B e referência: manter pontos, seed,
hiperparâmetros e folds; adicionar contexto climatológico na escala antiga
com nomes novos (`clim_coarse_*`) e manter o contexto fino. Calcular grade
coarse e interpolação em TODO treino/validação/inferência com o mesmo método.
Uma única variante predefinida; não procurar muitos raios pela nota.
Alternativa mais barata: substituir contexto fino por coarse em uma ablação.
Não sobrescrever o algoritmo antigo necessário para reproduzir o CSV.

### Referência antiga não é a configuração `wide`

`generate_ocean_submission.py` usa 224 folhas, LR .025, 700 rounds, mas herda
`PARAMS` sem as regularizações novas e usa features/contexto distintos.
`wide` no pipeline atual não é um backtest da entrega 1,69874.
Os números 1,767188/1,859466 no metadata antigo estão escritos como constantes:
não considerá-los evidência executada de comparação nos folds atuais. Buscar
os resultados que os originaram ou recalcular com o algoritmo exato.

### Escolha de ensemble: medir erros, não só correlação das previsões

Para cada segundo ano calcular `eA = A - truth`, `eB = B - truth`, MSE A/B e
`mean(eA*eB)`. A MSE da média é `(MSE_A + MSE_B + 2*mean(eA*eB))/4`.
Isso explica por que B ajuda ou não: correlação alta das previsões totais pode
vir da climatologia comum e esconder diferenças úteis nos erros/anomalias.
Registrar decomposição por fold; não escolher dezenas de pesos por região.
Se A/B não ajudar, priorizar um componente com hipótese realmente diferente,
em vez de repetir seeds de árvores parecidas.

### Candidato complementar de baixo custo antes de uma U-Net longa

Hipótese, ainda NÃO implementada/testada: modelo linear regularizado de
anomalias em componentes espaciais (EOF/PCA), com índices NOAA e componentes
da atmosfera T-1 como entradas. Ele pode capturar padrões de grande escala
com erros diferentes dos do LightGBM.

Contrato experimental: ajustar médias, escalas e PCA SOMENTE no treino de
cada fold; alvo residual pela climatologia do mesmo treino; componentes e
penalidade escolhidos num corte interno anterior ao fold externo. Começar
com uma configuração pequena predefinida, por exemplo 16 componentes de alvo,
e regressão Ridge; avaliar 24m e segundo ano em grade idêntica. Projetar e
reconstruir a grade toda preservando as coordenadas. Manter esse trilho como
opcional; primeiro resolver a comparação com o baseline.

### Quantidade de informação temporal limita redes grandes

8.000 pixels × 360 meses dá milhões de linhas, mas só 360 estados atmosféricos
mensais distintos por janela e pixels vizinhos correlacionados. Não tratar as
linhas como milhões de exemplos temporais independentes. Aumentar grade ajuda
representação espacial; não acrescenta eventos climáticos inéditos. Incerteza
de ganhos deve ser examinada por meses/anos/blocos, nunca bootstrap de pixels
como se fossem independentes. Só três folds não certificam vencer 2024.

### Diagnóstico regional deve seguir contribuição à métrica

Calcular SSE por caixas geográficas fixadas previamente, número de pontos e
SSE/n_total. Rankear contribuição ao erro global, não apenas RMSE regional.
O domínio tem oceano e continente e a métrica informada conta todos os pontos
igualmente. Não mascarar oceanos nem introduzir pesos por área/cos(latitude)
na métrica de seleção sem regra explícita. Caixas lat/lon não equivalem a uma
máscara terrestre; nomear como caixas, não afirmar classificação terra/mar.

### Checagens baratas antes da inferência final

- Paridade numérica: montar features de um mês histórico pelos caminhos
  `build_table` e `_test_month_frame` com as mesmas estatísticas e inputs;
  comparar por nome/posição nos mesmos `point_flat`. Esse teste pode pegar
  erro que um RMSE histórico bom jamais mostraria na submissão.
- `_prepare_full_inference` pressupõe ordem (time,lat,lon) no teste e âncora
  com dimensão temporal. Inspecionar dims e usar transpose explícito, validar
  24 datas consecutivas 2023-01..2024-12 e lead 1..24. Comparar âncora com
  `treino_tp` de dez/2022, não apenas igualdade entre as 24 cópias.
- `predict_unet_submission` atualmente não chama `_assert_sample_grid_order`;
  adicionar antes de atribuir valores. Auditar IDs depois não detecta valores
  permutados se os IDs corretos vieram do sample.
- Em `predict_submission`, verificar hashes das fontes/índices/dados de
  normalização contra os registrados no treino; o código atual registra parte
  deles, mas não impede inferência com entradas auxiliares alteradas.
- Desvio local é calculado como E[x²]-E[x]² em float32, suscetível a cancelamento
  em campos quase constantes. Se aparecerem zeros suspeitos, comparar com
  acumulação float64/desvios centrados; mudar igualmente treino e inferência.

### O que não concluir ainda

1998 tem ganho grande e pesa no agregado; reportar sempre os três ganhos
individuais e se o ganho persiste retirando esse fold. Não usar a transição
ENSO como justificativa para aumentar peso em 2024 sem validação. A queda de
desempenho da calibração entre regimes é evidência contra pesos frágeis.
Os próximos passos devem vir de uma hipótese registrada ANTES da execução.
Não redefinir os critérios de promoção para fazer um candidato passar.
