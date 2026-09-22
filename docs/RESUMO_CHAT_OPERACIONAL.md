# Resumo operacional do chat

Este arquivo complementa `docs/CONTINUACAO_PODIO.md`. O handoff principal tem
estado tecnico, metricas, comandos e proximos passos. Este resumo guarda o
contexto da conversa que pode ajudar na retomada em outro computador ou com
outro modelo.

## Situacao geral

- O usuario esta em um hackathon Kaggle/WORCAP de previsao climatica de
  precipitacao sobre a America do Sul.
- A meta declarada e tentar chegar ao topo, com referencia aspiracional de
  score publico `1.48895`.
- A melhor submissao existente do time antes deste trabalho tinha score publico
  `1.69874`.
- Existe outra submissao pior, `1.71154`, considerada util apenas para
  comparacao historica.
- O usuario esta sob pressao de tempo e quer uma submissao nova logo, mas sem
  fazer algo que seja claramente overfit ou irregular pelas regras.

## Regras e interpretacao decididas na conversa

- O leaderboard publico mede 2023.
- A avaliacao oficial/final do hackathon usa 2024, escondido ate o encerramento.
- Portanto, melhorar so o publico de 2023 pode ser overfit ao leaderboard.
- O criterio de escolha deve priorizar validacoes historicas causais de 24 meses,
  principalmente o segundo ano de cada fold.
- Dados observados de precipitacao de 2023/2024 nao devem ser usados, mesmo que
  o ERA5 esteja publicamente disponivel no Copernicus.
- Tambem nao devemos usar atmosfera do proprio mes-alvo por meio da linha
  seguinte do `teste_features.nc`.
- A informacao permitida para alvo T e a atmosfera ate T-1, alem de variaveis
  historicas/indices externos permitidos ja presentes.
- Nenhum envio ao Kaggle deve acontecer automaticamente; o usuario precisa
  confirmar antes de gastar tentativa.
- Apagar submissao no Kaggle nao foi tratado como estrategia confiavel para
  recuperar tentativa. Nao depender disso.

## Estado do Kaggle/notebook

- O notebook usado foi:
  `https://www.kaggle.com/code/gustavofloresribeiro/worcap-valida-o-causal-final/edit`
- O dataset privado criado pelo usuario no Kaggle se chama `Notebook-Gustavo`.
- O usuario liberou recursos que estavam bloqueados por verificacao da conta,
  incluindo GPU, mas a sessao observada continuava em CPU/No Accelerator.
- Foi orientado deixar a aba aberta durante execucoes longas do Kaggle, porque
  fechar a aba pode interromper ou deixar a sessao inconsistente.
- A execucao B demorou, mas nao estava travada: havia uso de CPU e output de
  carregamento das variaveis.

## Dores e decisoes de ritmo

- O usuario ficou frustrado porque o trabalho levou o dia inteiro sem gerar CSV.
- Foi reconhecido que o processo ficou conservador demais antes da primeira
  submissao nova.
- A decisao operacional passou a ser: gerar uma candidata A+B causal primeiro,
  sem mais rodadas longas de pesquisa antes de ter CSV na mao.
- Com apenas cerca de 10 minutos antes de sair do computador, foi recomendado
  nao iniciar treino final, pois dois modelos finais mais inferencia em grade
  cheia provavelmente passam de 10 minutos em CPU.
- O melhor uso desse tempo curto foi salvar/pushar tudo e garantir retomada.

## Git e preservacao

- O usuario pediu para atualizar o git e fazer push.
- Foi criado commit na branch `gustavo`:
  `d30bee7 Add reproducible WORCAP pipeline`
- O push foi feito para:
  `https://github.com/GustaRibeiror/desafiodogoverno.git`
- O status local ficou limpo apos o push.
- Foram incluidos codigo, docs, notebook, baseline preservado, testes e
  registry minimo.
- Nao foram incluidos arquivos pesados de treino/submissao/modelo; o `.gitignore`
  foi ajustado para evitar artefatos pesados.
- Testes locais passaram antes do commit: `Ran 9 tests OK`.

## Como retomar em outro computador

1. Clonar ou atualizar o repositorio.

```bash
git clone https://github.com/GustaRibeiror/desafiodogoverno.git
cd desafiodogoverno
git checkout gustavo
```

2. Ler primeiro:

```text
docs/CONTINUACAO_PODIO.md
```

3. Se for usar outro modelo/Codex, colar este pedido inicial:

```text
Leia e siga docs/CONTINUACAO_PODIO.md e docs/RESUMO_CHAT_OPERACIONAL.md.
Nao refaca A/B. Estamos no hackathon WORCAP/Kaggle. O proximo passo e treinar
final A+B regularized 8000 e gerar artifacts/submission_regularized_ab_8000.csv
sem submeter automaticamente.
```

4. No Kaggle, confirmar que os dados da competicao e o dataset privado com o
   codigo estao anexados antes de rodar o bloco final.

## Proximo passo recomendado

O proximo passo util, quando houver tempo suficiente, e rodar no Kaggle o bloco
de treino final A+B salvo em `docs/CONTINUACAO_PODIO.md`.

Resultado esperado:

```text
/kaggle/working/desafiodogoverno/artifacts/submission_regularized_ab_8000.csv
```

Depois disso, conferir a auditoria impressa:

- `rows = 1885464`
- `unique_ids = 1885464`
- ids na ordem do `sample_submission.csv`
- valores finitos
- nenhum valor negativo

Somente apos essa conferencia o usuario decide se submete no Kaggle.

