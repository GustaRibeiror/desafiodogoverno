# Protocolo de experimentos

1. Não alterar folds, gates ou métrica depois de observar o leaderboard.
2. Executar `wide`, `regularized` e `shallow` na amostra espacial `a`.
3. Promover uma configuração somente pelos três folds causais.
4. Repetir apenas a vencedora na amostra `b`.
5. Aceitar média `a/b` somente se melhorar todos os folds.
6. Calibrar a amplitude por horizonte com OOF leave-one-fold-out.
7. Rodar a U-Net primeiro em 2015–2016 e aplicar os gates documentados.
8. Treinar o final somente depois de congelar configuração, features e pesos.
9. Auditar e guardar hashes antes de qualquer upload.
10. Solicitar confirmação explícita do responsável antes de gastar tentativa.

O registro canônico é `experiments/registry.csv`. Resultados não registrados não
podem justificar a seleção final.
