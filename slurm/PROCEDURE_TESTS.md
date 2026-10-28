# Procédure de test — un seul rebond, directives MIG impératives

## Principe
Un seul `ssh` vers `hedy`, une seule commande `sbatch`. Tout (arbre + venvs + données)
provient du dossier partagé `/home/projects/m26266/letma2026-ml-labs` : aucun second
rebond vers un nœud de calcul, aucune copie manuelle.

## Étapes

```bash
# 1) Le seul rebond : se connecter au frontal
ssh hedy

# 2) Lancer TOUTES les validations en une commande (3 labs + extension Darcy + 2 notebooks JAX)
sbatch /home/projects/m26266/letma2026-ml-labs/slurm/test_all_mig.sbatch
# -> renvoie "Submitted batch job <ID>"

# 3) Suivre la progression (facultatif)
squeue -u $USER                          # état du job
tail -f /home/projects/m26266/letma2026-ml-labs/letma_tests_<ID>.out

# 4) Vérifier le résultat
grep ALL_DONE /home/projects/m26266/letma2026-ml-labs/letma_tests_<ID>.out   # doit apparaître
grep -c Error  /home/projects/m26266/letma2026-ml-labs/letma_tests_<ID>.err  # doit être 0
sacct -j <ID> --format=State,Elapsed -n                                    # COMPLETED + durée
```

Durée mesurée (dernière validation, 12 min au total sur une instance MIG `1g.20gb`) :
labs 64 s + 39 s + 48 s, extension Darcy ~9 min (200 époques), JAX 7 s + 10 s —
**zéro erreur sur les 6 notebooks**, résultats identiques aux références CPU.

## Directives MIG impératives (déjà incluses dans test_all_mig.sbatch)

```bash
#SBATCH --nodelist=m-gpu[07-08]
#SBATCH --reservation=test_mig_letma
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --account=m26266
```

Ces quatre lignes sont **obligatoires** pour toute soumission sur les instances MIG de
l'école. Le nom du GRES (`nvidia_h100_80gb_hbm3_1g.20gb`) est celui exposé par Slurm ;
une soumission sans lui échoue (ou capte un GPU entier). Les deux nœuds m-gpu07 et
m-gpu08 sont miggués : `--nodelist=m-gpu[07-08]` laisse Slurm choisir le nœud libre.

## Ce que fait le script test_all_mig.sbatch
1. Exécute les 3 notebooks de lab (solutions) avec `env/fno-gnn` (venv partagé).
2. Exécute l'extension Darcy (FFNO2d, 200 époques) — le plus long (~9 min sur MIG).
3. Exécute les 2 notebooks JAX avec `env/jax-hedy`.
4. Écrit `ALL_DONE` à la fin ; toute erreur est dans `letma_tests_<ID>.err`.
Les notebooks exécutés (avec toutes leurs sorties) restent dans le dossier partagé :
ils peuvent être relus directement ou copiés vers un home personnel.
