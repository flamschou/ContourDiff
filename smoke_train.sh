#!/usr/bin/env bash
# smoke_train.sh — entraînement de vérification (3 epochs, vraies données)
#
# Objectif : valider que toute la pipeline fonctionne de bout en bout
# (chargement, augmentation, forward pass, loss, sauvegarde) sans attendre
# un vrai entraînement complet.
#
# Usage :
#   bash smoke_train.sh                      # chemins JSON tels quels
#   bash smoke_train.sh /mnt/data            # si les données sont montées ailleurs
#
# Adapter DATA_ROOT si les fichiers .mat ne sont pas à l'emplacement du split.json.
# Laisser vide si les chemins du JSON sont déjà corrects sur cette machine.

set -euo pipefail

SPLIT_JSON="split.json"
DATA_ROOT="${1:-}"          # 1er argument optionnel = nouveau préfixe des chemins
OUTPUT_DIR="runs/smoke_test_real"

DATA_ROOT_ARG=""
if [ -n "$DATA_ROOT" ]; then
    DATA_ROOT_ARG="--data_root $DATA_ROOT"
fi

python train.py \
    --split_json     "$SPLIT_JSON"   \
    $DATA_ROOT_ARG                   \
    --output_dir     "$OUTPUT_DIR"   \
    --model_type     ddpm            \
    --img_size       64              \
    --num_epochs     3               \
    --train_batch_size  4            \
    --eval_batch_size   4            \
    --noise_step        1000         \
    --learning_rate     1e-4         \
    --lr_warmup_steps   50           \
    --save_image_epochs 1            \
    --save_model_epochs 1            \
    --contour_guided                 \
    --near_guided                    \
    --contour_channel_mode multi     \
    --workers        0               \
    --seed           42

echo ""
echo "Smoke test terminé. Sorties dans : $OUTPUT_DIR"
echo "  - monitor.html       : courbes de loss"
echo "  - samples/           : images générées à chaque epoch"
echo "  - model_epoch_{1,2,3}: checkpoints"
