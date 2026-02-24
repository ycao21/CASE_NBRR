
# # echo "Running inst_new_pie_bce"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_pie_instacart.yaml --local --tag inst_new_pie_bce > logs/inst_new_pie_bce.log 2>&1

# # echo "Running dc_pie_bce"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_pie_dc.yaml --local --tag dc_pie_bce > logs/dc_pie_bce.log 2>&1

# # echo "Running tafeng_pie_bce"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_pie_tafeng.yaml --local --tag tafeng_pie_bce > logs/tafeng_pie_bce.log 2>&1

# echo "Running walmart_pie_bce"
# uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_pie_walmart.yaml --local --tag walmart_pie_bce > logs/walmart_pie_bce.log 2>&1

# echo "================================================"

# # echo "Running inst_new_settran_pma_ranker"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_instacart_ranker.yaml --local --tag inst_new_settran_pma_ranker > logs/inst_new_settran_pma_ranker.log 2>&1

# # echo "Running dc_settran_pma_ranker"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_dc_ranker.yaml --local --tag dc_settran_pma_ranker > logs/dc_settran_pma_ranker.log 2>&1

# # echo "Running tafeng_settran_pma_ranker"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_tafeng_ranker.yaml --local --tag tafeng_settran_pma_ranker > logs/tafeng_settran_pma_ranker.log 2>&1

# echo "Running walmart_settran_pma_ranker"
# uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_walmart_ranker.yaml --local --tag walmart_settran_pma_ranker > logs/walmart_settran_pma_ranker.log 2>&1

# echo "================================================"

# # echo "Running inst_new_settran_pma_bce"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_instacart.yaml --local --tag inst_new_settran_pma_bce > logs/inst_new_settran_pma_bce.log 2>&1

# # echo "Running dc_settran_pma_bce"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_dc.yaml --local --tag dc_settran_pma_bce > logs/dc_settran_pma_bce.log 2>&1

# # echo "Running tafeng_settran_pma_bce"
# # uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_tafeng.yaml --local --tag tafeng_settran_pma_bce > logs/tafeng_settran_pma_bce.log 2>&1

# echo "Running walmart_settran_pma_bce"
# uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_walmart.yaml --local --tag walmart_settran_pma_bce > logs/walmart_settran_pma_bce.log 2>&1

# echo "================================================"

# echo "Running walmart_no_cnn (ablation)"
# uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_walmart_no_cnn.yaml --local --tag walmart_no_cnn > logs/walmart_no_cnn.log 2>&1

# echo "Running walmart_no_pma (ablation)"
# uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_walmart_no_pma.yaml --local --tag walmart_no_pma > logs/walmart_no_pma.log 2>&1

# echo "Running walmart_no_settranpma (ablation)"
# uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_walmart_no_settranpma.yaml --local --tag walmart_no_settranpma > logs/walmart_no_settranpma.log 2>&1

# echo "Running walmart_no_settran (ablation)"
# uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_walmart_no_settran.yaml --local --tag walmart_no_settran > logs/walmart_no_settran.log 2>&1


# # gsutil cp -r experiments gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/experiment_0222_new2


echo "Running walmart_intrinsic_only (ablation)"
uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_walmart_scoring_intrinsic.yaml --local --tag walmart_intrinsic_only > logs/walmart_intrinsic_only.log 2>&1

echo "Running walmart_compatibility_only (ablation)"
uv run python -m src.runner --config configs/local_train_cnn_1024seq_hard_settran_pma_walmart_scoring_compatibility.yaml --local --tag walmart_compatibility_only > logs/walmart_compatibility_only.log 2>&1
