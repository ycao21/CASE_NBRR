# # PIE============
# # inst_new_pie_bce
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/instacart_small/test --tag run_20260222_185816_inst_new_pie_bce --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_185816_inst_new_pie_bce/checkpoints/best.pt

# # dc_pie_bce
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/dc/test --tag run_20260222_190644_dc_pie_bce --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_190644_dc_pie_bce/checkpoints/best.pt

# # tafeng_pie_bce
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/tafeng/test --tag run_20260222_194946_tafeng_pie_bce --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_194946_tafeng_pie_bce/checkpoints/best.pt

# # walmart_pie_bce
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260222_230417_walmart_pie_bce --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_230417_walmart_pie_bce/checkpoints/best.pt

# # ranker============
# # inst_new_settran_pma_ranker
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/instacart_small/test --tag run_20260222_195308_inst_new_settran_pma_ranker --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_195308_inst_new_settran_pma_ranker/checkpoints/best.pt

# # dc_settran_pma_ranker
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/dc/test --tag run_20260222_200430_dc_settran_pma_ranker --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_200430_dc_settran_pma_ranker/checkpoints/best.pt

# # tafeng_settran_pma_ranker
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/tafeng/test --tag run_20260222_211036_tafeng_settran_pma_ranker --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_211036_tafeng_settran_pma_ranker/checkpoints/best.pt

# # walmart_settran_pma_ranker
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260222_231456_walmart_settran_pma_ranker --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_231456_walmart_settran_pma_ranker/checkpoints/best.pt

# # BCE 
# # inst_new_settran_pma_bce
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/instacart_small/test --tag run_20260222_211506_inst_new_settran_pma_bce --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_211506_inst_new_settran_pma_bce/checkpoints/best.pt

# # dc_settran_pma_bce
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/dc/test --tag run_20260222_212624_dc_settran_pma_bce --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_212624_dc_settran_pma_bce/checkpoints/best.pt

# # tafeng_settran_pma_bce
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/tafeng/test --tag run_20260222_223206_tafeng_settran_pma_bce --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_223206_tafeng_settran_pma_bce/checkpoints/best.pt

# # walmart_settran_pma_bce
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260222_232807_walmart_settran_pma_bce --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_232807_walmart_settran_pma_bce/checkpoints/best.pt

# # ablation============
# # walmart_no_cnn
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260222_234108_walmart_no_cnn --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_234108_walmart_no_cnn/checkpoints/best.pt

# # walmart_no_pma
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260222_235330_walmart_no_pma --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260222_235330_walmart_no_pma/checkpoints/best.pt

# # walmart_no_settranpma
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260223_000609_walmart_no_settranpma --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260223_000609_walmart_no_settranpma/checkpoints/best.pt


# # walmart_no_settranpma
# uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260223_013120_walmart_no_settran --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260223_013120_walmart_no_settran/checkpoints/best.pt



uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260223_202135_walmart_intrinsic_only --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260223_202135_walmart_intrinsic_only/checkpoints/best.pt

uv run python -m src.inference --config configs/inference.yaml --local --inference_data_path gs://p13n-storage2/user/y0c07th/pb_public_data/cnntsp/walmart/test --tag run_20260223_203634_walmart_compatibility_only --checkpoint /home/yanan_cao_walmart_com/work/PB_cnn_sab_spf/cnn_st_spf_nbr/experiments/run_20260223_203634_walmart_compatibility_only/checkpoints/best.pt
