python3 download_msmarco_data.py --num_queries 250 --num_docs 100
python3 generate_feature_attribution_explanations_text.py --split validation --top_k 10 --num_queries 250 --num_docs 100
python3 evaluate_rankingshap_text_fidelity.py --split validation --top_k 10 --num_queries 250 --num_docs 100