mkdir ../output/2years ../output/3years ../output/4years ../output/5years ../output/7years ../output/9years
python stable_test.py --data_path ../raw_data/2.xlsx --output_dir ../output/2years --result_col 2years --preprocess_func miNNseq
python stable_test.py --data_path ../raw_data/3.xlsx --output_dir ../output/3years --result_col 3years --preprocess_func miNNseq
python stable_test.py --data_path ../raw_data/4.xlsx --output_dir ../output/4years --result_col 4years --preprocess_func miNNseq
python stable_test.py --data_path ../raw_data/5.xlsx --output_dir ../output/5years --result_col 5years --preprocess_func miNNseq
python stable_test.py --data_path ../raw_data/7.xlsx --output_dir ../output/7years --result_col 7years --preprocess_func miNNseq
python stable_test.py --data_path ../raw_data/9.xlsx --output_dir ../output/9years --result_col 9years --preprocess_func miNNseq
