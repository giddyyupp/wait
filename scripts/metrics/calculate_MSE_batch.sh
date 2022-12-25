source_dir=$1  # path to original animation test images (e.g. ./datasets/AS/testA/")
result_dir=$2  # path to generated images (e.g. ./results/AS/)

bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_100/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_110/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_120/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_130/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_140/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_150/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_160/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_170/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_180/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_190/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_200/fake
