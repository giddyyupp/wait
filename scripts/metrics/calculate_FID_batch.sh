source_dir=$1
result_dir=$2

bash scripts/metrics/calculate_FID.sh ${source_dir} ${result_dir}/test_50/fake 
bash scripts/metrics/calculate_FID.sh ${source_dir} ${result_dir}/test_80/fake
bash scripts/metrics/calculate_FID.sh ${source_dir} ${result_dir}/test_100/fake
bash scripts/metrics/calculate_FID.sh ${source_dir} ${result_dir}/test_110/fake
#bash scripts/calculate_fid.sh ${source_dir} ${result_dir}/test_150/fake
#bash scripts/calculate_fid.sh ${source_dir} ${result_dir}/test_180/fake
#bash scripts/calculate_fid.sh ${source_dir} ${result_dir}/test_200/fake
