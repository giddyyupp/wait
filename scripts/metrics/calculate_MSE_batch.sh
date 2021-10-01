source_dir=$1
result_dir=$2
dataset=$3


if [ "$dataset" == "axel" ]:
then

echo "MSE: Running Axel"
# axels
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_50/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_80/fake 
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_90/fake 
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_100/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_110/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_120/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_130/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_140/fake

else

echo "MSE: Running Peter"

# peters
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_50/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_80/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_100/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_120/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_150/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_180/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_200/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_220/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_240/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_250/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_270/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_290/fake
bash scripts/metrics/calculate_MSE.sh ${source_dir} ${result_dir}/test_300/fake

fi
