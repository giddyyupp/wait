source_dir=$1
fake_dir=$2

python scripts/metrics/calculate_MSE.py -s ${source_dir} -f ${fake_dir}
