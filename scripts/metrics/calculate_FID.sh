source_dir=$1
fake_dir=$2

echo ${source_dir}
echo ${fake_dir}

python -m pytorch_fid ${source_dir} ${fake_dir}
