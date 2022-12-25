wait_dir=$1  # path to wait repo
EXP_ID=$2  # name of the experiment
dataset=$3  # name of the dataset (folder name)
fwe_dir=$4  # path to FWE repo

set -ex

### 2. Build Fake folders
scripts/build_fake.sh "$wait_dir/results/$EXP_ID" $dataset

### 3. Test MSE
scripts/metrics/calculate_MSE_batch.sh "$wait_dir/datasets/$dataset/testA/" "$wait_dir/results/$EXP_ID" $dataset

### 4. Test FID
scripts/metrics/calculate_FID_batch.sh "$wait_dir/datasets/{$dataset}/trainB/" "$wait_dir/results/$EXP_ID" $dataset

# 5. Test FWE
cd $fwe_dir
./calculate_FWE.sh $EXP_ID $dataset $wait_dir 100 110 120 130 140 150 160 170 180 190 200
