res_folder=$1
dataset=$2

echo ${pwd}

if [ "$dataset" == "axel" ]
then

echo "Build Fake: Axel"

# axels
mkdir -p ${res_folder}/test_50/fake && cp ${res_folder}/test_50/images/*_fake_B_1.png ${res_folder}/test_50/fake/
mkdir -p ${res_folder}/test_80/fake && cp ${res_folder}/test_80/images/*_fake_B_1.png ${res_folder}/test_80/fake/
mkdir -p ${res_folder}/test_90/fake && cp ${res_folder}/test_90/images/*_fake_B_1.png ${res_folder}/test_90/fake/
mkdir -p ${res_folder}/test_100/fake && cp ${res_folder}/test_100/images/*_fake_B_1.png ${res_folder}/test_100/fake/
mkdir -p ${res_folder}/test_110/fake && cp ${res_folder}/test_110/images/*_fake_B_1.png ${res_folder}/test_110/fake/
mkdir -p ${res_folder}/test_120/fake && cp ${res_folder}/test_120/images/*_fake_B_1.png ${res_folder}/test_120/fake/
mkdir -p ${res_folder}/test_130/fake && cp ${res_folder}/test_130/images/*_fake_B_1.png ${res_folder}/test_130/fake/
mkdir -p ${res_folder}/test_140/fake && cp ${res_folder}/test_140/images/*_fake_B_1.png ${res_folder}/test_140/fake/
mkdir -p ${res_folder}/test_150/fake && cp ${res_folder}/test_150/images/*_fake_B_1.png ${res_folder}/test_150/fake/
mkdir -p ${res_folder}/test_160/fake && cp ${res_folder}/test_160/images/*_fake_B_1.png ${res_folder}/test_160/fake/
mkdir -p ${res_folder}/test_170/fake && cp ${res_folder}/test_170/images/*_fake_B_1.png ${res_folder}/test_170/fake/
mkdir -p ${res_folder}/test_180/fake && cp ${res_folder}/test_180/images/*_fake_B_1.png ${res_folder}/test_180/fake/
mkdir -p ${res_folder}/test_190/fake && cp ${res_folder}/test_190/images/*_fake_B_1.png ${res_folder}/test_190/fake/
mkdir -p ${res_folder}/test_200/fake && cp ${res_folder}/test_200/images/*_fake_B_1.png ${res_folder}/test_200/fake/

else

echo "Build Fake: Peter"

# peters
mkdir -p ${res_folder}/test_50/fake && cp ${res_folder}/test_50/images/*_fake_B_1.png ${res_folder}/test_50/fake/
mkdir -p ${res_folder}/test_80/fake && cp ${res_folder}/test_80/images/*_fake_B_1.png ${res_folder}/test_80/fake/
mkdir -p ${res_folder}/test_100/fake && cp ${res_folder}/test_100/images/*_fake_B_1.png ${res_folder}/test_100/fake/
mkdir -p ${res_folder}/test_120/fake && cp ${res_folder}/test_120/images/*_fake_B_1.png ${res_folder}/test_120/fake/
mkdir -p ${res_folder}/test_150/fake && cp ${res_folder}/test_150/images/*_fake_B_1.png ${res_folder}/test_150/fake/
mkdir -p ${res_folder}/test_180/fake && cp ${res_folder}/test_180/images/*_fake_B_1.png ${res_folder}/test_180/fake/
mkdir -p ${res_folder}/test_200/fake && cp ${res_folder}/test_200/images/*_fake_B_1.png ${res_folder}/test_200/fake/
mkdir -p ${res_folder}/test_220/fake && cp ${res_folder}/test_220/images/*_fake_B_1.png ${res_folder}/test_220/fake/
mkdir -p ${res_folder}/test_240/fake && cp ${res_folder}/test_240/images/*_fake_B_1.png ${res_folder}/test_240/fake/
mkdir -p ${res_folder}/test_250/fake && cp ${res_folder}/test_250/images/*_fake_B_1.png ${res_folder}/test_250/fake/
mkdir -p ${res_folder}/test_270/fake && cp ${res_folder}/test_270/images/*_fake_B_1.png ${res_folder}/test_270/fake/
mkdir -p ${res_folder}/test_290/fake && cp ${res_folder}/test_290/images/*_fake_B_1.png ${res_folder}/test_290/fake/
mkdir -p ${res_folder}/test_300/fake && cp ${res_folder}/test_300/images/*_fake_B_1.png ${res_folder}/test_300/fake/

fi
