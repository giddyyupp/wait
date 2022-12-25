MODEL_NAME=$1  # name of the experiment
DATASET=$2  # BP, AS
MAIN_PATH=$3  # wait repo dir
CARTOON_PATH=$MAIN_PATH/results/
FWE_PATH=path_to_fwe_repo

for num in ${@:4:99}
do
	# Build fake dirs
	CARTOON_FAKE="${CARTOON_PATH}${MODEL_NAME}/test_${num}/fake"
	FWE_FAKE="${FWE_PATH}data/test/${MODEL_NAME}/fwe/${DATASET}"
	FWE_MODEL="${FWE_PATH}data/test/${MODEL_NAME}"
	
  #echo ${CARTOON_FAKE}
 	#echo ${FWE_FAKE}
	#echo ${FWE_MODEL}
	
	# Clean up
	rm -rf ${FWE_MODEL}
	
	# Makedir with sub-dirs
	mkdir -p ${FWE_FAKE}
	
	# Link cartoon to fwe path so that it won't use disk space
	ln -s ${CARTOON_FAKE} ${FWE_FAKE}
	
	# Rename and calculate fwe
	python rename_video_frames.py ${MODEL_NAME} ${DATASET} > aaa.txt 2>&1
	#echo "Running Evaluation!!"
  python evaluate_WarpError.py -dataset ${DATASET} -method ${MODEL_NAME} > aa.txt 2>&1
	
	# Get score
	SCORE=$(sed -n '2p' "${FWE_FAKE}/WarpError.txt")
	echo "Epoch: ${num} FWE: ${SCORE}"
	
done
