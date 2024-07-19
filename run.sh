# Activate the Conda environment
conda activate figureClass
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Log in to Hugging Face using the token
huggingface-cli login --token $HUGGINGFACE_TOKEN
SEG_PATH=/nfs/turbo/umms-drjieliu/proj/medlineKG/ckpts/model_final.pth
CLASSIFIER_PATH=/nfs/turbo/umms-drjieliu/proj/medlineKG/ckpts/classifier_acl_aug_resume.pt
INPUT_DIR=/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/chunks
OUTPUT_DIR=/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/chunk_out

# Print out the Python version and environment details
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Installed packages: $(pip list)"

# Run your Python script with the Hugging Face token
python json_separation_and_pred.py --segment_model_path $SEG_PATH --classify_model_path $CLASSIFIER_PATH --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR

# Deactivate the Conda environment
conda deactivate