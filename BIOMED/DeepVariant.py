BUCKET="gs://deepvariant"
BIN_VERSION="0.4.0"
MODEL_VERSION="0.4.0"
MODEL_CL="174375304"

# Note that we don't specify the CL number for the binary, only the bin version.
BIN_BUCKET="${BUCKET}/binaries/DeepVariant/${BIN_VERSION}/DeepVariant-${BIN_VERSION}+cl-*"
MODEL_NAME="DeepVariant-inception_v3-${MODEL_VERSION}+cl-${MODEL_CL}.data-wgs_standard"
MODEL_BUCKET="${BUCKET}/models/DeepVariant/${MODEL_VERSION}/${MODEL_NAME}"
DATA_BUCKET="${BUCKET}/quickstart-testdata"

mkdir -p bin
# Download the DeepVariant binaries.
gsutil -m cp "${BIN_BUCKET}/*" bin/
